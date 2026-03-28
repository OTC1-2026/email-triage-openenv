from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from models import EmailTriageAction, EmailTriageObservation
from server.environment import EmailTriageEnvironment

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs = {
    "easy": EmailTriageEnvironment("easy"),
    "medium": EmailTriageEnvironment("medium"),
    "hard": EmailTriageEnvironment("hard"),
}
_active_task = "easy"

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: EmailTriageAction

class GraderRequest(BaseModel):
    task: Optional[str] = "easy"

@app.get("/health")
def health():
    return {"status": "ok", "environment": "email-triage-openenv"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _active_task
    task = req.task or "easy"
    if task not in _envs:
        raise HTTPException(400, f"task must be one of: {list(_envs.keys())}")
    _active_task = task
    _envs[task] = EmailTriageEnvironment(task, seed=req.seed)
    obs = _envs[task].reset()
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = _envs[_active_task].step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return _envs[_active_task].state

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"name": "easy", "difficulty": "easy", "max_steps": 1, "score_range": [0.0, 1.0],
         "action_schema": {"action_type": "classify", "category": "spam|urgent|billing|support|fyi"}},
        {"name": "medium", "difficulty": "medium", "max_steps": 20, "score_range": [0.0, 1.0],
         "action_schema": {"action_type": "prioritize|reply", "email_id": "str", "priority": "1-5"}},
        {"name": "hard", "difficulty": "hard", "max_steps": 20, "score_range": [0.0, 1.0],
         "action_schema": {"action_type": "escalate|archive|reply|snooze|skip", "email_id": "str"}},
    ]}

@app.post("/grader")
def grader(req: GraderRequest = GraderRequest()):
    task = req.task or _active_task
    s = _envs[task].state
    if not s["done"]:
        raise HTTPException(400, "Episode not finished.")
    return {"task": task, "episode_id": s["episode_id"], "score": s["cumulative_reward"], "steps": s["step_count"]}

@app.get("/baseline")
def baseline():
    from graders.grader_easy import grade_easy
    from graders.grader_medium import grade_medium
    from graders.grader_hard import grade_hard
    from data.emails import EASY_EMAILS, MEDIUM_INBOX, HARD_INBOX
    easy_s = round(sum(grade_easy("urgent", e["category"]) for e in EASY_EMAILS) / len(EASY_EMAILS), 4)
    urgent_kw = ["urgent","critical","crash","down","breach","deadline","emergency"]
    ph, rh = {}, {}
    for e in MEDIUM_INBOX:
        b = (e["subject"] + " " + e["body"]).lower()
        p = 1 if any(kw in b for kw in urgent_kw) else 3
        ph[e["email_id"]] = p
        if p == 1:
            rh[e["email_id"]] = "Our team is working immediately to fix and escalate this as highest priority."
    med_s = grade_medium(ph, rh, MEDIUM_INBOX)
    ha = {}
    for i, e in enumerate(HARD_INBOX):
        a = "escalate" if e["priority"]==1 else ("archive" if e["priority"]==5 else "snooze")
        ha[e["email_id"]] = {"action": a, "step": i+1}
    hard_s = grade_hard(ha, HARD_INBOX)
    return {"baseline_agent": "keyword-heuristic-v1", "seed": 42,
            "scores": {"easy": easy_s, "medium": med_s, "hard": hard_s},
            "note": "Deterministic. Reproducible across runs."}

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
