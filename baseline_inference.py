#!/usr/bin/env python3
import os, sys, json, argparse, requests
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:7860"
MODEL = "gpt-4o-mini"
SEED = 42

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    return OpenAI(api_key=api_key)

def env_reset(base_url, task, seed=SEED):
    r = requests.post(f"{base_url}/reset", json={"task": task, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(base_url, action):
    r = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grader(base_url, task):
    r = requests.post(f"{base_url}/grader", json={"task": task}, timeout=15)
    r.raise_for_status()
    return r.json()

def llm_call(client, system, user):
    resp = client.chat.completions.create(
        model=MODEL, seed=SEED, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def run_easy(client, base_url):
    print("\n── EASY TASK ─────────────────────────────────────")
    obs = env_reset(base_url, "easy")
    email = obs.get("current_email", {})
    prompt = (f"Subject: {email.get('subject')}\nFrom: {email.get('sender')}\n"
              f"Body: {email.get('body')}\n\nClassify: spam, urgent, billing, support, or fyi.\n"
              f"Respond with JSON: {{\"category\": \"<choice>\"}}")
    system = "You are an expert email classifier. Output valid JSON only."
    raw = llm_call(client, system, prompt)
    parsed = json.loads(raw)
    category = parsed.get("category", "fyi").lower().strip()
    print(f"  Classified as: {category}")
    result = env_step(base_url, {"action_type": "classify", "category": category})
    score = result["reward"]
    print(f"  Score: {score}")
    return score

def run_medium(client, base_url):
    print("\n── MEDIUM TASK ───────────────────────────────────")
    obs = env_reset(base_url, "medium")
    inbox = obs.get("inbox", [])
    inbox_text = "\n".join(
        f"[{e['email_id']}] From: {e['sender']} | Subject: {e['subject']}"
        for e in inbox)
    system = "You are an expert email triage assistant. Output valid JSON only."
    prompt = (f"Inbox:\n{inbox_text}\n\nOutput JSON:\n"
              '{"priorities": {"<email_id>": <1-5>}, "replies": {"<email_id>": "<reply>"}}')
    raw = llm_call(client, system, prompt)
    parsed = json.loads(raw)
    priorities = parsed.get("priorities", {})
    replies = parsed.get("replies", {})
    done = False
    for eid, txt in list(replies.items())[:3]:
        result = env_step(base_url, {"action_type": "reply", "email_id": eid, "reply_text": txt})
        done = result["done"]
        if done: break
    if not done:
        for eid, prio in priorities.items():
            result = env_step(base_url, {"action_type": "prioritize", "email_id": eid, "priority": int(prio)})
            done = result["done"]
            if done: break
    grader_result = env_grader(base_url, "medium")
    score = grader_result["score"]
    print(f"  Score: {score}")
    return score

def run_hard(client, base_url):
    print("\n── HARD TASK ─────────────────────────────────────")
    obs = env_reset(base_url, "hard")
    done = obs.get("done", False)
    step = 0
    while not done and step < 20:
        inbox = obs.get("inbox", [])
        if not inbox: break
        inbox_text = "\n".join(
            f"[{e['email_id']}] SLA:{e.get('sla_deadline_minutes','?')}min | {e['subject'][:60]}"
            for e in inbox[:10])
        system = ("Manage inbox. Choose: escalate|archive|reply|snooze|skip. "
                  "Urgent+short SLA=escalate. Spam=archive. Output JSON only.")
        prompt = (f"Inbox ({len(inbox)} emails):\n{inbox_text}\n\n"
                  'Pick ONE: {"email_id":"<id>","action_type":"<action>"}')
        raw = llm_call(client, system, prompt)
        parsed = json.loads(raw)
        result = env_step(base_url, {"action_type": parsed.get("action_type","skip"),
                                      "email_id": parsed.get("email_id","")})
        obs = result.get("observation", {})
        done = result["done"]
        step += 1
        print(f"  Step {step}: {parsed.get('email_id')} → {parsed.get('action_type')} | reward={result['reward']:.3f}")
    grader_result = env_grader(base_url, "hard")
    score = grader_result["score"]
    print(f"  Score: {score}")
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tasks", nargs="+", default=["easy","medium","hard"])
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    client = get_client()
    print(f"Email Triage OpenEnv — Baseline | Model: {MODEL} | Seed: {SEED}")
    scores = {}
    if "easy" in args.tasks: scores["easy"] = run_easy(client, base_url)
    if "medium" in args.tasks: scores["medium"] = run_medium(client, base_url)
    if "hard" in args.tasks: scores["hard"] = run_hard(client, base_url)
    print("\n" + "="*50)
    print("BASELINE SCORES:")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    return scores

if __name__ == "__main__":
    main()
