from __future__ import annotations
import uuid, random
from typing import Any, Dict, List, Optional, Tuple
from models import EmailTriageAction, EmailTriageObservation, EmailItem, EpisodeState
from data.emails import EASY_EMAILS, MEDIUM_INBOX, HARD_INBOX
from graders.grader_easy import grade_easy
from graders.grader_medium import grade_medium
from graders.grader_hard import grade_hard

class EmailTriageEnvironment:
    VALID_TASKS = ("easy", "medium", "hard")

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}")
        self._task = task
        self._rng = random.Random(seed)
        self._state = EpisodeState()
        self._pp: Dict[str, int] = {}
        self._rp: Dict[str, str] = {}
        self._ha: Dict[str, Dict[str, Any]] = {}
        self._cur = None
        self._inbox: List[Dict[str, Any]] = []

    def reset(self) -> EmailTriageObservation:
        self._state = EpisodeState(
            episode_id=str(uuid.uuid4()), task_name=self._task,
            step_count=0, max_steps=self._max_steps())
        self._pp = {}; self._rp = {}; self._ha = {}
        if self._task == "easy":
            self._cur = self._rng.choice(EASY_EMAILS)
            return self._easy_obs(0.0, False)
        elif self._task == "medium":
            self._inbox = list(MEDIUM_INBOX)
            return self._medium_obs(0.0, False)
        else:
            self._inbox = list(HARD_INBOX)
            return self._hard_obs(0.0, False)

    def step(self, action: EmailTriageAction):
        if self._state.done:
            raise RuntimeError("Episode done. Call reset().")
        self._state.step_count += 1
        reward = 0.0
        if self._task == "easy":
            gt = self._cur["category"]
            reward = grade_easy(action.category or "", gt)
            self._state.cumulative_reward = reward
            self._state.done = True
            return self._easy_obs(reward, True, {"ground_truth": gt}), reward, True, {"ground_truth": gt}
        elif self._task == "medium":
            eid = action.email_id or ""
            if action.action_type == "prioritize" and eid:
                self._pp[eid] = action.priority or 3
            elif action.action_type == "reply" and eid:
                self._rp[eid] = action.reply_text or ""
            done = len(self._pp) == len(self._inbox) or self._state.step_count >= self._state.max_steps
            if done:
                final = grade_medium(self._pp, self._rp, self._inbox)
                self._state.cumulative_reward = final
                self._state.done = True
                return self._medium_obs(final, True), final, True, {"final_score": final}
            return self._medium_obs(0.0, False), 0.0, False, {}
        else:
            eid = action.email_id or ""
            atype = action.action_type or "skip"
            if eid and eid not in self._ha:
                self._ha[eid] = {"action": atype, "step": self._state.step_count}
                ed = next((e for e in self._inbox if e["email_id"] == eid), None)
                if ed and atype == ed["correct_action"]: reward = 0.05
                elif ed and ed["priority"] == 1 and atype not in ("escalate", "reply"): reward = -0.05
            self._state.cumulative_reward += reward
            done = self._state.step_count >= self._state.max_steps
            if done:
                final = grade_hard(self._ha, self._inbox)
                self._state.cumulative_reward = final
                self._state.done = True
                return self._hard_obs(final, True), final, True, {"final_score": final}
            return self._hard_obs(reward, False), reward, False, {}

    @property
    def state(self):
        return {"episode_id": self._state.episode_id, "task_name": self._state.task_name,
                "step_count": self._state.step_count, "max_steps": self._state.max_steps,
                "done": self._state.done, "cumulative_reward": self._state.cumulative_reward}

    def _max_steps(self):
        return {"easy": 1, "medium": 20, "hard": 20}[self._task]

    def _easy_obs(self, r, d, info={}):
        e = self._cur
        return EmailTriageObservation(
            current_email=EmailItem(email_id=e["email_id"], sender=e["sender"],
                subject=e["subject"], body=e["body"], timestamp=e["timestamp"]),
            step_count=self._state.step_count, max_steps=self._state.max_steps,
            episode_id=self._state.episode_id, task_name="easy",
            instructions="Classify: spam|urgent|billing|support|fyi",
            reward=r, done=d, info=info)

    def _medium_obs(self, r, d):
        items = [EmailItem(email_id=e["email_id"], sender=e["sender"],
            subject=e["subject"], body=e["body"], timestamp=e["timestamp"])
            for e in self._inbox]
        return EmailTriageObservation(inbox=items, step_count=self._state.step_count,
            max_steps=self._state.max_steps, episode_id=self._state.episode_id,
            task_name="medium", instructions="Prioritize 1-5. Reply for top-3 urgent.",
            reward=r, done=d, info={"priorities_set": len(self._pp), "replies_drafted": len(self._rp)})

    def _hard_obs(self, r, d):
        acted = set(self._ha.keys())
        rem = [EmailItem(email_id=e["email_id"], sender=e["sender"],
            subject=e["subject"], body=e["body"], timestamp=e["timestamp"],
            sla_deadline_minutes=e["sla_deadline_minutes"])
            for e in self._inbox if e["email_id"] not in acted]
        return EmailTriageObservation(inbox=rem, step_count=self._state.step_count,
            max_steps=self._state.max_steps, episode_id=self._state.episode_id,
            task_name="hard", instructions="escalate|archive|reply|snooze|skip",
            reward=r, done=d, info={"emails_remaining": len(rem),
            "emails_actioned": len(acted), "cumulative_reward": self._state.cumulative_reward})
