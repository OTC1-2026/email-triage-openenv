from typing import Dict, Any, List

VALID_ACTIONS = {"escalate", "archive", "reply", "snooze", "skip"}
HIGH_PRIORITY_THRESHOLD = 1
SLA_SENSITIVE_MINUTES = 30

def grade_hard(actions_taken: Dict[str, Dict[str, Any]], inbox_data: List[Dict[str, Any]]) -> float:
    raw_score = 0.0
    max_achievable = 0.0
    for email in inbox_data:
        eid = email["email_id"]
        correct_action = email["correct_action"]
        priority = email["priority"]
        sla = email["sla_deadline_minutes"]
        is_high_priority = (priority == HIGH_PRIORITY_THRESHOLD)
        is_sla_sensitive = (sla <= SLA_SENSITIVE_MINUTES)
        potential = 0.8 + (0.2 if is_sla_sensitive else 0.0)
        max_achievable += potential
        taken = actions_taken.get(eid)
        if taken is None:
            if is_high_priority:
                raw_score -= 0.1
            elif priority in (2, 3):
                raw_score -= 0.05
            continue
        action_type = (taken.get("action") or "skip").lower()
        step_taken = int(taken.get("step", 20))
        if action_type == correct_action:
            raw_score += 0.8
        elif is_high_priority and action_type not in ("escalate", "reply"):
            raw_score -= 0.3
        if is_sla_sensitive and action_type == correct_action:
            if step_taken <= 10:
                raw_score += 0.2
            else:
                raw_score += 0.1
    if max_achievable <= 0:
        return 0.0
    return round(max(0.0, min(1.0, raw_score / max_achievable)), 4)
