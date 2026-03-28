from typing import Dict, List, Any

def _priority_score(predictions: Dict[str, int], ground_truths: Dict[str, int]) -> float:
    if not ground_truths:
        return 0.0
    total_error = 0.0
    for email_id, true_p in ground_truths.items():
        pred_p = predictions.get(email_id, 3)
        pred_p = max(1, min(5, int(pred_p)))
        total_error += abs(pred_p - true_p)
    mean_error = total_error / len(ground_truths)
    score = max(0.0, 1.0 - (mean_error / 4.0))
    return round(score * 0.6, 4)

def _reply_score(replies: Dict[str, str], top3_ids: List[str], reply_keywords: Dict[str, List[str]]) -> float:
    if not top3_ids:
        return 0.0
    total = 0.0
    for email_id in top3_ids:
        reply = (replies.get(email_id) or "").lower()
        keywords = reply_keywords.get(email_id, [])
        if not keywords:
            total += 1.0
            continue
        hits = sum(1 for kw in keywords if kw.lower() in reply)
        total += hits / len(keywords)
    return round((total / len(top3_ids)) * 0.4, 4)

def grade_medium(priority_predictions: Dict[str, int], reply_predictions: Dict[str, str], inbox_data: List[Dict[str, Any]]) -> float:
    ground_truth_priorities = {e["email_id"]: e["priority"] for e in inbox_data}
    reply_keyword_map = {e["email_id"]: e.get("reply_keywords", []) for e in inbox_data}
    top3 = sorted([e["email_id"] for e in inbox_data if e["priority"] == 1])[:3]
    p_score = _priority_score(priority_predictions, ground_truth_priorities)
    r_score = _reply_score(reply_predictions, top3, reply_keyword_map)
    return round(min(1.0, p_score + r_score), 4)
