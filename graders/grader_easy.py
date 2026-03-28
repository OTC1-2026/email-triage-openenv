VALID_CATEGORIES = {"spam", "urgent", "billing", "support", "fyi"}

ADJACENCY = {
    ("urgent", "support"): 0.3,
    ("support", "urgent"): 0.3,
    ("billing", "support"): 0.3,
    ("support", "billing"): 0.3,
    ("fyi", "support"): 0.2,
    ("support", "fyi"): 0.2,
}

def grade_easy(predicted_category: str, ground_truth_category: str) -> float:
    predicted_category = (predicted_category or "").strip().lower()
    ground_truth_category = (ground_truth_category or "").strip().lower()
    if predicted_category not in VALID_CATEGORIES:
        return 0.0
    if predicted_category == ground_truth_category:
        return 1.0
    return ADJACENCY.get((predicted_category, ground_truth_category), 0.0)
