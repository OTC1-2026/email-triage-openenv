import requests

BASE_URL = "https://otc1-2026-email-triage-openenv.hf.space"

def run_episode():
    r = requests.post(f"{BASE_URL}/reset", json={"task": "easy"})
    obs = r.json()["observation"]

    email = obs["current_email"]
    subject = email["subject"].lower()

    if "release" in subject:
        category = "notification"
    else:
        category = "newsletter"

    action = {
        "action": {
            "action_type": "classify",
            "category": category,
            "priority": 0,
            "reply_text": "",
            "email_id": email["email_id"]
        }
    }

    step = requests.post(f"{BASE_URL}/step", json=action)
    result = step.json()

    return result

if __name__ == "__main__":
    print(run_episode())
