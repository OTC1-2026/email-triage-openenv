import requests

BASE = "https://otc1-2026-email-triage-openenv.hf.space"

def run():
    r = requests.post(f"{BASE}/reset", json={"task": "easy"})
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

    result = requests.post(f"{BASE}/step", json=action)
    print(result.json())

if __name__ == "__main__":
    run()
