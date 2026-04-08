from __future__ import annotations
import json
import os
import sys
from openai import OpenAI

# ✅ FIXED IMPORTS (IMPORTANT)
from env import SupportTicketEnv
from models import Action
from tasks import list_tasks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

API_KEY = HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert customer support triage agent.

You must:
- classify ticket (billing, technical, account)
- assign priority (low, medium, high)
- write a helpful response
- optionally ask a follow-up question

Return ONLY valid JSON:

{
  "predicted_category": "...",
  "predicted_priority": "...",
  "response_message": "...",
  "optional_followup_question": null
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_message(obs_dict: dict) -> str:
    msg = f"""Ticket ID: {obs_dict['ticket_id']}
Ticket: {obs_dict['ticket_text']}
Step: {obs_dict['step']} / {obs_dict['max_steps']}"""

    return msg


def parse_action(content: str):
    try:
        content = content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)

        action = Action(
            predicted_category=data.get("predicted_category", "technical"),
            predicted_priority=data.get("predicted_priority", "medium"),
            response_message=data.get("response_message", ""),
            optional_followup_question=data.get("optional_followup_question"),
        )

        return action, None

    except Exception as e:
        fallback = Action(
            predicted_category="technical",
            predicted_priority="medium",
            response_message="Sorry, I will check this issue.",
            optional_followup_question=None,
        )
        return fallback, str(e)


# ---------------------------------------------------------------------------
# Core Episode
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str):
    env = SupportTicketEnv()

    print(f"[START] task={task_name} env=SupportTicketEnv model={MODEL_NAME}", flush=True)

    obs = env.reset(task_name)
    obs_dict = obs.model_dump()

    rewards = []
    steps_taken = 0
    success = False

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs_dict["done"]:
        user_msg = build_user_message(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            content = ""

        messages.append({"role": "assistant", "content": content})

        action, parse_error = parse_action(content)

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()

        steps_taken += 1
        rewards.append(reward.total)

        if info.get("correct_resolution"):
            success = True

        action_summary = json.dumps({
            "category": action.predicted_category,
            "priority": action.predicted_priority
        })

        error_field = parse_error if parse_error else "null"

        print(
            f"[STEP] step={steps_taken} action={action_summary} "
            f"reward={reward.total:.2f} done={str(done).lower()} error={error_field}",
            flush=True
        )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = min(max(sum(rewards) / max(len(rewards), 1), 0.0), 1.0)

    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("[ERROR] Missing API key", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    for task in list_tasks():
        run_episode(client, task)


if __name__ == "__main__":
    main()