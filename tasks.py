"""
Deterministic task definitions for SupportTicketEnv.

Each task is a dict with:
  - task_name: str
  - difficulty: "easy" | "medium" | "hard"
  - ticket_id: str
  - ticket_text: str
  - true_category: str
  - true_priority: str
  - clarification_answer: str  — used if the agent asks a follow-up question
  - expected_keywords: list[str]  — keywords a good response should contain
  - max_steps: int

No randomness — same input always produces the same output.
"""

from __future__ import annotations

TASKS: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # EASY: Simple, unambiguous billing complaint
    # -----------------------------------------------------------------------
    "easy_billing": {
        "task_name": "easy_billing",
        "difficulty": "easy",
        "ticket_id": "TKT-001",
        "ticket_text": (
            "Hi, I was charged twice for my subscription this month. "
            "I see two identical charges of $29.99 on my credit card from your company. "
            "Please refund the duplicate charge as soon as possible."
        ),
        "true_category": "billing",
        "true_priority": "high",
        "clarification_answer": (
            "The duplicate charge appeared on April 1st, two days after my renewal date."
        ),
        "expected_keywords": [
            "refund", "charge", "billing", "duplicate", "apologize", "credit"
        ],
        "max_steps": 5,
    },

    # -----------------------------------------------------------------------
    # MEDIUM: Slightly ambiguous — could be technical OR account issue
    # -----------------------------------------------------------------------
    "medium_login": {
        "task_name": "medium_login",
        "difficulty": "medium",
        "ticket_id": "TKT-002",
        "ticket_text": (
            "I cannot log in to my account. I tried resetting my password but "
            "the reset email never arrives. It's been two hours. "
            "I have an important meeting in an hour and need access urgently."
        ),
        "true_category": "account",
        "true_priority": "high",
        "clarification_answer": (
            "I checked my spam folder and there is no reset email there either. "
            "My email address is correct — I can log in with it on your mobile app."
        ),
        "expected_keywords": [
            "password", "reset", "email", "account", "access", "urgent", "check spam",
            "verify", "alternative"
        ],
        "max_steps": 6,
    },

    # -----------------------------------------------------------------------
    # HARD: Multi-issue ticket with vague language — billing + technical + account
    # -----------------------------------------------------------------------
    "hard_multi_issue": {
        "task_name": "hard_multi_issue",
        "difficulty": "hard",
        "ticket_id": "TKT-003",
        "ticket_text": (
            "Things have been really messed up lately. First, the app keeps crashing "
            "whenever I try to export my data — it just freezes and then closes. "
            "On top of that, I noticed something weird on my invoice last week, "
            "like there was a charge I didn't recognise, maybe for a plan upgrade? "
            "I never asked for that. And now I can't even find where to change my "
            "email address in the settings — it seems like that whole section is gone. "
            "Can someone just sort all of this out please?"
        ),
        "true_category": "technical",
        "true_priority": "high",
        "clarification_answer": (
            "The crash happens on both iOS and desktop. The strange charge was $49 "
            "for something called 'Pro Annual'. My current email is old and I need "
            "to update it to my work address."
        ),
        "expected_keywords": [
            "crash", "export", "billing", "charge", "invoice", "email", "settings",
            "investigate", "multiple", "issues", "priority", "follow up"
        ],
        "max_steps": 8,
    },
}


def get_task(task_name: str) -> dict:
    """
    Retrieve a task by name. Raises KeyError if not found.
    Always returns the same deterministic object.
    """
    if task_name not in TASKS:
        raise KeyError(
            f"Unknown task '{task_name}'. Available tasks: {list(TASKS.keys())}"
        )
    # Return a copy to prevent mutation of the global definition
    return dict(TASKS[task_name])


def list_tasks() -> list[str]:
    """Return all available task names."""
    return list(TASKS.keys())
