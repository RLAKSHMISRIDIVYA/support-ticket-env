"""
Deterministic grader for SupportTicketEnv.

Computes a structured reward breakdown for each agent action.
All scoring is deterministic — same inputs always yield the same scores.
"""

from __future__ import annotations
import math
from models import Action, Reward


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CATEGORY_SCORE = 0.4
MAX_PRIORITY_SCORE = 0.3
MAX_RESPONSE_QUALITY_SCORE = 0.2
MAX_CUSTOMER_SATISFACTION_SCORE = 0.1

PENALTY_WRONG_CATEGORY = -0.1
PENALTY_WRONG_PRIORITY = -0.1
PENALTY_EMPTY_RESPONSE = -0.2
PENALTY_REPEATED_MISTAKE = -0.1
PENALTY_UNNECESSARY_LONG = -0.05

# Threshold above which a response is considered unnecessarily long (chars)
RESPONSE_LENGTH_THRESHOLD = 1200

# Priority adjacency — being "one off" is less bad than being two off
PRIORITY_ORDER = ["low", "medium", "high"]

# Polite/helpful words that indicate good tone
TONE_POSITIVE_WORDS = [
    "apologize", "sorry", "understand", "help", "assist", "happy", "glad",
    "resolve", "fix", "ensure", "assure", "sincerely", "thank", "appreciate",
    "priority", "urgent", "immediately", "quickly", "promptly",
]

# Words that indicate the response is trying to resolve the issue
RESOLUTION_WORDS = [
    "refund", "credit", "reset", "update", "fix", "resolve", "investigate",
    "escalate", "follow up", "ticket", "team", "check", "verify", "confirm",
    "process", "issue", "problem", "solution",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _priority_distance(predicted: str, true: str) -> int:
    """Return the ordinal distance between two priority levels (0, 1, or 2)."""
    try:
        return abs(PRIORITY_ORDER.index(predicted) - PRIORITY_ORDER.index(true))
    except ValueError:
        return 2  # Treat unknown as worst case


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    """
    Fraction of keywords present in the text (case-insensitive).
    Returns a float in [0.0, 1.0].
    """
    if not keywords:
        return 1.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def _tone_score(response: str) -> float:
    """
    Score the tone of a response based on presence of polite/helpful words.
    Returns a float in [0.0, 1.0].
    """
    return min(1.0, _keyword_coverage(response, TONE_POSITIVE_WORDS) * 2.5)


def _resolution_score(response: str) -> float:
    """
    Score how well the response addresses resolution based on action words.
    Returns a float in [0.0, 1.0].
    """
    return min(1.0, _keyword_coverage(response, RESOLUTION_WORDS) * 3.0)


def _relevance_score(response: str, ticket_text: str, expected_keywords: list[str]) -> float:
    """
    Score relevance as coverage of expected keywords from the ticket context.
    Returns a float in [0.0, 1.0].
    """
    return _keyword_coverage(response, expected_keywords)


def _clarity_score(response: str) -> float:
    """
    Heuristic clarity score based on response length and structure.
    - Too short (< 50 chars): likely incomplete
    - Good range (50–800 chars): full score
    - Getting long (800–1200): slight reduction
    Returns a float in [0.0, 1.0].
    """
    length = len(response)
    if length < 30:
        return 0.1
    elif length < 50:
        return 0.5
    elif length <= 800:
        return 1.0
    elif length <= 1200:
        return 0.8
    else:
        return 0.6


# ---------------------------------------------------------------------------
# Main grader function
# ---------------------------------------------------------------------------

def compute_reward(
    action: Action,
    task: dict,
    previous_actions: list[dict],
) -> Reward:
    """
    Compute a structured reward for an agent's action on a given task.

    Args:
        action: The agent's current action.
        task: The task dict (from tasks.py).
        previous_actions: All actions taken before this one in the episode.

    Returns:
        A Reward instance with full breakdown.
    """
    true_category: str = task["true_category"]
    true_priority: str = task["true_priority"]
    ticket_text: str = task["ticket_text"]
    expected_keywords: list[str] = task.get("expected_keywords", [])

    penalties: dict[str, float] = {}
    breakdown: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Category score (0–0.4)
    # ------------------------------------------------------------------
    if action.predicted_category == true_category:
        category_score = MAX_CATEGORY_SCORE
        breakdown["category"] = f"Correct ({action.predicted_category}) → full score"
    else:
        category_score = 0.0
        penalty = PENALTY_WRONG_CATEGORY
        penalties["wrong_category"] = penalty
        breakdown["category"] = (
            f"Wrong (predicted={action.predicted_category}, true={true_category}) → "
            f"0 + {penalty} penalty"
        )

    # ------------------------------------------------------------------
    # 2. Priority score (0–0.3) — partial credit for being "one off"
    # ------------------------------------------------------------------
    dist = _priority_distance(action.predicted_priority, true_priority)
    if dist == 0:
        priority_score = MAX_PRIORITY_SCORE
        breakdown["priority"] = f"Correct ({action.predicted_priority}) → full score"
    elif dist == 1:
        priority_score = MAX_PRIORITY_SCORE * 0.5
        penalties["wrong_priority"] = PENALTY_WRONG_PRIORITY * 0.5
        breakdown["priority"] = (
            f"Off by one (predicted={action.predicted_priority}, true={true_priority}) "
            f"→ partial score"
        )
    else:
        priority_score = 0.0
        penalties["wrong_priority"] = PENALTY_WRONG_PRIORITY
        breakdown["priority"] = (
            f"Wrong (predicted={action.predicted_priority}, true={true_priority}) "
            f"→ 0 + penalty"
        )

    # ------------------------------------------------------------------
    # 3. Response quality score (0–0.2)
    # ------------------------------------------------------------------
    response = action.response_message

    if not response:
        response_quality_score = 0.0
        penalties["empty_response"] = PENALTY_EMPTY_RESPONSE
        breakdown["response_quality"] = "Empty response → 0 + penalty"
    else:
        relevance = _relevance_score(response, ticket_text, expected_keywords)
        clarity = _clarity_score(response)
        # Weighted combination: relevance 50%, clarity 50%
        raw_quality = (relevance * 0.5 + clarity * 0.5)
        response_quality_score = round(raw_quality * MAX_RESPONSE_QUALITY_SCORE, 4)
        breakdown["response_quality"] = (
            f"relevance={relevance:.2f}, clarity={clarity:.2f} → {response_quality_score:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Customer satisfaction score (0–0.1)
    # ------------------------------------------------------------------
    if not response:
        customer_satisfaction_score = 0.0
        breakdown["customer_satisfaction"] = "Empty response → 0"
    else:
        tone = _tone_score(response)
        resolution = _resolution_score(response)
        # Category/priority correctness boosts satisfaction
        correctness_bonus = (
            (1.0 if action.predicted_category == true_category else 0.0) * 0.3
            + (1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)) * 0.2
        )
        raw_satisfaction = min(1.0, (tone * 0.35 + resolution * 0.35 + correctness_bonus))
        customer_satisfaction_score = round(
            raw_satisfaction * MAX_CUSTOMER_SATISFACTION_SCORE, 4
        )
        breakdown["customer_satisfaction"] = (
            f"tone={tone:.2f}, resolution={resolution:.2f}, "
            f"correctness_bonus={correctness_bonus:.2f} → {customer_satisfaction_score:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Penalties
    # ------------------------------------------------------------------

    # Unnecessarily long response
    if len(response) > RESPONSE_LENGTH_THRESHOLD:
        penalties["unnecessary_long"] = PENALTY_UNNECESSARY_LONG
        breakdown["penalty_long"] = (
            f"Response too long ({len(response)} chars > {RESPONSE_LENGTH_THRESHOLD}) → "
            f"{PENALTY_UNNECESSARY_LONG} penalty"
        )

    # Repeated mistakes — check if same wrong category/priority was given before
    if previous_actions:
        prev_wrong_cats = [
            p for p in previous_actions
            if p.get("predicted_category") == action.predicted_category
            and action.predicted_category != true_category
        ]
        prev_wrong_pris = [
            p for p in previous_actions
            if p.get("predicted_priority") == action.predicted_priority
            and dist > 0
        ]
        if prev_wrong_cats or prev_wrong_pris:
            penalties["repeated_mistake"] = PENALTY_REPEATED_MISTAKE
            breakdown["penalty_repeated"] = (
                "Repeated incorrect category or priority → "
                f"{PENALTY_REPEATED_MISTAKE} penalty"
            )

    # ------------------------------------------------------------------
    # 6. Compute total (clamped to [0, 1])
    # ------------------------------------------------------------------
    raw_total = (
        category_score
        + priority_score
        + response_quality_score
        + customer_satisfaction_score
        + sum(penalties.values())
    )
    total = round(max(0.0, min(1.0, raw_total)), 4)

    breakdown["total"] = (
        f"raw={raw_total:.4f} → clamped total={total:.4f}"
    )

    return Reward(
        category_score=round(max(0.0, category_score), 4),
        priority_score=round(max(0.0, priority_score), 4),
        response_quality_score=round(max(0.0, response_quality_score), 4),
        customer_satisfaction_score=round(max(0.0, customer_satisfaction_score), 4),
        total=total,
        penalties=penalties,
        breakdown=breakdown,
    )
