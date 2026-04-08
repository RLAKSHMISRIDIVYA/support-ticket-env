"""
SupportTicketEnv — OpenEnv-compliant environment for customer support ticket triage.

The environment simulates a real-world customer support system where an AI agent:
  1. Receives a customer support ticket as an observation
  2. Predicts the category and priority of the ticket
  3. Responds to the customer
  4. Optionally asks follow-up questions (multi-step interaction)
  5. Receives a structured reward for each action

Conforms to the OpenEnv specification:
  - Pydantic models: Observation, Action, Reward
  - Methods: reset(), step(), state()
"""

from __future__ import annotations
import copy
from typing import Any, Optional
from models import Observation, Action, Reward
from tasks import get_task
from grader import compute_reward


class SupportTicketEnv:
    """
    OpenEnv-compliant environment for customer support ticket triage.

    Episode lifecycle:
        1. Call reset(task_name) to start a new episode.
        2. Call step(action) repeatedly until done=True.
        3. Call state() at any point to inspect the current state.

    Multi-step interaction:
        - The agent may set `optional_followup_question` in its action.
        - The environment responds in the next observation's `clarification_answer`.
        - Each step counts against the max_steps limit regardless of followup.
        - The episode ends when:
            (a) the agent gives a correct category + priority, OR
            (b) max_steps is reached.
    """

    def __init__(self) -> None:
        self._task: Optional[dict] = None
        self._step: int = 0
        self._done: bool = False
        self._previous_actions: list[dict] = []
        self._rewards: list[float] = []
        self._last_observation: Optional[Observation] = None
        self._pending_followup_answer: Optional[str] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """
        Reset the environment and start a new episode for the given task.

        Args:
            task_name: Name of the task (see tasks.py for available tasks).

        Returns:
            Initial Observation with the ticket text and metadata.

        Raises:
            KeyError: If task_name is not recognized.
        """
        self._task = get_task(task_name)
        self._step = 0
        self._done = False
        self._previous_actions = []
        self._rewards = []
        self._pending_followup_answer = None

        obs = Observation(
            ticket_id=self._task["ticket_id"],
            ticket_text=self._task["ticket_text"],
            step=0,
            max_steps=self._task["max_steps"],
            previous_actions=[],
            clarification_answer=None,
            done=False,
            info={
                "task_name": self._task["task_name"],
                "difficulty": self._task["difficulty"],
                "message": "Episode started. Analyze the ticket and respond.",
            },
        )
        self._last_observation = obs
        return obs

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action | dict) -> tuple[Observation, Reward, bool, dict]:
        """
        Process the agent's action and advance the environment by one step.

        This method is designed to NEVER crash — invalid inputs are handled
        gracefully and result in a safe default observation with a zero reward.

        Args:
            action: An Action instance or a dict that can be coerced to one.

        Returns:
            Tuple of (observation, reward, done, info):
              - observation: Next Observation
              - reward: Structured Reward for this step
              - done: Whether the episode has ended
              - info: Additional metadata dict
        """
        # --- Guard: episode must be active ---
        if self._task is None:
            return self._error_response("Call reset() before step().")

        if self._done:
            return self._error_response("Episode is already done. Call reset() to start a new one.")

        # --- Coerce dict → Action ---
        error_msg: Optional[str] = None
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except Exception as e:
                error_msg = f"Invalid action dict: {e}"
                action = self._default_action()

        # --- Validate Action type ---
        if not isinstance(action, Action):
            error_msg = f"Expected Action or dict, got {type(action).__name__}"
            action = self._default_action()

        # --- Compute reward ---
        try:
            reward = compute_reward(
                action=action,
                task=self._task,
                previous_actions=self._previous_actions,
            )
        except Exception as e:
            # Grader should never crash, but protect anyway
            reward = Reward(
                category_score=0.0,
                priority_score=0.0,
                response_quality_score=0.0,
                customer_satisfaction_score=0.0,
                total=0.0,
                penalties={},
                breakdown={"error": str(e)},
            )

        self._rewards.append(reward.total)

        # --- Record action ---
        action_record = {
            "step": self._step,
            "predicted_category": action.predicted_category,
            "predicted_priority": action.predicted_priority,
            "response_message": action.response_message,
            "optional_followup_question": action.optional_followup_question,
            "reward": reward.total,
        }
        self._previous_actions.append(action_record)

        # --- Advance step counter ---
        self._step += 1
        true_cat = self._task["true_category"]
        true_pri = self._task["true_priority"]
        max_steps = self._task["max_steps"]

        # --- Determine done ---
        correct_resolution = (
            action.predicted_category == true_cat
            and action.predicted_priority == true_pri
        )
        reached_max = self._step >= max_steps
        self._done = correct_resolution or reached_max

        # --- Prepare clarification answer for next step ---
        followup_answer: Optional[str] = None
        if action.optional_followup_question and not self._done:
            followup_answer = self._task.get("clarification_answer")

        # --- Build info ---
        info: dict[str, Any] = {
            "task_name": self._task["task_name"],
            "step": self._step,
            "correct_resolution": correct_resolution,
            "reached_max_steps": reached_max,
            "cumulative_rewards": list(self._rewards),
            "total_score": round(sum(self._rewards) / max(len(self._rewards), 1), 4),
            "reward_breakdown": reward.breakdown,
        }
        if error_msg:
            info["error"] = error_msg
        else:
            info["error"] = None

        if self._done:
            info["message"] = (
                "Episode complete — correct resolution!"
                if correct_resolution
                else "Episode complete — max steps reached."
            )

        # --- Build next observation ---
        obs = Observation(
            ticket_id=self._task["ticket_id"],
            ticket_text=self._task["ticket_text"],
            step=self._step,
            max_steps=max_steps,
            previous_actions=copy.deepcopy(self._previous_actions),
            clarification_answer=followup_answer,
            done=self._done,
            info=info,
        )
        self._last_observation = obs
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """
        Return the current full internal state of the environment.

        Returns:
            Dict with all state fields including task metadata, step count,
            cumulative rewards, and previous actions.
        """
        if self._task is None:
            return {"status": "not_started", "message": "Call reset() to begin."}

        return {
            "task_name": self._task["task_name"],
            "ticket_id": self._task["ticket_id"],
            "difficulty": self._task["difficulty"],
            "step": self._step,
            "max_steps": self._task["max_steps"],
            "done": self._done,
            "previous_actions": copy.deepcopy(self._previous_actions),
            "rewards": list(self._rewards),
            "cumulative_score": round(
                sum(self._rewards) / max(len(self._rewards), 1), 4
            ) if self._rewards else 0.0,
            "true_category": self._task["true_category"],
            "true_priority": self._task["true_priority"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_action(self) -> Action:
        """Return a safe default action for error recovery."""
        return Action(
            predicted_category="technical",
            predicted_priority="medium",
            response_message="",
            optional_followup_question=None,
        )

    def _error_response(
        self, message: str
    ) -> tuple[Observation, Reward, bool, dict]:
        """
        Return a safe default tuple when the environment is in an invalid state.
        Never raises an exception.
        """
        zero_reward = Reward(
            category_score=0.0,
            priority_score=0.0,
            response_quality_score=0.0,
            customer_satisfaction_score=0.0,
            total=0.0,
            penalties={},
            breakdown={"error": message},
        )
        obs = Observation(
            ticket_id="N/A",
            ticket_text="",
            step=self._step,
            max_steps=0,
            previous_actions=[],
            done=True,
            info={"error": message},
        )
        info = {
            "error": message,
            "total_score": 0.0,
            "cumulative_rewards": [],
            "reward_breakdown": {},
            "correct_resolution": False,
            "reached_max_steps": False,
        }
        return obs, zero_reward, True, info
