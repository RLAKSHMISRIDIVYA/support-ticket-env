"""
Pydantic models for the SupportTicketEnv OpenEnv environment.
Defines Observation, Action, and Reward data structures.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# --- Type Aliases ---
Category = Literal["billing", "technical", "account"]
Priority = Literal["low", "medium", "high"]


class Observation(BaseModel):
    """
    Observation returned to the agent at each step.

    Attributes:
        ticket_id: Unique identifier for the current ticket.
        ticket_text: The raw customer complaint text.
        step: Current step number within the episode.
        max_steps: Maximum steps allowed per episode.
        previous_actions: History of actions taken so far.
        clarification_answer: Answer to the agent's last follow-up question (if any).
        done: Whether the episode has ended.
        info: Additional metadata or error messages.
    """
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    ticket_text: str = Field(..., description="Customer complaint text")
    step: int = Field(..., description="Current step number (0-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode")
    previous_actions: list[dict] = Field(
        default_factory=list,
        description="List of previous actions taken in this episode"
    )
    clarification_answer: Optional[str] = Field(
        default=None,
        description="Answer to the agent's last follow-up question, if applicable"
    )
    done: bool = Field(default=False, description="Whether the episode is complete")
    info: dict = Field(default_factory=dict, description="Additional metadata")


class Action(BaseModel):
    """
    Action submitted by the agent at each step.

    Attributes:
        predicted_category: The agent's predicted ticket category.
        predicted_priority: The agent's predicted ticket priority.
        response_message: The agent's response to the customer.
        optional_followup_question: If set, the agent is asking a clarification question
                                    and the environment will respond in the next step.
    """
    predicted_category: Category = Field(..., description="Predicted ticket category")
    predicted_priority: Priority = Field(..., description="Predicted ticket priority")
    response_message: str = Field(..., description="Agent's response to the customer")
    optional_followup_question: Optional[str] = Field(
        default=None,
        description="Optional follow-up question for multi-step clarification"
    )

    @field_validator("response_message")
    @classmethod
    def response_message_not_empty(cls, v: str) -> str:
        # Strip but do not raise — env will penalize empty responses
        return v.strip()


class Reward(BaseModel):
    """
    Structured reward breakdown for a single step.

    Attributes:
        category_score: Score for correct category prediction (0–0.4).
        priority_score: Score for correct priority prediction (0–0.3).
        response_quality_score: Score for response quality (0–0.2).
        customer_satisfaction_score: Score for customer satisfaction (0–0.1).
        total: Final clamped reward in [0.0, 1.0].
        penalties: Dictionary of applied penalties.
        breakdown: Human-readable explanation of each score component.
    """
    category_score: float = Field(..., ge=0.0, le=0.4)
    priority_score: float = Field(..., ge=0.0, le=0.3)
    response_quality_score: float = Field(..., ge=0.0, le=0.2)
    customer_satisfaction_score: float = Field(..., ge=0.0, le=0.1)
    total: float = Field(..., ge=0.0, le=1.0)
    penalties: dict[str, float] = Field(default_factory=dict)
    breakdown: dict[str, str] = Field(default_factory=dict)
