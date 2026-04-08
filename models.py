"""Typed models for the Incident-Response-Detective OpenEnv environment."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IncidentAction:
    """Agent submits a remediation command."""
    action: str  # One of the valid action strings
    reasoning: str = ""  # Optional Chain-of-Thought explanation


@dataclass
class LogEntry:
    ts: str
    level: str
    service: str
    msg: str


@dataclass
class ChatMessage:
    user: str
    time: str
    msg: str


@dataclass
class RewardBreakdown:
    safety_score: float = 0.0
    safety_reason: str = ""
    efficiency_score: float = 0.0
    efficiency_reason: str = ""


@dataclass
class IncidentObservation:
    """What the agent sees each step."""
    task_id: str = ""
    task_name: str = ""
    task_description: str = ""
    logs: list[dict] = field(default_factory=list)
    chat_history: list[dict] = field(default_factory=list)
    runbook: str = ""
    available_actions: list[str] = field(default_factory=list)
    step: int = 0
    max_steps: int = 3
    done: bool = False
    score: float = 0.0
    last_reward: float = 0.0
    reward_breakdown: dict = field(default_factory=dict)
    feedback: str = ""
    last_action_error: Optional[str] = None


@dataclass
class IncidentState:
    """Internal episode state."""
    episode_id: str = ""
    task_id: str = ""
    step_count: int = 0
    done: bool = False
    actions_taken: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cumulative_reward: float = 0.0
    resolved: bool = False
