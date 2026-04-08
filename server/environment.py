"""Core environment logic for Incident-Response-Detective."""

import uuid
from models import IncidentAction, IncidentObservation, IncidentState
from task_definitions import TASKS, ACTIONS, compute_reward


class IncidentResponseEnvironment:
    """
    OpenEnv-compatible environment for incident triage.
    Implements reset(), step(), state() following the Gymnasium-style API.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._episodes: dict[str, dict] = {}

    def get_tasks(self) -> list[dict]:
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
            }
            for t in TASKS.values()
        ]

    def reset(self, task_id: str = "task_easy") -> tuple[str, dict]:
        """Start a new episode. Returns (episode_id, observation_dict)."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Choose from: {list(TASKS.keys())}")

        episode_id = str(uuid.uuid4())
        task = TASKS[task_id]

        self._episodes[episode_id] = {
            "task_id": task_id,
            "step_count": 0,
            "done": False,
            "resolved": False,
            "actions_taken": [],
            "rewards": [],
            "cumulative_reward": 0.0,
        }

        obs = {
            "task_id": task_id,
            "task_name": task["name"],
            "task_description": task["description"],
            "logs": task["observation"]["logs"],
            "chat_history": task["observation"]["chat_history"],
            "runbook": task["observation"]["runbook"],
            "available_actions": ACTIONS,
            "step": 0,
            "max_steps": task["max_steps"],
            "done": False,
            "score": 0.0,
            "last_reward": 0.0,
            "reward_breakdown": {},
            "feedback": "Episode started. Analyze the observation and choose a remediation action.",
            "last_action_error": None,
        }
        return episode_id, obs

    def step(self, episode_id: str, action_dict: dict) -> dict:
        """Execute an action. Returns observation dict."""
        if episode_id not in self._episodes:
            raise ValueError(f"Unknown episode_id: {episode_id}")

        ep = self._episodes[episode_id]
        if ep["done"]:
            raise ValueError("Episode already finished.")

        action_str = action_dict.get("action", "")
        if action_str not in ACTIONS:
            # Return error observation without consuming a step
            task = TASKS[ep["task_id"]]
            return {
                "task_id": ep["task_id"],
                "task_name": task["name"],
                "task_description": task["description"],
                "logs": task["observation"]["logs"],
                "chat_history": task["observation"]["chat_history"],
                "runbook": task["observation"]["runbook"],
                "available_actions": ACTIONS,
                "step": ep["step_count"],
                "max_steps": task["max_steps"],
                "done": False,
                "score": ep["cumulative_reward"],
                "last_reward": 0.0,
                "reward_breakdown": {},
                "feedback": f"Invalid action: {action_str}",
                "last_action_error": f"Invalid action '{action_str}'. Choose from: {ACTIONS}",
            }

        ep["step_count"] += 1
        ep["actions_taken"].append(action_str)

        reward_info = compute_reward(ep["task_id"], action_str, ep["step_count"])
        ep["rewards"].append(reward_info["reward"])
        ep["cumulative_reward"] = round(sum(ep["rewards"]), 3)

        if reward_info["done"]:
            ep["done"] = True
            ep["resolved"] = reward_info["resolved"]

        task = TASKS[ep["task_id"]]

        feedback_parts = [
            reward_info["safety"]["reason"],
            reward_info["efficiency"]["reason"],
        ]
        if ep["done"]:
            if ep["resolved"]:
                feedback_parts.append("INCIDENT RESOLVED.")
            else:
                feedback_parts.append("INCIDENT NOT RESOLVED. Episode ended.")

        return {
            "task_id": ep["task_id"],
            "task_name": task["name"],
            "task_description": task["description"],
            "logs": task["observation"]["logs"],
            "chat_history": task["observation"]["chat_history"],
            "runbook": task["observation"]["runbook"],
            "available_actions": ACTIONS,
            "step": ep["step_count"],
            "max_steps": task["max_steps"],
            "done": ep["done"],
            "score": ep["cumulative_reward"],
            "last_reward": reward_info["reward"],
            "reward_breakdown": {
                "safety": reward_info["safety"],
                "efficiency": reward_info["efficiency"],
            },
            "feedback": " | ".join(feedback_parts),
            "last_action_error": None,
        }

    def get_state(self, episode_id: str) -> dict:
        if episode_id not in self._episodes:
            raise ValueError(f"Unknown episode_id: {episode_id}")
        ep = self._episodes[episode_id]
        return {
            "episode_id": episode_id,
            "task_id": ep["task_id"],
            "step_count": ep["step_count"],
            "done": ep["done"],
            "resolved": ep["resolved"],
            "actions_taken": ep["actions_taken"],
            "rewards": ep["rewards"],
            "cumulative_reward": ep["cumulative_reward"],
        }

    def grade(self, episode_id: str) -> dict:
        """Grade an episode. Returns score in 0.0-1.0."""
        if episode_id not in self._episodes:
            raise ValueError(f"Unknown episode_id: {episode_id}")
        ep = self._episodes[episode_id]

        if ep["resolved"]:
            # Perfect resolution on step 1 = 0.999, later steps lower
            base = 0.999 if ep["step_count"] == 1 else max(0.5, 0.999 - 0.15 * (ep["step_count"] - 1))
            return {"score": round(base, 3), "resolved": True, "steps": ep["step_count"]}
        else:
            # Partial credit: did the agent avoid dangerous actions?
            task = TASKS[ep["task_id"]]
            dangerous_taken = [a for a in ep["actions_taken"] if a in task["dangerous_actions"]]
            if dangerous_taken:
                return {"score": 0.001, "resolved": False, "steps": ep["step_count"]}
            else:
                return {"score": 0.15, "resolved": False, "steps": ep["step_count"]}
