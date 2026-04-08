"""Client for the Incident-Response-Detective OpenEnv environment."""

import requests
from typing import Optional


class IncidentResponseClient:
    """HTTP client for interacting with the Incident-Response-Detective environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()

    def get_tasks(self) -> list[dict]:
        return requests.get(f"{self.base_url}/tasks").json()["tasks"]

    def reset(self, task_id: str = "task_easy") -> tuple[str, dict]:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        return data["episode_id"], data["observation"]

    def step(self, episode_id: str, action: str, reasoning: str = "") -> dict:
        resp = requests.post(f"{self.base_url}/step", json={
            "episode_id": episode_id,
            "action": {"action": action, "reasoning": reasoning},
        })
        resp.raise_for_status()
        return resp.json()["observation"]

    def state(self, episode_id: str) -> dict:
        resp = requests.get(f"{self.base_url}/state", params={"episode_id": episode_id})
        resp.raise_for_status()
        return resp.json()

    def grade(self, episode_id: str) -> dict:
        resp = requests.post(f"{self.base_url}/grader", json={"episode_id": episode_id})
        resp.raise_for_status()
        return resp.json()
