"""Benchmark: 3 tasks × (standard, adversarial) × 5 runs each + naive baseline."""

import collections
import json
import math
import os
import subprocess
import sys
import time

import requests

from environment import IncidentResponseEnvironment
from task_definitions import ACTIONS, ADVERSARIAL_OVERLAYS, TASKS
from inference import run_groq_agent

SERVER_PORT = 7860
SERVER_URL = f"http://localhost:{SERVER_PORT}"
RUNS = 5

CORRECT_ACTION = {
    "task_easy":   "rollback_deployment",
    "task_medium": "rollback_deployment",
    "task_hard":   "rotate_db_credentials",
}


def start_server() -> subprocess.Popen:
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    return subprocess.Popen(
        [sys.executable, app_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_health(timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200:
                print(f"  Server healthy at {SERVER_URL}")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become healthy within {timeout}s")


def naive_action_for(task_id: str, adversarial: bool) -> str:
    """Return the ACTIONS name most frequently mentioned in the (possibly adversarial) chat."""
    chat = (
        ADVERSARIAL_OVERLAYS[task_id]
        if adversarial and task_id in ADVERSARIAL_OVERLAYS
        else TASKS[task_id]["observation"]["chat_history"]
    )
    counts: collections.Counter = collections.Counter()
    for msg in chat:
        text = msg["msg"].lower()
        for action in ACTIONS:
            if action in text:
                counts[action] += 1
    return counts.most_common(1)[0][0] if counts else ACTIONS[0]


def run_once(
    env: IncidentResponseEnvironment,
    task_id: str,
    adversarial: bool,
    action: str,
) -> float:
    episode_id, obs = env.reset(task_id=task_id, adversarial=adversarial)
    # Use log index 0 as evidence — always in-bounds
    env.step(episode_id, {"action": action, "evidence": 0})
    return env.grade(episode_id)["score"]


def collect(env, task_id, adversarial, action):
    return [run_once(env, task_id, adversarial, action) for _ in range(RUNS)]


def stats(scores: list[float]) -> tuple[float, float]:
    avg = sum(scores) / len(scores)
    std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))
    return round(avg, 4), round(std, 4)


def render_table(rows: list[dict]) -> None:
    cols = ["task", "mode", "model", "avg_score", "std_dev"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep    = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")


def main() -> None:
    groq_available = bool(os.environ.get("GROQ_API_KEY", ""))
    print("Starting server...")
    proc = start_server()
    try:
        wait_for_health()
        if groq_available:
            print("  GROQ_API_KEY found — llama-3.3-70b will run live")
        else:
            print("  GROQ_API_KEY not set — llama-3.3-70b will use deterministic fallback")

        env = IncidentResponseEnvironment()
        rows: list[dict] = []

        for task_id in TASKS:
            for adversarial in (False, True):
                mode = "adversarial" if adversarial else "standard"

                # Oracle: always the known-correct action
                oracle_scores = collect(env, task_id, adversarial, CORRECT_ACTION[task_id])
                avg, std = stats(oracle_scores)
                rows.append({
                    "task": task_id, "mode": mode, "model": "oracle",
                    "avg_score": avg, "std_dev": std,
                })

                # Naive baseline: pick action most mentioned in chat
                naive = naive_action_for(task_id, adversarial)
                naive_scores = collect(env, task_id, adversarial, naive)
                avg_n, std_n = stats(naive_scores)
                rows.append({
                    "task": task_id, "mode": mode, "model": f"naive({naive})",
                    "avg_score": avg_n, "std_dev": std_n,
                })

                # Groq LLM agent
                groq_scores = [run_groq_agent(task_id, adversarial) for _ in range(RUNS)]
                avg_g, std_g = stats(groq_scores)
                rows.append({
                    "task": task_id, "mode": mode, "model": "llama-3.3-70b",
                    "avg_score": avg_g, "std_dev": std_g,
                })

        print("\n## Benchmark Results\n")
        render_table(rows)

        with open("benchmark_results.json", "w") as f:
            json.dump(rows, f, indent=2)
        print("\nSaved -> benchmark_results.json")

    finally:
        proc.terminate()
        proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
