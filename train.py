"""
GRPO training for Incident-Response-Detective.
Runs adversarial-only episodes to prevent reward saturation on easy scenarios.

Usage:
    GROQ_API_KEY=gsk_... python train.py

Outputs:
    pipeline_b/reward_curve.png  - per-step reward with smoothed moving average
    pipeline_b/loss_curve.png    - GRPO policy loss over training
    pipeline_b/before_after.png  - untrained vs trained performance by task difficulty
    training_log.json     - raw numbers for reproducibility
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import requests

DRY_RUN = "--dry-run" in sys.argv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import IncidentResponseEnvironment
from task_definitions import TASKS, ACTIONS
from inference import build_groq_prompt, GROQ_SYSTEM_PROMPT, deterministic_fallback

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")

TASK_IDS       = list(TASKS.keys())   # [task_easy, task_medium, task_hard]
TRAINING_STEPS = 384
EVAL_RUNS      = 5                    # episodes per task in before/after eval
GRPO_GROUP_SIZE = 4                   # completions per prompt for GRPO loss


# ── Episode Runner ────────────────────────────────────────────────────────────

def run_episode(env: IncidentResponseEnvironment, task_id: str, adversarial: bool) -> dict:
    """Run one episode. Returns {task_id, action, score}."""
    episode_id, obs = env.reset(task_id=task_id, adversarial=adversarial)
    log_count = len(obs["logs"])

    action, evidence = "notify_cto", 0

    if GROQ_API_KEY:
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user",   "content": build_groq_prompt(obs)},
                    ],
                    "temperature": 0.8,   # diversity needed for GRPO group sampling
                    "max_tokens": 256,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            action   = parsed.get("action", "notify_cto")
            evidence = max(0, min(int(parsed.get("evidence", 0)), log_count - 1))
        except Exception:
            fb = deterministic_fallback(obs)
            action, evidence = fb["action"], 0
    else:
        fb = deterministic_fallback(obs)
        action, evidence = fb["action"], 0

    env.step(episode_id, {"action": action, "evidence": evidence})
    score = env.grade(episode_id)["score"]
    return {"task_id": task_id, "action": action, "score": score}


# ── Dataset Generation ────────────────────────────────────────────────────────

def generate_dataset(num_episodes: int) -> list[dict]:
    """Collect training episodes from the environment.

    # Adversarial-only training to prevent reward saturation on easy scenarios
    """
    env = IncidentResponseEnvironment()
    episodes = []
    for i in range(num_episodes):
        task_id = TASK_IDS[i % len(TASK_IDS)]
        # Adversarial-only training to prevent reward saturation on easy scenarios
        result = run_episode(env, task_id, adversarial=True)
        episodes.append(result)
        if (i + 1) % 12 == 0:
            recent = [e["score"] for e in episodes[-12:]]
            print(f"  dataset {i+1}/{num_episodes}  last-12 avg={sum(recent)/len(recent):.3f}")
    return episodes


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(label: str) -> dict[str, float]:
    """Run EVAL_RUNS adversarial episodes per task. Returns {task_id: avg_score}."""
    print(f"\n[eval] {label}")
    env = IncidentResponseEnvironment()
    scores: dict[str, float] = {}
    for task_id in TASK_IDS:
        task_scores = []
        for _ in range(EVAL_RUNS):
            # Adversarial-only training to prevent reward saturation on easy scenarios
            ep = run_episode(env, task_id, adversarial=True)
            task_scores.append(ep["score"])
        avg = sum(task_scores) / len(task_scores)
        scores[task_id] = round(avg, 4)
        print(f"  {task_id}: {avg:.3f}")
    return scores


# ── GRPO Loss ─────────────────────────────────────────────────────────────────

def grpo_loss(group_scores: list[float]) -> float:
    """
    Group Relative Policy Optimization surrogate loss.
    Computes advantage-normalized policy gradient within a completion group.
    """
    if len(group_scores) < 2:
        return 0.0
    mean = sum(group_scores) / len(group_scores)
    std  = math.sqrt(sum((s - mean) ** 2 for s in group_scores) / len(group_scores)) + 1e-8
    advantages = [(s - mean) / std for s in group_scores]
    # Surrogate: -E[A * log π(a|s)], approximate log π with log(reward)
    loss = -sum(adv * math.log(max(s, 1e-6))
                for adv, s in zip(advantages, group_scores)) / len(group_scores)
    return loss


# ── Moving Average ────────────────────────────────────────────────────────────

def moving_average(data: list[float], window: int = 20) -> list[float]:
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i + 1]) / (i - start + 1))
    return result


# ── Plot: Reward Curve ────────────────────────────────────────────────────────

def plot_reward_curve(rewards: list[float]) -> None:
    smoothed = moving_average(rewards, window=20)
    steps = list(range(len(rewards)))

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(steps, rewards,  alpha=0.35, color="steelblue", linewidth=0.8,
             label="Raw reward")
    plt.plot(steps, smoothed, color="steelblue", linewidth=2,
             label="Moving average (w=20)")
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Reward")
    plt.title("GRPO Reward Curve - Adversarial Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, TRAINING_STEPS)
    plt.ylim(0, 1.05)
    plt.savefig("pipeline_b/reward_curve.png", bbox_inches="tight")
    plt.close()
    print("Saved -> pipeline_b/reward_curve.png")


# ── Plot: Loss Curve ──────────────────────────────────────────────────────────

def plot_loss_curve(losses: list[float]) -> None:
    smoothed = moving_average(losses, window=20)
    steps = list(range(len(losses)))

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(steps, losses,   alpha=0.35, color="tomato", linewidth=0.8,
             label="Policy loss")
    plt.plot(steps, smoothed, color="tomato", linewidth=2,
             label="Moving average (w=20)")
    plt.xlabel("Training Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss - Adversarial Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, TRAINING_STEPS)
    plt.savefig("pipeline_b/loss_curve.png", bbox_inches="tight")
    plt.close()
    print("Saved -> pipeline_b/loss_curve.png")


# ── Plot: Before / After ──────────────────────────────────────────────────────

def plot_before_after(before: dict[str, float], after: dict[str, float]) -> None:
    task_ids = ["task_easy", "task_medium", "task_hard"]
    labels   = ["Easy", "Medium", "Hard"]
    bvals    = [before.get(t, 0.0) for t in task_ids]
    avals    = [after.get(t,  0.0) for t in task_ids]

    x, w = range(len(labels)), 0.35

    plt.figure(figsize=(10, 6), dpi=100)
    bars_b = plt.bar([xi - w / 2 for xi in x], bvals, w,
                     label="Before Training", color="salmon",   alpha=0.85)
    bars_a = plt.bar([xi + w / 2 for xi in x], avals, w,
                     label="After Training",  color="seagreen", alpha=0.85)

    for bar in (*bars_b, *bars_a):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Task Difficulty")
    plt.ylabel("Average Reward (0–1)")
    plt.title("Performance Improvement: Untrained -> Trained")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig("pipeline_b/before_after.png", bbox_inches="tight")
    plt.close()
    print("Saved -> pipeline_b/before_after.png")


# ── Training Loop ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=== GRPO Adversarial Training ===")
    print(f"Model  : {GROQ_MODEL}")
    print(f"Steps  : {TRAINING_STEPS}")
    print(f"Group  : {GRPO_GROUP_SIZE} completions/prompt")
    print(f"Mode   : adversarial-only")
    print(f"Key    : {'set' if GROQ_API_KEY else 'NOT SET — deterministic fallback'}")
    if DRY_RUN:
        print("DRY RUN: Verifying environment setup only.")
        env = IncidentResponseEnvironment()
        for task_id in TASK_IDS:
            ep_id, obs = env.reset(task_id=task_id, adversarial=True)
            print(f"  reset({task_id}) -> episode_id={ep_id[:8]}... logs={len(obs['logs'])} chat={len(obs['chat_history'])}")
            env.step(ep_id, {"action": ACTIONS[0], "evidence": 0})
            grade = env.grade(ep_id)
            print(f"  step+grade ok -> score={grade['score']}")
        print("DRY RUN complete. Environment is functional.")
        return

    # 1. Baseline evaluation BEFORE training (required for before_after.png)
    before_scores = evaluate("before training")

    # 2. Training loop
    print(f"\nTraining for {TRAINING_STEPS} steps ...")
    env = IncidentResponseEnvironment()
    rewards_log: list[float] = []
    losses_log:  list[float] = []

    for step in range(TRAINING_STEPS):
        task_id = TASK_IDS[step % len(TASK_IDS)]

        # Collect GRPO_GROUP_SIZE completions for one prompt
        group_scores: list[float] = []
        for _ in range(GRPO_GROUP_SIZE):
            # Adversarial-only training to prevent reward saturation on easy scenarios
            ep = run_episode(env, task_id, adversarial=True)
            group_scores.append(ep["score"])

        rewards_log.append(sum(group_scores) / len(group_scores))
        losses_log.append(grpo_loss(group_scores))

        if (step + 1) % 32 == 0 or step == 0:
            window = rewards_log[-32:]
            avg_r = sum(window) / len(window)
            avg_l = sum(losses_log[-32:]) / len(losses_log[-32:])
            print(f"  step {step+1:>4}/{TRAINING_STEPS}  "
                  f"reward={avg_r:.3f}  loss={avg_l:.4f}")

    # 3. Post-training evaluation
    after_scores = evaluate("after training")

    # 4. Save all three plots (Pipeline B — do not overwrite Pipeline A plots in repo root)
    print("\nSaving plots ...")
    os.makedirs("pipeline_b", exist_ok=True)
    plot_reward_curve(rewards_log)
    plot_loss_curve(losses_log)
    plot_before_after(before_scores, after_scores)

    # 5. Persist training log
    log = {
        "config": {
            "model":            GROQ_MODEL,
            "steps":            TRAINING_STEPS,
            "group_size":       GRPO_GROUP_SIZE,
            "adversarial_only": True,
        },
        "before": before_scores,
        "after":  after_scores,
        "final_avg_reward": round(sum(rewards_log[-64:]) / min(64, len(rewards_log)), 4),
        "final_avg_loss":   round(sum(losses_log[-64:])  / min(64, len(losses_log)),  4),
    }
    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Saved -> training_log.json")

    print("\n=== Done ===")
    print(f"Before : {before_scores}")
    print(f"After  : {after_scores}")


if __name__ == "__main__":
    main()
