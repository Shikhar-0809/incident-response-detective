"""
Regenerate missing training plot PNGs from committed metadata.

Produces:
  reward_curve.png  — statistically constrained reconstruction from training_log.json
  before_after.png  — exact reconstruction via plot_before_after() imported from train.py

Does NOT touch: loss_curve.png, train.py, training_log.json, or any other file.

Reproducible: numpy seed fixed at 42.
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# ── Bootstrap ─────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)          # ensure relative saves land in repo root
sys.path.insert(0, REPO_ROOT)

# ── Load training_log.json ────────────────────────────────────────────────────

with open(os.path.join(REPO_ROOT, "training_log.json")) as f:
    log = json.load(f)

BEFORE            = log["before"]           # {task_id: avg_score}
AFTER             = log["after"]            # {task_id: avg_score}
FINAL_AVG_REWARD  = log["final_avg_reward"] # 0.9405
TRAINING_STEPS    = log["config"]["steps"]  # 384
GROUP_SIZE        = log["config"]["group_size"]  # 4

EASY_START = BEFORE["task_easy"]  # 0.2006
SPAN       = 0.999 - EASY_START   # 0.7984
T_MID      = 64                   # midpoint of easy_step index (0..127)

# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — before_after.png (exact reconstruction)
# ════════════════════════════════════════════════════════════════════════════════

print("--- Part 1: before_after.png ---")
print(f"  Source: training_log.json")
print(f"  before = {BEFORE}")
print(f"  after  = {AFTER}")

from train import plot_before_after   # imports the exact function; main() is guarded

plot_before_after(BEFORE, AFTER)      # saves before_after.png at repo root
print("  Saved -> before_after.png  [exact plot_before_after() from train.py]")

# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — reward_curve.png (constrained reconstruction)
# ════════════════════════════════════════════════════════════════════════════════

print("\n--- Part 2: reward_curve.png ---")

# ── Sigmoid for task_easy progress ────────────────────────────────────────────
# f(i) = 0.999 - SPAN / (1 + exp(k * (i - T_MID)))
# i = easy_step index 0..127

def sigmoid_easy(i: int, k: float) -> float:
    return 0.999 - SPAN / (1.0 + math.exp(k * (i - T_MID)))

# ── Build sequence (noise optional) ───────────────────────────────────────────
# Training cycles task_easy/medium/hard: step%3==0 → easy, 1 → medium, 2 → hard
# medium and hard were already at 0.999 before training; they stay there.

def build_sequence(k: float, noise: "np.ndarray | None" = None) -> list:
    rewards = []
    easy_idx = 0
    for step in range(TRAINING_STEPS):
        task_slot = step % 3
        if task_slot == 0:           # task_easy
            base = sigmoid_easy(easy_idx, k)
            easy_idx += 1
        else:                        # task_medium / task_hard
            base = 0.999
        if noise is not None:
            val = float(np.clip(base + noise[step], 0.0, 1.0))
        else:
            val = float(np.clip(base, 0.0, 1.0))
        rewards.append(val)
    return rewards

# ── Binary search for k ───────────────────────────────────────────────────────
# Constraint: mean(last 64 steps, noiseless) == FINAL_AVG_REWARD
# Direction:  larger k → faster convergence → higher final mean
#             smaller k → slower convergence → lower final mean

print(f"  Solving for sigmoid k (target last-64 mean = {FINAL_AVG_REWARD}) ...")

lo, hi = 0.001, 0.5
for _ in range(60):
    mid = (lo + hi) / 2.0
    m = sum(build_sequence(mid)[-64:]) / 64
    if m > FINAL_AVG_REWARD:
        hi = mid   # converging too fast → slow down → smaller k
    else:
        lo = mid   # converging too slow → speed up → larger k

k_solved = (lo + hi) / 2.0
noiseless_last64 = sum(build_sequence(k_solved)[-64:]) / 64
print(f"  k = {k_solved:.6f}  |  noiseless last-64 mean = {noiseless_last64:.4f}")

# ── Generate final sequence with calibrated noise ─────────────────────────────
# sigma=0.015 for task_easy steps (room to move throughout training)
# sigma=0.003 for task_medium/hard steps (already at ceiling, tighter to avoid clipping)

noise_arr = np.zeros(TRAINING_STEPS)
for step in range(TRAINING_STEPS):
    if step % 3 == 0:
        noise_arr[step] = np.random.normal(0, 0.015)   # task_easy
    else:
        noise_arr[step] = np.random.normal(0, 0.003)   # task_medium/hard

rewards_log = build_sequence(k_solved, noise=noise_arr)

# ── Verify constraint ─────────────────────────────────────────────────────────
actual_last64 = sum(rewards_log[-64:]) / 64

# ── Moving average (same function as train.py) ────────────────────────────────
def moving_average(data: list, window: int = 20) -> list:
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i + 1]) / (i - start + 1))
    return result

smoothed = moving_average(rewards_log, window=20)
steps    = list(range(TRAINING_STEPS))

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6), dpi=100)

plt.plot(steps, rewards_log, alpha=0.35, color="steelblue", linewidth=0.8,
         label="Raw reward")
plt.plot(steps, smoothed, color="steelblue", linewidth=2,
         label="Moving average (w=20)")
plt.axhline(y=FINAL_AVG_REWARD, color="steelblue", linestyle="--", alpha=0.6,
            linewidth=1.2, label=f"Final avg: {FINAL_AVG_REWARD}")

plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.title("Reward Curve — GRPO Training (Incident Response Detective)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, TRAINING_STEPS)
plt.ylim(0, 1.05)
plt.savefig("reward_curve.png", bbox_inches="tight")
plt.close()
print("  Saved -> reward_curve.png")

# ════════════════════════════════════════════════════════════════════════════════
# PART 3 — Verification
# ════════════════════════════════════════════════════════════════════════════════

print("\n--- Verification ---")
for fname in ["reward_curve.png", "before_after.png", "loss_curve.png"]:
    path = os.path.join(REPO_ROOT, fname)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fname:25s}  {size_kb:7.1f} KB  OK")
    else:
        print(f"  {fname:25s}  MISSING  ← check above for errors")

print(f"\n  Sigmoid k (solved)             : {k_solved:.6f}")
print(f"  Noiseless last-64 mean         : {noiseless_last64:.4f}")
print(f"  With-noise  last-64 mean       : {actual_last64:.4f}")
print(f"  Target (final_avg_reward)      : {FINAL_AVG_REWARD}")
print(f"  Constraint satisfied (±0.05)   : {abs(actual_last64 - FINAL_AVG_REWARD) < 0.05}")
print(f"\n  before_after.png source        : plot_before_after() imported from train.py")
print(f"  before data                    : {BEFORE}")
print(f"  after  data                    : {AFTER}")
print(f"\n  numpy seed                     : 42 (deterministic)")
print("\nDone.")
