"""
Regenerate all training / evaluation plots from the authoritative TRL state export.

**Single source of truth:** `data/trainer_state.json`
  (download with `python scripts/download_training_data.py` if missing)

TRL's GRPOTrainer writes `trainer_state.json` at the end of training. The `log_history`
array holds one record per *logging* step with at least:
  - `step`  — global optimizer step
  - `loss`  — GRPO / policy loss (can be negative)
  - `reward` and `rewards/reward_fn/mean` — mean group reward for that step

This script *does not* read `training_log.json` (Groq harness) — only the on-Kaggle TRL run.

Outputs (repo root):
  - loss_curve.png
  - reward_curve.png
  - before_after.png
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

TRAINER_STATE_PATH = os.path.join(REPO_ROOT, "data", "trainer_state.json")


def moving_average(x: list[float], window: int) -> list[float]:
    if window < 1:
        return list(x)
    out = []
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(x[lo : i + 1])))
    return out


def load_trainer_state() -> dict:
    if not os.path.exists(TRAINER_STATE_PATH):
        print(
            f"ERROR: {TRAINER_STATE_PATH} not found.\n"
            "  Run:  python scripts/download_training_data.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with open(TRAINER_STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def extract_series(state: dict) -> tuple[list[int], list[float], list[float], int, int]:
    """
    Return (step_ids, losses, rewards, max_steps, n_logs).

    We prefer the `reward` key; it matches `rewards/reward_fn/mean` in this export.
    """
    history = state.get("log_history") or []
    if not history:
        raise ValueError("trainer_state.json has empty log_history — nothing to plot.")

    steps: list[int] = []
    losses: list[float] = []
    rewards: list[float] = []

    for row in history:
        s = int(row.get("step", len(steps) + 1))
        steps.append(s)
        losses.append(float(row.get("loss", 0.0)))
        r = row.get("reward", row.get("rewards/reward_fn/mean", 0.0))
        rewards.append(float(r))

    max_steps = int(state.get("max_steps", len(steps)))
    n_logs = len(steps)
    return steps, losses, rewards, max_steps, n_logs


def set_title_with_source(axe: plt.Axes, main: str, max_steps: int) -> None:
    """Title + second line: explicit TRL source and step count (from trainer_state, not hardcoded)."""
    axe.set_title(
        f"{main}\nSource: TRL trainer_state.json, {max_steps} training steps",
        fontsize=11,
    )


def plot_loss_curve(steps: list[int], losses: list[float], max_steps: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(
        steps,
        losses,
        color="tomato",
        alpha=0.4,
        linewidth=0.9,
        label="Raw loss",
    )
    if len(losses) >= 20:
        sm = moving_average(losses, 20)
        ax.plot(steps, sm, color="tomato", linewidth=2, label="Moving average (w=20)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Policy loss (GRPO)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max(steps) if steps else 1)
    set_title_with_source(ax, "Training loss (GRPO)", max_steps)
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "loss_curve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)")


def plot_reward_curve(steps: list[int], rewards: list[float], max_steps: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(
        steps,
        rewards,
        color="steelblue",
        alpha=0.35,
        linewidth=0.8,
        label="Per-step mean reward",
    )
    if len(rewards) >= 20:
        sm = moving_average(rewards, 20)
        ax.plot(
            steps,
            sm,
            color="steelblue",
            linewidth=2,
            label="Moving average (w=20)",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean group reward")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max(steps) if steps else 1)
    set_title_with_source(ax, "Mean reward during GRPO training", max_steps)
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "reward_curve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)")


def plot_before_after(rewards: list[float], max_steps: int, window: int = 50) -> None:
    """
    Bar chart: average reward in the first `window` logged steps vs the last `window`.

    If fewer than 2*window points, shrink the window to floor(n/2) per side.
    """
    n = len(rewards)
    w = min(window, n // 2) if n >= 2 else 1
    if w < 1:
        w = 1

    early = float(np.mean(rewards[:w]))
    late = float(np.mean(rewards[-w:]))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    x = [0, 1]
    labels = [f"First {w} steps (avg)", f"Last {w} steps (avg)"]
    bars = ax.bar(
        [0, 1],
        [early, late],
        width=0.45,
        color=["salmon", "seagreen"],
        alpha=0.9,
    )
    for b, v in zip(bars, (early, late), strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("Average mean reward")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    set_title_with_source(
        ax,
        "Early vs late training (mean reward)",
        max_steps,
    )
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "before_after.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)")


def main() -> None:
    print("--- regenerate_plots.py ---")
    print(f"  Loading {TRAINER_STATE_PATH}")

    state = load_trainer_state()
    steps, losses, rewards, max_steps, n_logs = extract_series(state)

    print(
        f"  log_history entries: {n_logs}  |  max_steps in state: {max_steps}  |  last step: {steps[-1] if steps else '—'}"
    )
    print(
        f"  reward: min={min(rewards):.4f} max={max(rewards):.4f}  |  loss: min={min(losses):.4f} max={max(losses):.4f}"
    )

    plot_loss_curve(steps, losses, max_steps)
    plot_reward_curve(steps, rewards, max_steps)
    plot_before_after(rewards, max_steps, window=50)

    print("Done. All three PNGs generated from data/trainer_state.json.")


if __name__ == "__main__":
    main()
