"""Benchmark: 3 tasks × (standard, adversarial, runbook_injection) × 5 runs each + naive baseline.

Targeted reruns (merge into existing benchmark_results.json by default):

    python benchmark.py --tasks task_easy --modes adversarial --models large
    python benchmark.py --tasks task_easy,task_hard --modes runbook_injection --models small
    python benchmark.py --tasks task_medium --modes standard --models oracle,naive --no-merge
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from datetime import date
from typing import Literal

from environment import IncidentResponseEnvironment
from task_definitions import ACTIONS, ADVERSARIAL_OVERLAYS, TASKS
from inference import GROQ_MODEL_70B, GROQ_MODEL_8B, run_groq_agent

RUNS = 5
RESULTS_PATH = "benchmark_results.json"

CORRECT_ACTION = {
    "task_easy":   "rollback_deployment",
    "task_medium": "rollback_deployment",
    "task_hard":   "rotate_db_credentials",
}

# (mode label, reset kwargs) — extends the existing standard/adversarial loop
BENCHMARK_MODES: list[tuple[str, dict]] = [
    ("standard", {"injection_mode": "none"}),
    ("adversarial", {"adversarial": True}),
    ("runbook_injection", {"injection_mode": "runbook"}),
]

ModelKind = Literal["oracle", "naive", "large", "small"]

MODEL_KIND_ALIASES: dict[str, ModelKind] = {
    "oracle": "oracle",
    "naive": "naive",
    "large": "large",
    "small": "small",
    "70b": "large",
    "8b": "small",
    "120b": "large",
    "20b": "small",
    "gpt-oss-120b": "large",
    "gpt-oss-20b": "small",
    GROQ_MODEL_70B: "large",
    GROQ_MODEL_8B: "small",
    "openai/gpt-oss-120b": "large",
    "openai/gpt-oss-20b": "small",
}


def naive_action_for(task_id: str, mode: str) -> str:
    """Return the ACTIONS name most frequently mentioned in the (possibly adversarial) chat."""
    use_adversarial_chat = mode == "adversarial"
    chat = (
        ADVERSARIAL_OVERLAYS[task_id]
        if use_adversarial_chat and task_id in ADVERSARIAL_OVERLAYS
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
    reset_kwargs: dict,
    action: str,
) -> float:
    episode_id, obs = env.reset(task_id=task_id, **reset_kwargs)
    # Use log index 0 as evidence — always in-bounds
    env.step(episode_id, {"action": action, "evidence": 0})
    return env.grade(episode_id)["score"]


def collect(
    env: IncidentResponseEnvironment,
    task_id: str,
    reset_kwargs: dict,
    action: str,
) -> list[float]:
    return [run_once(env, task_id, reset_kwargs, action) for _ in range(RUNS)]


def stats(scores: list[float]) -> tuple[float, float]:
    avg = sum(scores) / len(scores)
    std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))
    return round(avg, 4), round(std, 4)


def run_groq_episodes(
    task_id: str,
    groq_kwargs: dict,
    model: str | None = None,
) -> tuple[list[float], int, int]:
    """Run RUNS Groq episodes; return scores and fallback/real API counts."""
    scores: list[float] = []
    fallback_count = 0
    real_call_count = 0
    kwargs = dict(groq_kwargs)
    if model is not None:
        kwargs["model"] = model
    for _ in range(RUNS):
        score, used_fallback = run_groq_agent(task_id, **kwargs)
        scores.append(score)
        if used_fallback:
            fallback_count += 1
        else:
            real_call_count += 1
    return scores, fallback_count, real_call_count


def is_groq_result_row(model_label: str) -> bool:
    """True for LLM benchmark rows (not oracle or naive baselines)."""
    return not model_label.startswith("naive(") and model_label != "oracle"


def row_key(row: dict) -> tuple[str, str, str]:
    return row["task"], row["mode"], row["model"]


def parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


def resolve_tasks(raw: str | None) -> list[str]:
    selected = parse_csv_list(raw)
    if selected is None:
        return list(TASKS.keys())
    unknown = [t for t in selected if t not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown --tasks: {unknown}. Valid: {list(TASKS.keys())}")
    return selected


def resolve_modes(raw: str | None) -> list[tuple[str, dict]]:
    selected = parse_csv_list(raw)
    known = {name: kwargs for name, kwargs in BENCHMARK_MODES}
    if selected is None:
        return list(BENCHMARK_MODES)
    unknown = [m for m in selected if m not in known]
    if unknown:
        raise SystemExit(f"Unknown --modes: {unknown}. Valid: {list(known.keys())}")
    return [(name, known[name]) for name in selected]


def resolve_model_kinds(raw: str | None) -> set[ModelKind] | None:
    selected = parse_csv_list(raw)
    if selected is None:
        return None
    kinds: set[ModelKind] = set()
    unknown: list[str] = []
    for item in selected:
        kind = MODEL_KIND_ALIASES.get(item.lower())
        if kind is None:
            unknown.append(item)
        else:
            kinds.add(kind)
    if unknown:
        valid = sorted(set(MODEL_KIND_ALIASES.keys()))
        raise SystemExit(f"Unknown --models: {unknown}. Valid aliases: {valid}")
    return kinds


def groq_kwargs_for_mode(mode: str) -> dict:
    if mode == "adversarial":
        return {"adversarial": True}
    if mode == "runbook_injection":
        return {"injection_mode": "runbook"}
    return {}


def run_benchmark_cell(
    env: IncidentResponseEnvironment,
    task_id: str,
    mode: str,
    reset_kwargs: dict,
    model_kinds: set[ModelKind] | None,
    groq_model_label: str,
    groq_8b_label: str,
) -> list[dict]:
    """Run selected model variants for one (task, mode) cell."""
    rows: list[dict] = []
    run = model_kinds is None

    if run or "oracle" in model_kinds:
        oracle_scores = collect(env, task_id, reset_kwargs, CORRECT_ACTION[task_id])
        avg, std = stats(oracle_scores)
        rows.append({
            "task": task_id, "mode": mode, "model": "oracle",
            "avg_score": avg, "std_dev": std,
            "real_call_count": 0, "fallback_count": 0,
        })

    if run or "naive" in model_kinds:
        naive = naive_action_for(task_id, mode)
        naive_scores = collect(env, task_id, reset_kwargs, naive)
        avg_n, std_n = stats(naive_scores)
        rows.append({
            "task": task_id, "mode": mode, "model": f"naive({naive})",
            "avg_score": avg_n, "std_dev": std_n,
            "real_call_count": 0, "fallback_count": 0,
        })

    groq_kwargs = groq_kwargs_for_mode(mode)

    if run or "large" in model_kinds:
        groq_scores, fb_count, real_count = run_groq_episodes(task_id, groq_kwargs)
        avg_g, std_g = stats(groq_scores)
        rows.append({
            "task": task_id, "mode": mode, "model": groq_model_label,
            "avg_score": avg_g, "std_dev": std_g,
            "real_call_count": real_count, "fallback_count": fb_count,
        })

    if run or "small" in model_kinds:
        groq_8b_scores, fb_8b, real_8b = run_groq_episodes(
            task_id, groq_kwargs, model=GROQ_MODEL_8B
        )
        avg_8b, std_8b = stats(groq_8b_scores)
        rows.append({
            "task": task_id, "mode": mode, "model": groq_8b_label,
            "avg_score": avg_8b, "std_dev": std_8b,
            "real_call_count": real_8b, "fallback_count": fb_8b,
        })

    return rows


def merge_result_rows(existing: list[dict], updated: list[dict]) -> list[dict]:
    """Replace matching (task, mode, model) rows; preserve order of existing file."""
    updates = {row_key(r): r for r in updated}
    merged = [updates.pop(row_key(r), r) for r in existing]
    merged.extend(updates.values())
    return merged


def load_existing_results(path: str) -> list[dict] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data if isinstance(data, list) else None)


def print_contamination_summary(rows: list[dict]) -> None:
    """Warn when Groq rows used any deterministic fallback."""
    contaminated = [
        r for r in rows
        if is_groq_result_row(r["model"]) and r.get("fallback_count", 0) > 0
    ]
    if not contaminated:
        return
    print(
        "\n## Contaminated results (fallback present — do not trust avg_score)\n",
        file=sys.stderr,
    )
    for r in contaminated:
        print(
            f"  - {r['task']} | {r['mode']} | {r['model']}: "
            f"real_call_count={r['real_call_count']}, "
            f"fallback_count={r['fallback_count']} / {RUNS} runs",
            file=sys.stderr,
        )
    print(
        "\nRerun a cell: python benchmark.py "
        "--tasks <task> --modes <mode> --models large|small|oracle|naive",
        file=sys.stderr,
    )


def render_table(rows: list[dict]) -> None:
    cols = ["task", "mode", "model", "avg_score", "std_dev"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep    = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cross-validate oracle, naive, and Groq agents across tasks and modes.",
    )
    ap.add_argument(
        "--tasks",
        metavar="ID",
        help="Comma-separated task ids (default: all). Example: task_easy,task_hard",
    )
    ap.add_argument(
        "--modes",
        metavar="MODE",
        help="Comma-separated modes (default: all). "
        "Choices: standard, adversarial, runbook_injection",
    )
    ap.add_argument(
        "--models",
        metavar="MODEL",
        help="Comma-separated model kinds to run per cell (default: all). "
        "Aliases: oracle, naive, large, small, 120b, 20b, "
        f"{GROQ_MODEL_70B}, {GROQ_MODEL_8B}",
    )
    ap.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Merge rerun rows into existing benchmark_results.json "
        "(default: on when any filter is set, off for full sweep)",
    )
    ap.add_argument(
        "--output",
        default=RESULTS_PATH,
        help=f"Results JSON path (default: {RESULTS_PATH})",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_ids = resolve_tasks(args.tasks)
    modes = resolve_modes(args.modes)
    model_kinds = resolve_model_kinds(args.models)

    filtered = (
        args.tasks is not None
        or args.modes is not None
        or args.models is not None
    )
    merge = args.merge if args.merge is not None else filtered

    groq_available = bool(os.environ.get("GROQ_API_KEY", ""))
    if groq_available:
        print(
            f"GROQ_API_KEY found — {GROQ_MODEL_70B} and {GROQ_MODEL_8B} will run live"
        )
        groq_model_label = GROQ_MODEL_70B
        groq_8b_label = GROQ_MODEL_8B
    else:
        print(
            f"WARNING: GROQ_API_KEY not set — using deterministic_fallback(), "
            f"NOT calling {GROQ_MODEL_70B} or {GROQ_MODEL_8B}.",
            file=sys.stderr,
        )
        groq_model_label = f"deterministic_fallback (not {GROQ_MODEL_70B})"
        groq_8b_label = f"deterministic_fallback (not {GROQ_MODEL_8B})"

    if filtered:
        scope = (
            f"tasks={task_ids}, modes={[m for m, _ in modes]}, "
            f"models={sorted(model_kinds) if model_kinds else 'all'}"
        )
        print(f"Targeted rerun: {scope}" + (" (merge on)" if merge else " (merge off)"))

    env = IncidentResponseEnvironment()
    new_rows: list[dict] = []

    for task_id in task_ids:
        for mode, reset_kwargs in modes:
            new_rows.extend(
                run_benchmark_cell(
                    env, task_id, mode, reset_kwargs, model_kinds,
                    groq_model_label, groq_8b_label,
                )
            )

    if merge:
        existing = load_existing_results(args.output)
        if existing is not None:
            rows = merge_result_rows(existing, new_rows)
            print(f"Merged {len(new_rows)} row(s) into {args.output}")
        else:
            rows = new_rows
            if filtered:
                print(
                    f"WARNING: {args.output} not found — writing {len(rows)} row(s) only",
                    file=sys.stderr,
                )
    else:
        rows = new_rows

    print("\n## Benchmark Results\n")
    render_table(new_rows)
    print_contamination_summary(rows)

    command = "python benchmark.py"
    if argv is not None:
        command += " " + " ".join(argv)
    elif len(sys.argv) > 1:
        command += " " + " ".join(sys.argv[1:])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "generated_at": date.today().isoformat(),
                    "command": command,
                    "live_api": groq_available,
                    "filtered_run": filtered,
                    "filter": {
                        "tasks": task_ids,
                        "modes": [m for m, _ in modes],
                        "models": sorted(model_kinds) if model_kinds else None,
                    } if filtered else None,
                    "merged": merge,
                    "groq_models": {
                        "large": GROQ_MODEL_70B,
                        "small": GROQ_MODEL_8B,
                    } if groq_available else None,
                    "results_model_labels": {
                        "large": groq_model_label,
                        "small": groq_8b_label,
                    },
                    "runs_per_cell": RUNS,
                    "notes": (
                        "Verified live GROQ_API_KEY run (not deterministic_fallback)."
                        if groq_available
                        else "GROQ_API_KEY not set — LLM row used deterministic_fallback."
                    ),
                },
                "results": rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
