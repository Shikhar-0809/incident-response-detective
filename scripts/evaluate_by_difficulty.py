#!/usr/bin/env python3
"""
Evaluate Qwen2.5-0.5B (base) vs. LoRA adapter (Shiggii/qwen-incident-response-grpo)
on ``IncidentResponseEnvironment`` from ``server.environment``, by difficulty, **adversarial** mode.

**Difficulty → task mapping (aligned with project docs)**
  - **Easy**   — ``task_easy``  + ``adversarial=True``  (Slack “authority” pushes a dangerous fix; runbook says rollback)
  - **Medium** — ``task_medium`` + ``adversarial=True``  (unanimous wrong chat; runbook prohibits flush, correct fix is rollback)
  - **Hard**   — ``task_hard``  + ``adversarial=True``  (command pressure to rollback; true fix is ``rotate_db_credentials``)

**Reference plot numbers** (e.g. Untrained 0.201 / 0.999 / 0.999) match ``training_log.json`` **before** row and were obtained with
the **Groq** harness and ``llama-3.1-8b-instant``, not local Qwen. This script is the *reproducible local Qwen* protocol: means will not
identically match every decimal, but the qualitative pattern (train improves Easy; Medium/Hard often near ceiling) should hold.

**Dependencies** (not in the minimal app ``requirements.txt``)::

    pip install torch transformers peft accelerate

**Run** (GPU strongly recommended)::

    set HF_TOKEN=hf_...   & :: if adapter is private
    python scripts/evaluate_by_difficulty.py
    python scripts/evaluate_by_difficulty.py --plot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_ID = "Shiggii/qwen-incident-response-grpo"

# Match ``training_log.json`` regime: all tasks evaluated in adversarial mode
ADVERSARIAL = True
EPISODES_PER_CELL = 10
EVALTEMP = float(os.environ.get("EVAL_TEMPERATURE", "0.8"))

TASK_ORDER = [("task_easy", "Easy"), ("task_medium", "Medium"), ("task_hard", "Hard")]


def import_deps():
    from inference import GROQ_SYSTEM_PROMPT, build_groq_prompt, deterministic_fallback
    from server.environment import IncidentResponseEnvironment

    return GROQ_SYSTEM_PROMPT, build_groq_prompt, deterministic_fallback, IncidentResponseEnvironment


def import_torch():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(
            "Missing dependencies. Install:\n  pip install torch transformers peft accelerate",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def load_model(torch, am, tok_cls, peft, trained: bool, hf_token: str | None):
    tok = tok_cls.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=hf_token
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    else:
        kwargs = {"device_map": None, "torch_dtype": torch.float32}
    base = am.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=hf_token, **kwargs
    )
    if kwargs.get("device_map") is None:
        base = base.to("cpu")
    if trained:
        m = peft.from_pretrained(
            base, ADAPTER_ID, is_trainable=False, token=hf_token
        )
    else:
        m = base
    m.eval()
    return tok, m


def run_one_episode(
    torch,
    model,
    tokenizer,
    env,
    system_prompt: str,
    build_groq_prompt,
    deterministic_fallback,
    task_id: str,
    seed: int,
) -> float:
    import torch as T
    import re

    random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)

    episode_id, obs = env.reset(task_id=task_id, adversarial=ADVERSARIAL)
    log_count = len(obs.get("logs", []))
    user = build_groq_prompt(obs)

    def finish(action: str, evidence: int) -> float:
        if log_count:
            evidence = max(0, min(int(evidence), log_count - 1))
        else:
            evidence = 0
        env.step({"action": action, "evidence": evidence}, episode_id=episode_id)
        return float(env.grade(episode_id)["score"])

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n\n{user}"
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with T.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=EVALTEMP > 0,
                temperature=EVALTEMP if EVALTEMP > 0 else 1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        new_tokens = gen[0, in_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception:  # noqa: BLE001
        fb = deterministic_fallback(obs)
        return finish(fb["action"], 0)

    if "```" in text:
        part = text.split("```", 1)[1]
        if part.startswith("json"):
            part = part[4:]
        text = part.strip()
    try:
        parsed = json.loads(text)
    except Exception as e:
        print(
            f"[warn] {task_id} seed={seed}: JSON parse failed ({type(e).__name__}); using deterministic fallback",
            file=sys.stderr,
        )
        fb = deterministic_fallback(obs)
        return finish(fb["action"], 0)

    # Robust handling for malformed-but-parseable outputs:
    # - expected: dict with {"action": "...", "evidence": N}
    # - bad case: JSON string containing free-form text
    if isinstance(parsed, dict):
        act = parsed.get("action", "notify_cto")
        try:
            ev = int(parsed.get("evidence", 0))
        except (TypeError, ValueError):
            ev = 0
        return finish(act, ev)

    if isinstance(parsed, str):
        # Try to salvage an action name from plain text.
        # Accept either explicit `action: xyz` or any known action token.
        action_regex = re.compile(
            r"(rollback_deployment|scale_infrastructure|flush_redis_cache|notify_cto|restart_api_gateway|rotate_db_credentials|enable_circuit_breaker|purge_cdn_cache)"
        )
        m = action_regex.search(parsed)
        if m:
            act = m.group(1)
            ev_match = re.search(r"evidence\\s*[:=]\\s*(-?\\d+)", parsed, flags=re.IGNORECASE)
            try:
                ev = int(ev_match.group(1)) if ev_match else 0
            except (TypeError, ValueError):
                ev = 0
            print(
                f"[warn] {task_id} seed={seed}: parsed JSON string; recovered action={act}",
                file=sys.stderr,
            )
            return finish(act, ev)

        print(
            f"[warn] {task_id} seed={seed}: parsed JSON string without valid action; using deterministic fallback",
            file=sys.stderr,
        )
        fb = deterministic_fallback(obs)
        return finish(fb["action"], 0)

    print(
        f"[warn] {task_id} seed={seed}: parsed JSON type={type(parsed).__name__}; using deterministic fallback",
        file=sys.stderr,
    )
    fb = deterministic_fallback(obs)
    return finish(fb["action"], 0)


def run_grid(
    torch,
    model,
    tokenizer,
    env,
    system_prompt: str,
    build_groq_prompt,
    deterministic_fallback,
    episodes: int,
    base_seed: int,
    seed_offset: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for ti, (tid, _label) in enumerate(TASK_ORDER):
        scores: list[float] = []
        for epi in range(episodes):
            # Deterministic across runs: separate ranges per task index + ep index + model offset
            seed = base_seed + seed_offset + ti * 10_000 + epi
            s = run_one_episode(
                torch,
                model,
                tokenizer,
                env,
                system_prompt,
                build_groq_prompt,
                deterministic_fallback,
                tid,
                seed,
            )
            scores.append(s)
        out[tid] = sum(scores) / len(scores)
    return out


def maybe_plot(untrained: dict[str, float], trained: dict[str, float], outpath: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lbl for _tid, lbl in TASK_ORDER]
    x = range(len(labels))
    w = 0.35
    uvals = [untrained["task_easy"], untrained["task_medium"], untrained["task_hard"]]
    tvals = [trained["task_easy"], trained["task_medium"], trained["task_hard"]]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.bar([i - w / 2 for i in x], uvals, w, label="Untrained (base Qwen0.5B)", color="salmon")
    ax.bar([i + w / 2 for i in x], tvals, w, label="Trained (LoRA)", color="seagreen")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Mean grader score (10 eps)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Evaluation by difficulty (adversarial mode) — Qwen2.5-0.5B + LoRA")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Write evaluation_by_difficulty.png in repo root",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_CELL,
        help="Episodes per (model, difficulty) cell (default: 10)",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42)",
    )
    args = ap.parse_args()

    GROQ_SYSTEM_PROMPT, build_groq_prompt, det_fb, IrEnv = import_deps()
    torch, am, tok_cls, peft = import_torch()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    print("--- evaluate_by_difficulty.py ---")
    print(f"  Base: {BASE_MODEL}")
    print(f"  Adapter: {ADAPTER_ID} (trained run)")
    print(f"  Adversarial: {ADVERSARIAL}  |  episodes/cell: {args.episodes}  |  temp: {EVALTEMP}")
    print("  Reference (README / training_log.json before, Groq 8B): Easy ~0.20, Med/Hard ~0.999\n")

    results_untrained: dict[str, float] = {}
    results_trained: dict[str, float] = {}

    env = IrEnv()

    print("[1/2] Untrained (base only)...")
    tok, model = load_model(torch, am, tok_cls, peft, False, hf_token)
    results_untrained = run_grid(
        torch,
        model,
        tok,
        env,
        GROQ_SYSTEM_PROMPT,
        build_groq_prompt,
        det_fb,
        args.episodes,
        args.base_seed,
        seed_offset=0,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[2/2] Trained (LoRA)...")
    tok, model = load_model(torch, am, tok_cls, peft, True, hf_token)
    results_trained = run_grid(
        torch,
        model,
        tok,
        env,
        GROQ_SYSTEM_PROMPT,
        build_groq_prompt,
        det_fb,
        args.episodes,
        args.base_seed,
        seed_offset=1_000_000,
    )

    # Print table
    print("\n=== Mean grader score (adversarial, 1 step, env.grade) ===\n")
    print(f"{'':12} {'Easy':>10} {'Medium':>10} {'Hard':>10}")
    print(
        f"{'Untrained':12} {results_untrained['task_easy']:10.3f} "
        f"{results_untrained['task_medium']:10.3f} {results_untrained['task_hard']:10.3f}"
    )
    print(
        f"{'Trained':12} {results_trained['task_easy']:10.3f} "
        f"{results_trained['task_medium']:10.3f} {results_trained['task_hard']:10.3f}"
    )
    print(
        f"\nReference (training_log before/after, different model): "
        f"0.201 / 0.999 / 0.999  →  0.999 / 0.999 / 0.999"
    )

    if args.plot:
        outp = os.path.join(ROOT, "evaluation_by_difficulty.png")
        maybe_plot(results_untrained, results_trained, outp)


if __name__ == "__main__":
    main()
