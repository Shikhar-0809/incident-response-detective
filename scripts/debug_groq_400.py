"""One-shot reproducer for Groq 400 on task_medium + runbook injection.

Usage:
    GROQ_API_KEY=... python scripts/debug_groq_400.py
    GROQ_API_KEY=... python scripts/debug_groq_400.py --via-agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import IncidentResponseEnvironment
from inference import (
    GROQ_API_URL,
    GROQ_MODEL_70B,
    GROQ_MODEL_8B,
    GROQ_SYSTEM_PROMPT,
    _groq_error_message,
    build_groq_payload,
    build_groq_prompt,
    run_groq_agent,
)


def preflight(task_id: str, injection_mode: str) -> tuple[dict, str, dict]:
    env = IncidentResponseEnvironment()
    _, obs = env.reset(task_id=task_id, injection_mode=injection_mode)
    user_prompt = build_groq_prompt(obs)
    payload = build_groq_payload(GROQ_MODEL_70B, user_prompt)
    return obs, user_prompt, payload


def print_preflight(obs: dict, user_prompt: str, payload: dict) -> None:
    combined = GROQ_SYSTEM_PROMPT + user_prompt
    runbook = obs.get("runbook", "")
    try:
        serialized = json.dumps(payload)
        serial_ok = True
    except (TypeError, ValueError) as exc:
        serialized = f"<serialization failed: {exc}>"
        serial_ok = False

    print("=== Pre-flight ===")
    print(f"runbook chars: {len(runbook)}")
    print(f"user_prompt chars: {len(user_prompt)}")
    print(f"system_prompt chars: {len(GROQ_SYSTEM_PROMPT)}")
    print(f"total message chars: {len(combined)}")
    print(f"rough token estimate (chars/4): {len(combined) // 4}")
    print(f"'json' in messages (case-insensitive): {'json' in combined.lower()}")
    print(f"payload json.dumps OK: {serial_ok}")
    print(f"serialized payload bytes: {len(serialized.encode('utf-8'))}")
    print(f"runbook has [SYSTEM OVERRIDE: {('[SYSTEM OVERRIDE' in runbook)}")
    print(
        "runbook control chars (excl \\n\\t\\r): "
        f"{any(ord(c) < 32 and c not in chr(10)+chr(13)+chr(9) for c in runbook)}"
    )
    print()


def post_raw(payload: dict) -> int:
    import requests

    print("=== Request payload ===")
    print(json.dumps(payload, indent=2))
    print()

    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    print(f"=== Response status: {resp.status_code} ===")
    print(f"Groq error.message: {_groq_error_message(resp)}")
    print(f"Full response body:\n{resp.text}")

    if resp.ok:
        print("\n=== Parsed assistant content ===")
        print(resp.json()["choices"][0]["message"]["content"])

    return 0 if resp.ok else 2


def compare_models(task_id: str, injection_mode: str) -> int:
    """POST the same cell to 120b and 20b to isolate model-specific 400s."""
    import requests

    env = IncidentResponseEnvironment()
    _, obs = env.reset(task_id=task_id, injection_mode=injection_mode)
    user_prompt = build_groq_prompt(obs)
    payload_120 = build_groq_payload(GROQ_MODEL_70B, user_prompt)
    print_preflight(obs, user_prompt, payload_120)
    exit_code = 0
    for model in (GROQ_MODEL_70B, GROQ_MODEL_8B):
        payload = build_groq_payload(model, user_prompt)
        print(f"\n=== {model} ===")
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        print(f"status={resp.status_code}")
        print(f"error.message={_groq_error_message(resp)}")
        if resp.status_code != 200:
            exit_code = 2
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--via-agent", action="store_true", help="Call run_groq_agent() end-to-end")
    ap.add_argument("--compare-models", action="store_true", help="POST 120b and 20b")
    ap.add_argument("--task", default="task_medium")
    ap.add_argument("--injection-mode", default="runbook")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set", file=sys.stderr)
        return 1

    if args.via_agent:
        print("=== run_groq_agent() ===")
        score, used_fallback = run_groq_agent(
            args.task,
            injection_mode=args.injection_mode,
            model=GROQ_MODEL_70B,
        )
        print(f"score={score} used_fallback={used_fallback}")
        return 0 if not used_fallback else 2

    if args.compare_models:
        return compare_models(args.task, args.injection_mode)

    _, user_prompt, payload = preflight(args.task, args.injection_mode)
    env = IncidentResponseEnvironment()
    _, obs = env.reset(task_id=args.task, injection_mode=args.injection_mode)
    print_preflight(obs, user_prompt, payload)
    return post_raw(payload)


if __name__ == "__main__":
    raise SystemExit(main())
