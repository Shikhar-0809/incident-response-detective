# CLAUDE.md — Incident Response Detective

## What this project is
OpenEnv-compliant RL environment for training agents to triage production incidents under conflicting signals (logs, chat, runbook). Meta × HuggingFace OpenEnv Hackathon submission (Top 100 of 31,000+ teams, team optimusCryme).

## Architecture (read ARCHITECTURE.md for full detail)
- `server/environment.py` — canonical OpenEnv Environment subclass. Single source of truth for reset/step/grade.
- `environment.py` — thin compatibility shim wrapping server/environment.py for legacy scripts.
- `task_definitions.py` — scenario data, action space, reward logic, adversarial overlays.
- `server/app.py` — FastAPI server exposing the environment over HTTP.
- `inference.py` — agent script (LLM + deterministic fallback).
- `eval_harness.py` — Pipeline B: Groq evaluation harness (does NOT train weights).
- `benchmark.py` — cross-validation across oracle/naive/LLM baselines.
- `procedural_generator.py` — EXPERIMENTAL infinite scenario generation (not wired into runtime)
- `regenerate_plots.py` — generates plots from Pipeline A data (data/trainer_state.json).

## Key invariants
- `grade()` is the ONLY scoring function that matters for final results. `compute_reward()` is per-step feedback only.
- Adversarial mode swaps chat_history only. Logs and runbook are identical.
- All scripts must send `evidence` (int) alongside `action` to avoid the -0.1 penalty.
- Plots in repo root MUST come from Pipeline A (regenerate_plots.py). eval_harness.py must NOT overwrite them.
- Two pipelines exist and must never be conflated:
  - **Pipeline A** — Real GRPO training of Qwen 2.5-0.5B on Kaggle. Produces `data/trainer_state.json` and the HF adapter.
  - **Pipeline B** — Groq API evaluation harness using llama-3.1-8b-instant. Produces `training_log.json`. No weight updates.

## Commands
- `uvicorn server.app:app --host 0.0.0.0 --port 7860` — run server
- `python inference.py` — run deterministic baseline (standard mode)
- `ADVERSARIAL=true python inference.py` — run deterministic baseline in adversarial mode
- `python benchmark.py` — full cross-validation
- `python eval_harness.py --dry-run` — validate environment setup
- `python regenerate_plots.py` — regenerate plots from trainer_state.json
- `openenv validate` — OpenEnv spec validation (also runs in CI; see `.github/workflows/openenv-validate.yml`)
- `docker build -t incident-response-detective . && docker run --rm -p 7860:7860 incident-response-detective` — Docker

## Code style
- Python 3.10+, type hints on all function signatures
- No unused imports, no dead code, no unused dataclasses
- Raw dicts for environment I/O (not dataclasses) — this is intentional for OpenEnv compat
- Every file has a module docstring explaining its role
- No hardcoded local paths (use env vars or CLI args)
- No secrets in code or git history

## Known patterns
- `compute_reward()` returns per-step reward dict used inside `step()`
- `grade()` returns final episode score used by all evaluation scripts
- These are DIFFERENT calculations — never conflate them in docs
- Evidence penalty applies in `step()` only, does NOT affect `grade()` output
