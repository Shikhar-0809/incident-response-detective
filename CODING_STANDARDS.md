# Coding Standards

## Python

- Python 3.10+ required (match `pyproject.toml`)
- Type hints on all function signatures and return types
- Module docstring on every `.py` file explaining its purpose and role in the project
- No bare `except Exception` without logging — catch specific errors or at minimum log the exception with context
- No unused imports — enforce with `ruff check` or `flake8`
- No dead code — if a function, class, or import is not called anywhere, delete it
- Use `from __future__ import annotations` when forward references are needed

## Naming

- Files: `snake_case.py`
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Episode/task dicts use string keys (OpenEnv compat)

## Security

- No secrets in code. Use environment variables: `GROQ_API_KEY`, `HF_TOKEN`, `API_BASE_URL`
- No hardcoded local filesystem paths (e.g., `C:\Users\...`). Use env vars, CLI args, or relative paths
- `.gitignore` must cover: `__pycache__`, `*.pyc`, `.env`, `secrets`, `full_dump.txt`
- Before committing: verify no API keys, tokens, or credentials in any file or git history
- If a secret is accidentally committed: rotate the key AND rewrite git history (`git filter-branch` or `git filter-repo`)

## Documentation

- README claims must match code exactly — no aspirational or rounded numbers
- Two scoring systems (`compute_reward` vs `grade`) must never be conflated in docs or comments
- Pipeline A vs Pipeline B must be labeled on every plot, metric, and data file
- All numbers in README must trace to a specific file, function, and data source
- When a metric is quoted, parenthetically note the source: e.g., "(from `grade()` in `server/environment.py`)"

## File hygiene

- No generated files committed unless they are canonical outputs (e.g., `benchmark_results.json` + `benchmark_results.md`)
- Plot PNGs in repo root come from `regenerate_plots.py` only — `eval_harness.py` saves to `pipeline_b/`
- Upload/utility scripts must not contain hardcoded local paths
- `.gitignore` should include: `pipeline_b/`, `full_dump.txt`, `.env`
- No self-referential files (don't run dump commands that include the dump output itself)

## Testing

- `python eval_harness.py --dry-run` must pass before each commit
- `openenv validate` must pass (enforced in CI via `.github/workflows/openenv-validate.yml`)
- After fixing a bug, verify with the simplest possible test:
  - BUG-1: `reset(adversarial=True)` → `step()` → check `chat_history` in returned observation
  - BUG-2/3: `step()` with evidence → verify no penalty message in feedback
- Deterministic fallback (`inference.py`) must produce `0.999` on all three standard tasks

## Dependencies

- Keep `requirements.txt` and `pyproject.toml` in sync
- Pin exact versions for reproducibility where possible (e.g., `matplotlib==3.9.0`)
- Separate runtime deps (in `requirements.txt`) from dev/eval deps (torch, transformers, peft — documented in scripts but not in requirements.txt)
