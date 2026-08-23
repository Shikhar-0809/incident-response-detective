# Known Bugs & Discrepancies — Fix All Before Proceeding

Status key: ⬜ Not started | 🔧 In progress | ✅ Fixed

---

## 🔴 Critical

### BUG-1: step() returns wrong chat in adversarial mode ✅
**File:** `server/environment.py`
**Problem:** `step()` always returns `task["observation"]["chat_history"]` (standard), even when the episode was started with `adversarial=True`. The adversarial overlay is only applied in `reset()`. This means after step 1, the agent sees standard (non-adversarial) chat in subsequent observations. The invalid-action branch has the same bug.
**Fix:** In `step()`, check `ep["adversarial"]` and return the correct chat_history using `ADVERSARIAL_OVERLAYS`. Apply the same logic to the invalid-action return path. Both branches must use:
```python
chat_history = (
    ADVERSARIAL_OVERLAYS[ep["task_id"]]
    if ep["adversarial"] and ep["task_id"] in ADVERSARIAL_OVERLAYS
    else task["observation"]["chat_history"]
)
```
**Test:** Reset with adversarial=True, take an action, verify returned observation has adversarial chat.

### BUG-2: client.py never sends evidence ✅
**File:** `client.py`
**Problem:** `step()` sends `{"action": action, "reasoning": reasoning}` — no `evidence` field. This always triggers the -0.1 evidence penalty in the environment.
**Fix:** Change `step()` signature to accept `evidence: int = 0`. Send `{"action": action, "evidence": evidence}` in the request body. Drop the `reasoning` parameter (unused by server).
**Before:** `def step(self, episode_id: str, action: str, reasoning: str = "") -> dict:`
**After:** `def step(self, episode_id: str, action: str, evidence: int = 0) -> dict:`

### BUG-3: inference.py main path never sends evidence ✅
**File:** `inference.py`
**Problem:** `call_llm()` returns `{"action", "reasoning"}` with no `evidence` key. `run_task()` passes this directly as `action_dict`. The environment always applies the -0.1 penalty.
**Fix:**
1. Update `SYSTEM_PROMPT` to request evidence (log index), matching `GROQ_SYSTEM_PROMPT` format.
2. Update `call_llm()` to parse `evidence` from LLM response and include it in the returned dict.
3. Update `run_task()` to include `evidence` in `action_dict`.
4. Update `deterministic_fallback()` to also return an `evidence` key (index of the most relevant log line).

### BUG-4: README reward values conflate compute_reward and grade ✅
**File:** `README.md`
**Problem:** The "Reward Function" section describes `compute_reward()` values (1.0/0.7/0.4 efficiency) but presents them as if they determine final scores. `grade()` uses a completely different formula (0.999/0.849/0.699). Readers can't tell which scoring system produces the numbers in the benchmark table.
**Fix:** Split into two clearly labeled sections:
1. **"Per-Step Reward"** — describes `compute_reward()` with exact formula: `(0.5 * safety) + (0.5 * efficiency)`
2. **"Final Episode Score"** — describes `grade()` with exact formula: `0.999 if step 1, else max(0.5, 0.999 - 0.15 * (step - 1))`, dangerous = 0.001, unresolved = 0.15
3. Add explicit note: "All scores reported in benchmark tables use `grade()`, not `compute_reward()`."
4. Add note: "The evidence penalty affects per-step feedback only and does NOT change the final episode grade."

---

## 🟡 Medium

### BUG-5: train.py overwrites Pipeline A plots ✅
**File:** `train.py`
**Problem:** Saves to `reward_curve.png`, `loss_curve.png`, `before_after.png` — the exact same filenames that `regenerate_plots.py` outputs from Pipeline A data. Running `train.py` silently overwrites Pipeline A plots with Pipeline B data.
**Fix:** Change `train.py` to save plots to a `pipeline_b/` subdirectory:
- `pipeline_b/reward_curve.png`
- `pipeline_b/loss_curve.png`
- `pipeline_b/before_after.png`
Create the directory with `os.makedirs("pipeline_b", exist_ok=True)`. Update all `plt.savefig()` calls and print statements.

### BUG-6: benchmark_results.json labels don't match README ✅
**File:** `README.md`
**Problem:** `benchmark_results.json` has specific labels like `"naive(scale_infrastructure)"`, `"naive(flush_redis_cache)"`, `"naive(rollback_deployment)"`. README table just says `"naive (keyword)"` for all, losing information about which action was picked.
**Fix:** Update the README cross-validation table to show which action the naive baseline picked, matching the JSON. Example: `naive (→scale_infrastructure)` or simply use the JSON labels directly.

### BUG-7: before_after.png baseline conflates models ✅
**File:** `README.md`
**Problem:** The Model Evaluation table shows `0.201` as "Untrained Baseline" for easy adversarial. This number comes from `training_log.json` (Pipeline B, `llama-3.1-8b-instant`). The "After Training" column shows Qwen post-GRPO numbers. The two columns compare different models without making this obvious (the footnote helps but is easy to miss).
**Fix:** Restructure the table:
- Rename "Untrained Baseline" → "Pre-training baseline (llama-3.1-8b via Groq harness)"
- Rename "After Training" → "Post-GRPO (Qwen 2.5-0.5B + LoRA)"
- Or split into two separate tables — one for Pipeline A (Qwen before/after), one for Pipeline B (Groq harness).
- Apply the same fix to `WRITEUP.md` table.

---

## 🟢 Cleanup

### CLEAN-1: models.py is dead code ✅
**File:** `models.py`
**Problem:** Defines `IncidentAction`, `IncidentObservation`, `IncidentState`, `LogEntry`, `ChatMessage`, `RewardBreakdown` — none are used anywhere in the codebase. `server/environment.py` imports them but never instantiates them. Everything uses raw dicts.
**Fix:**
1. Delete `models.py`
2. Remove `from models import IncidentAction, IncidentObservation, IncidentState` from `server/environment.py`
3. Grep entire codebase to confirm no other file imports from `models.py`

### CLEAN-2: Hardcoded Windows paths ✅
**File:** `upload_trainer_state.py`
**Problem:** `SEARCH_DIRS` includes `C:\Users\shikh\Downloads\...` — personal local paths committed to the repo.
**Fix:** Replace with argparse `--search-dir` flag (repeatable). Keep repo root as the only default search location. Example:
```python
ap = argparse.ArgumentParser()
ap.add_argument("--search-dir", action="append", default=[],
                help="Additional directories to search for files")
args = ap.parse_args()
search_dirs = [os.path.dirname(os.path.abspath(__file__))] + args.search_dir
```

### CLEAN-3: Evidence penalty invisible to grade() ✅
**File:** `server/environment.py`, `README.md`
**Problem:** Evidence penalty applies in `step()` to per-step reward, but `grade()` computes its score independently from `resolved`, `step_count`, and `dangerous_actions`. It never reads `cumulative_reward`. The penalty is effectively invisible to all reported scores.
**Decision:** Do NOT change `grade()` behavior. Document explicitly in README:
> "Note: The evidence penalty (-0.1) affects per-step feedback returned by `step()` but does not change the final episode score returned by `grade()`. All benchmark and evaluation scores use `grade()`."

### CLEAN-4: WRITEUP.md before-training number ✅
**File:** `WRITEUP.md`
**Problem:** Shows `0.2006` as "Before Training" for easy adversarial — this is Pipeline B (`llama-3.1-8b-instant`), not the Qwen baseline (which was 0.769 at step 1 per `trainer_state.json`).
**Fix:** Add a note below the table clarifying the source model: "Before Training values reflect the Pipeline B evaluation harness (llama-3.1-8b-instant via Groq API), not the Qwen model's initial performance."

### CLEAN-5: full_dump.txt committed to repo ✅
**File:** `full_dump.txt`
**Problem:** The debug dump file was created inside the repo and may be committed.
**Fix:** Delete it. Add `full_dump.txt` to `.gitignore`.

### CLEAN-6: Duplicate content in dump (self-referential) ✅
**File:** N/A (artifact of the dump process)
**Problem:** The full dump included itself recursively, doubling the file. Not a code bug.
**Fix:** No code fix needed. Just delete `full_dump.txt` from the repo.

---

## 🆕 New Issues (found post-v3 audit)

### NEW-4: client.reset() missing adversarial parameter ✅
**File:** `client.py`
**Problem:** reset() only sent task_id, never adversarial. HTTP-mode agents could never start adversarial episodes.
**Fix:** Added adversarial: bool = False parameter, forwarded in JSON body.

### NEW-5: inference.py run_task() never ran adversarial mode ✅
**File:** `inference.py`
**Problem:** env.reset() called without adversarial argument, always standard mode.
**Fix:** Added ADVERSARIAL env var config, passed to reset(), logged in [START] line.

### NEW-6: SYSTEM_PROMPT dead duplicate of GROQ_SYSTEM_PROMPT ✅
**File:** `inference.py`
**Problem:** Two byte-for-byte identical constants. Maintenance hazard.
**Fix:** Deleted SYSTEM_PROMPT, call_llm() now uses GROQ_SYSTEM_PROMPT.

### NEW-7: has_deploy_success and prohibit_scale dead variables ✅
**File:** `inference.py`
**Problem:** Computed but never referenced in the decision tree.
**Fix:** Deleted both lines.

### NEW-8: train.py dropped fb["evidence"] in fallback paths ✅
**File:** `train.py`
**Problem:** Both fallback branches (except + else) in run_episode() hardcoded evidence=0 instead of using fb.get("evidence", 0). Correct fallback evidence was silently discarded, always incurring the -0.1 penalty.
**Fix:** Replaced `action, evidence = fb["action"], 0` with `action, evidence = fb["action"], fb.get("evidence", 0)` in both branches.

### NEW-9: README inference.py standard-mode claim stale ✅
**File:** `README.md`
**Problem:** README said inference.py "runs the standard chat overlay" after ADVERSARIAL env var was added in NEW-5.
**Fix:** Updated README to document ADVERSARIAL=true usage and added it to the Run Inference code block.

### NEW-10: numpy missing from requirements.txt and pyproject.toml ✅
**Files:** `requirements.txt`, `pyproject.toml`
**Problem:** regenerate_plots.py imports numpy but it was not listed as an explicit dependency.
**Fix:** Added numpy>=1.26.0 to both files.

### NEW-11: Unused Optional import in client.py ✅
**File:** `client.py`
**Problem:** from typing import Optional was imported but never used.
**Fix:** Removed the import.

### NEW-12: CLAUDE.md missing ADVERSARIAL command ✅
**File:** `CLAUDE.md`
**Problem:** Commands section did not document ADVERSARIAL=true python inference.py.
**Fix:** Added the adversarial mode command line to the Commands section.

### NEW-13: Broken regex in evaluate_by_difficulty.py ✅
**File:** `scripts/evaluate_by_difficulty.py`
**Problem:** ev_match regex uses double-escaped backslashes (r"evidence\\s*...") which match literal backslashes, never "evidence: 5". ev_match is always None; evidence always defaults to 0.
**Fix:** Change to r"evidence\s*[:=]\s*(-?\d+)".

### NEW-14: evaluate_by_difficulty.py fallback paths hardcode evidence=0 ✅
**File:** `scripts/evaluate_by_difficulty.py`
**Problem:** All four finish(fb["action"], 0) calls discard fb["evidence"]. Same class as NEW-8.
**Fix:** Replace all four with finish(fb["action"], fb.get("evidence", 0)).

### NEW-15: BUGS.md NEW-13 and NEW-14 status markers not updated ✅
**File:** `BUGS.md`
**Problem:** NEW-13 and NEW-14 were fixed in code but their BUGS.md entries still showed ⬜.
**Fix:** Changed both markers to ✅.

### NEW-16: benchmark.py collect() missing all type hints ✅
**File:** `benchmark.py`
**Problem:** collect() had no type annotations on any parameter or return type.
**Fix:** Added full signature: collect(env: IncidentResponseEnvironment, task_id: str, adversarial: bool, action: str) -> list[float].

### NEW-17: inference.py get_env() and main() missing return type annotations ✅
**File:** `inference.py`
**Problem:** get_env() and main() had no -> return type annotations.
**Fix:** Added -> tuple to get_env() and -> int to main().

### NEW-18: server/app.py unused `import os` ✅
**File:** `server/app.py`
**Problem:** import os at top of file; os is never referenced anywhere in server/app.py.
**Fix:** Delete the import line.

### NEW-19: server/app.py main() missing -> None return type ✅
**File:** `server/app.py`
**Problem:** def main(): has no return type annotation. Every other main() in the codebase has -> None.
**Fix:** Change to def main() -> None:.

### NEW-20: dag_demo.html confirmed present in repo (not in dump) ✅
**File:** README.md
**Problem:** dag_demo.html was referenced in README but not captured in full_dump.txt.
**Fix:** Confirmed present in repo. No change needed.

### NEW-21: task_easy acceptable_actions dead branch ✅
**File:** task_definitions.py
**Problem:** acceptable_actions was identical to optimal_actions for task_easy, making the 0.7 safety score branch unreachable in compute_reward().
**Fix:** Changed task_easy acceptable_actions to [].

### NEW-24: environment.py shim step() missing -> dict return annotation ✅
**File:** environment.py
**Problem:** step() had no return type annotation, violating CODING_STANDARDS.
**Fix:** Added -> dict to the method signature.

### NEW-25: procedural_generator.py missing all type annotations ✅
**File:** procedural_generator.py
**Problem:** All functions lacked parameter and return type annotations.
**Fix:** Added full type annotations to all function signatures.

### NEW-27: upload_pngs.py module-level side-effects ✅
**File:** upload_pngs.py
**Problem:** Token prompt, HfApi instantiation, and upload loop ran at module level.
**Fix:** Moved all execution into main() with if __name__ == "__main__" guard.

### NEW-28: upload_trainer_state.py module-level side-effects ✅
**File:** upload_trainer_state.py
**Problem:** parse_args(), file discovery, confirmation prompt, and upload loop ran at module level.
**Fix:** Moved all execution into main() with if __name__ == "__main__" guard.

### NEW-29: .gitignore missing or incomplete ✅
**File:** .gitignore
**Problem:** .gitignore was absent or missing required entries per CODING_STANDARDS.
**Fix:** Created/updated .gitignore with __pycache__/, *.pyc, .env, full_dump*.txt, pipeline_b/, *.egg-info/, dist/, .DS_Store.

### NEW-31: task_hard acceptable_actions dead branch ✅
**File:** task_definitions.py
**Problem:** task_hard had acceptable_actions == optimal_actions (["rotate_db_credentials"]), making the 0.7 safety score branch unreachable in compute_reward(). Same defect as NEW-21 for task_easy.
**Fix:** Changed task_hard acceptable_actions to [].

### NEW-32: procedural_generator.py _gen_easy and _gen_hard dead acceptable_actions ✅
**File:** procedural_generator.py
**Problem:** _gen_easy returned acceptable_actions: ["rollback_deployment"] == optimal_actions. _gen_hard returned acceptable_actions: ["rotate_db_credentials"] == optimal_actions. Same dead branch as NEW-21/31.
**Fix:** Changed both to acceptable_actions: [].

### NEW-33: .graphify_python hardcoded personal path not gitignored ✅
**File:** .gitignore
**Problem:** .graphify_python contains a hardcoded personal Windows path (C:\Users\shikh\...) and was not listed in .gitignore.
**Fix:** Added .graphify_python to .gitignore.

### NEW-34: inference.py _first_log_index missing Callable type annotation ✅
**File:** inference.py
**Problem:** msg_predicate parameter had no type annotation. Added from typing import Callable.
**Fix:** Annotated as msg_predicate: Callable[[str], bool] and added Callable to typing imports.

### NEW-35: inference.py run_task env parameter missing type annotation ✅
**File:** inference.py
**Problem:** env parameter in run_task() had no type annotation.
**Fix:** Annotated as env: Any, added Any to typing imports.

### NEW-36: server/app.py endpoint functions missing return type annotations ✅
**File:** server/app.py
**Problem:** All six FastAPI endpoint functions had no return type annotations.
**Fix:** Added return type annotations to root, health, get_tasks, reset, step, state, grader.

### NEW-37: procedural_generator.py lambda assignment in _gen_hard (E731) ✅
**File:** procedural_generator.py
**Problem:** base_ts was assigned as a lambda, violating PEP8 E731.
**Fix:** Replaced with a nested def base_ts(h, m, s) -> str.

### NEW-38: task_medium acceptable_actions dead entry ✅
**File:** task_definitions.py
**Problem:** "rollback_deployment" appeared in both optimal_actions and acceptable_actions for task_medium. compute_reward() checks optimal first, so rollback in acceptable_actions was unreachable.
**Fix:** `acceptable_actions` is `["scale_infrastructure"]` only (verified 2026-08-23; rollback_deployment removed from acceptable_actions).

### NEW-39: inference.py _first_log_index unnecessary string-quoted annotation ⬜
**File:** inference.py
**Problem:** msg_predicate: "Callable[[str], bool]" uses a forward-reference string quote even though Callable is already imported at module level. Inconsistent with every other annotation in the file.
**Fix:** Change to msg_predicate: Callable[[str], bool] (remove the quotes).

### NEW-40: inference.py get_env() bare tuple return type ⬜
**File:** inference.py
**Problem:** def get_env() -> tuple: uses unparameterized tuple, providing no information about the two return values.
**Fix:** Change to -> tuple[Any, str].
