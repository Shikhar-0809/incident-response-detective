# PROMPTS.md — Cursor Prompt Sequence

Paste these prompts into Cursor one at a time, in order. Wait for each fix to complete before moving to the next. After each fix, verify the change works before proceeding.

---

## Phase 1: Critical Bugs

### Prompt 1 — BUG-1: Fix adversarial chat in step()
```
Read BUGS.md BUG-1. Fix server/environment.py so that step() returns the correct chat_history based on ep["adversarial"]. The adversarial flag is already stored in the episode dict during reset(). Both the normal step path and the invalid-action return path must check ep["adversarial"] and return ADVERSARIAL_OVERLAYS[task_id] when True. Import ADVERSARIAL_OVERLAYS at the top if not already imported. Don't change any other behavior.
```

**Verify:** Add this temporary test at bottom of server/environment.py and run it:
```python
if __name__ == "__main__":
    env = IncidentResponseEnvironment()
    eid, obs = env.reset(task_id="task_easy", adversarial=True)
    adv_chat = obs["chat_history"]
    obs2 = env.step({"action": "notify_cto", "evidence": 0}, episode_id=eid)
    assert obs2["chat_history"] == adv_chat, "BUG-1 NOT FIXED: step() returned standard chat"
    print("BUG-1 VERIFIED: adversarial chat preserved in step()")
```

### Prompt 2 — BUG-2: Fix client.py evidence
```
Read BUGS.md BUG-2. Fix client.py: change step() to accept evidence (int, default 0) instead of reasoning (str). Send {"action": action, "evidence": evidence} in the request JSON body. Remove the reasoning parameter entirely.
```

### Prompt 3 — BUG-3: Fix inference.py evidence
```
Read BUGS.md BUG-3. Fix inference.py so the main inference path sends evidence:

1. Update SYSTEM_PROMPT to request evidence (integer log index), matching GROQ_SYSTEM_PROMPT's format: {"action": "...", "evidence": <log_index>, "reasoning": "..."}
2. Update call_llm() to parse "evidence" from the LLM response and return it in the dict.
3. Update deterministic_fallback() to return an "evidence" key — pick the index of the most relevant log line for each detection pattern (e.g., the credential rotation log for rotate_db_credentials).
4. Update run_task() to include evidence in action_dict when calling env.step().
5. Keep backward compat: if evidence is missing from LLM response, default to 0.
```

### Prompt 4 — BUG-4: Fix README scoring sections
```
Read BUGS.md BUG-4. Fix README.md: split the current "Reward Function" section into two clearly labeled subsections:

1. "### Per-Step Reward (compute_reward)" — show the safety/efficiency tables and the formula (0.5*safety + 0.5*efficiency). Note this is used for step() feedback during training.

2. "### Final Episode Score (grade)" — show the exact formula from code:
   - Resolved step 1: 0.999
   - Resolved step 2+: max(0.5, 0.999 - 0.15 * (step_count - 1))
   - Dangerous action: 0.001
   - Unresolved, no dangerous: 0.15
   Note: "All scores in the benchmark table below use grade(), not compute_reward()."

3. Add under evidence penalty: "The evidence penalty affects per-step feedback only and does NOT change the final episode grade."
```

---

## Phase 2: Medium Bugs

### Prompt 5 — BUG-5: train.py plot paths
```
Read BUGS.md BUG-5. Fix train.py: change all three plt.savefig() calls to save to a pipeline_b/ subdirectory instead of repo root. Add os.makedirs("pipeline_b", exist_ok=True) before saving. Update the print statements to show the new paths. Files should be:
- pipeline_b/reward_curve.png
- pipeline_b/loss_curve.png
- pipeline_b/before_after.png
```

### Prompt 6 — BUG-6: README naive labels
```
Read BUGS.md BUG-6. Fix README.md: update the Cross-Validation Results table so naive baseline rows show which action was picked, matching benchmark_results.json. Change "naive (keyword)" to the specific action from the JSON:
- task_easy standard: naive (rollback_deployment)
- task_easy adversarial: naive (scale_infrastructure)
- task_medium standard: naive (rollback_deployment)
- task_medium adversarial: naive (flush_redis_cache)
- task_hard standard: naive (rollback_deployment)
- task_hard adversarial: naive (rollback_deployment)
```

### Prompt 7 — BUG-7: Fix evaluation table model labels
```
Read BUGS.md BUG-7. Fix README.md Model Evaluation section: the table currently shows "Untrained Baseline" (0.201) next to "After Training" (0.999). These are from different models.

Rename columns:
- "Untrained Baseline" → "Pipeline B baseline (llama-3.1-8b)"
- "After Training" → "Post-GRPO (Qwen 0.5B + LoRA)"

Also fix WRITEUP.md: add a note below its results table clarifying: "Before Training values reflect the Pipeline B evaluation harness (llama-3.1-8b-instant via Groq API), not the Qwen model's initial performance."
```

---

## Phase 3: Cleanup

### Prompt 8 — CLEAN-1: Delete dead code
```
Read BUGS.md CLEAN-1. Delete models.py entirely. Remove the import line "from models import IncidentAction, IncidentObservation, IncidentState" from server/environment.py. Search the entire codebase for any other imports from models and remove them. These dataclasses are never instantiated anywhere.
```

### Prompt 9 — CLEAN-2: Fix hardcoded paths
```
Read BUGS.md CLEAN-2. Fix upload_trainer_state.py: replace the hardcoded SEARCH_DIRS list (which contains C:\Users\shikh\Downloads paths) with argparse. Add a --search-dir flag that can be repeated. Default search location is repo root only. Example usage: python upload_trainer_state.py --search-dir ~/Downloads --search-dir ~/kaggle_output
```

### Prompt 10 — CLEAN-3: Document evidence penalty scope
```
Read BUGS.md CLEAN-3. Add a clear note to README.md in the reward/scoring section:

"**Note on evidence penalty scope:** The -0.1 evidence penalty is applied within `step()` to per-step reward feedback. It does NOT affect the final episode score returned by `grade()`, which is computed solely from whether the incident was resolved, how many steps it took, and whether any dangerous actions were taken. All benchmark and evaluation scores in this document use `grade()`."

Do NOT change the grade() function behavior.
```

### Prompt 11 — CLEAN-4 & CLEAN-5: Final cleanup
```
1. Delete full_dump.txt from the repo if it exists.
2. Add these to .gitignore if not already present: full_dump.txt, pipeline_b/, .env
3. Verify .gitignore already covers __pycache__, *.pyc
```

---

## Phase 4: Verification

### Prompt 12 — Final check
```
Run these commands and report any errors:
1. python train.py --dry-run
2. python inference.py
3. python regenerate_plots.py
4. openenv validate

Then do a final consistency check: read README.md and verify every number traces to a specific function (grade() or compute_reward()) and every plot traces to a specific script (regenerate_plots.py or train.py). Flag any remaining inconsistencies.
```
