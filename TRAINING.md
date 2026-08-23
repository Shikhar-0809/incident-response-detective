# Pipeline A — GRPO Training Reproduction

This document records the **exact configuration** for the real weight-updating training run (Pipeline A). The training loop itself is **not** vendored in this repository — it runs in an external Kaggle notebook because it requires multi-hour GPU access (Tesla T4 ×2) and the full TRL/transformers/peft stack.

## Why training is external

Pipeline A fine-tunes Qwen 2.5-0.5B-Instruct with LoRA + GRPO on adversarial incident-triage episodes. That workload needs sustained GPU time and large ML dependencies. This repo ships the **OpenEnv environment**, evaluation harnesses, and training **artifacts** (`data/trainer_state.json`, the Hugging Face adapter). Reproducing weight updates requires running the Kaggle notebook (or porting its cells to your own GPU machine using the config below).

**Kaggle notebook:** [notebookb5136cd284](https://www.kaggle.com/code/shikharkumarsanjay/notebookb5136cd284)

**Trained adapter:** [Shiggii/qwen-incident-response-grpo](https://huggingface.co/Shiggii/qwen-incident-response-grpo)

---

## Base model

| Field | Value |
|-------|-------|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Training framework | TRL 1.2.0 (per HF model card) |
| Method | GRPO + LoRA |

---

## LoRA configuration

Source: [`adapter_config.json`](https://huggingface.co/Shiggii/qwen-incident-response-grpo/blob/main/adapter_config.json) on the published adapter (fetched 2026-08-23).

| Field | Value |
|-------|-------|
| `peft_type` | `LORA` |
| `task_type` | `CAUSAL_LM` |
| `r` (rank) | `16` |
| `lora_alpha` | `32` |
| `lora_dropout` | `0.05` |
| `bias` | `none` |
| `target_modules` | `gate_proj`, `o_proj`, `down_proj`, `q_proj`, `k_proj`, `up_proj`, `v_proj` |
| `base_model_name_or_path` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `init_lora_weights` | `true` |
| `inference_mode` | `true` (in saved adapter) |
| `peft_version` | `0.18.1` |
| `use_dora` | `false` |
| `use_rslora` | `false` |

---

## Trainer settings (from `data/trainer_state.json`)

These values are logged by TRL's `GRPOTrainer` at the end of the Kaggle run. **Do not treat unstated hyperparameters as known** — only fields present in this file are documented here.

| Field | Value |
|-------|-------|
| `global_step` / `max_steps` | `384` |
| `num_train_epochs` | `3.0` |
| `train_batch_size` | `4` |
| `logging_steps` | `1` |
| `save_steps` | `500` |
| `eval_steps` | `500` |
| `num_input_tokens_seen` | `1,850,395` |

**Learning rate:** Per-step values are logged in `log_history[].learning_rate` (warmup from `0.0` at step 1, peak ~`3.7e-6` mid-run, decay to ~`1.0e-10` at step 384). The schedule endpoints and optimizer type are defined in the Kaggle notebook, not in this repo.

**GRPO-specific knobs** (group size, KL coefficient, clip range, etc.): not present in `trainer_state.json`. See the Kaggle notebook for those settings.

---

## Artifacts in this repo

| File | Contents |
|------|----------|
| `data/trainer_state.json` | Full TRL `log_history` (384 steps × ~20 metrics: reward, loss, grad_norm, kl, entropy, …) |
| `regenerate_plots.py` | Regenerates `reward_curve.png`, `loss_curve.png`, `before_after.png` from `trainer_state.json` |
| HF adapter | LoRA weights at `Shiggii/qwen-incident-response-grpo` |

Download `trainer_state.json` if missing:

```bash
python scripts/download_training_data.py
```

---

## Local evaluation of the trained adapter

To compare base Qwen vs LoRA on adversarial episodes (using `grade()` scores):

```bash
pip install torch transformers peft accelerate
python scripts/evaluate_by_difficulty.py --plot
```

Default evaluation uses `temperature=0` for reproducibility. See README Table A.
