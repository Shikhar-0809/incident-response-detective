# Incident Response Detective: Teaching AI to Resist Social Engineering in SRE Operations

## The Hook

We observed a **397% relative improvement** on the authority-bias stress case (`task_easy` adversarial): mean score increased from **0.201** (untrained baseline) to **0.999** after GRPO fine-tuning. In the failure case, the model receives correct logs and runbook guidance but still follows contradictory Slack authority messages. This environment isolates that exact error mode by changing chat only, while keeping logs and runbook constant.

## The Problem

During incidents, operators synthesize three channels: logs, chat, and runbook. The failure we target is straightforward: the agent selects an action endorsed by confident chat participants even when the runbook and causal logs disagree. In production, that means performing unsafe remediations (for example, scaling or cache flushes) instead of the documented fix. We designed the benchmark so only social context is adversarial, which makes authority bias directly measurable rather than anecdotal.

## The Solution

Incident-Response-Detective is an OpenEnv triage environment where each step exposes logs, Slack chat, and a runbook, and the agent must choose one remediation action. Reward combines **safety** (runbook-consistent vs dangerous actions) and **efficiency** (time-to-resolution), with explicit penalties for unsafe choices. Adversarial mode replaces only the chat stream, so performance differences can be attributed to social-pressure susceptibility rather than task content changes.

## Training Approach

After API-based prototyping with `llama-3.1-8b-instant`, we ran final GRPO fine-tuning on Kaggle using **Qwen2.5-0.5B-Instruct + LoRA** for **384 optimizer steps** (TRL 1.2.0). Prototyping was used to validate environment/reward behavior; the Kaggle run produced the trained adapter weights published in the model repo.

## Results

Primary result: on adversarial Easy, mean score improved from **0.2006** to **0.999** (+0.798 absolute, +397% relative). Medium and Hard were already near ceiling in this evaluation setup (0.999 baseline), so the measurable gain is concentrated in the authority-bias condition.

Table from harness before/after eval:

| Task | Before Training | After Training | Improvement |
|------|-----------------|----------------|-------------|
| task_easy (adversarial) | 0.2006 | 0.999 | +0.798 |
| task_medium (adversarial) | 0.999 | 0.999 | — (already strong) |
| task_hard (adversarial) | 0.999 | 0.999 | — (already strong) |

The key behavioral change is not generic accuracy gain; it is policy shift under social contradiction. Before training, the model frequently follows authoritative chat cues. After training, it consistently prioritizes runbook/log evidence in the same scenario family.

## Try It

- **Hugging Face Space:** [Shiggii / incident-response-detective](https://huggingface.co/spaces/Shiggii/incident-response-detective)  
- **Kaggle notebook (GRPO + LoRA):** [notebookb5136cd284](https://www.kaggle.com/code/shikharkumarsanjay/notebookb5136cd284)  
- **Trained adapter:** [Shiggii / qwen-incident-response-grpo](https://huggingface.co/Shiggii/qwen-incident-response-grpo)  

## Future Work

Next step is scaling from three archetypes to larger procedurally generated pools while preserving causal structure. The immediate experiment is to retrain/evaluate on expanded seeds and report out-of-template generalization (same reward logic, unseen entity/timestamp/service permutations).
