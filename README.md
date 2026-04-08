---
title: Incident Response Detective
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
short_description: OpenEnv incident triage environment with conflicting signals.
tags:
  - openenv
---

# Incident-Response-Detective — OpenEnv Environment

An RL environment for training AI agents on **complex, multi-signal incident triage**. The agent must parse raw system logs, Slack-style engineer chatter, and official runbook procedures to select the safest remediation action — even when the evidence sources conflict.

## Why This Environment is Hard

Most incident-response environments present clear signals. This one is designed around **three types of reasoning failures** that naive agents make:

| Difficulty | Trap | What fails |
|---|---|---|
| **Easy** | None — teammate says exactly what to do | Random agents |
| **Medium** | Logs scream "flush cache" but Runbook prohibits it | Log-only agents, keyword matchers |
| **Hard** | Chat is full of panicked engineers suggesting wrong fixes. Root cause is buried in INFO-level vault logs before the first ERROR. | Keyword voting, chat-trusting agents |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Start new episode: `{"task_id": "task_easy"}` |
| `POST` | `/step` | Take action: `{"episode_id": "...", "action": {"action": "rollback_deployment"}}` |
| `GET` | `/state?episode_id=...` | Get episode state |
| `POST` | `/grader` | Grade episode: `{"episode_id": "..."}` |

## Action Space

`rollback_deployment`, `scale_infrastructure`, `flush_redis_cache`, `notify_cto`, `restart_api_gateway`, `rotate_db_credentials`, `enable_circuit_breaker`, `purge_cdn_cache`

## Observation Space

Each observation contains:

- `logs` — Timestamped system log entries with level, service, and message
- `chat_history` — Slack-style messages from on-call engineers
- `runbook` — Official procedure document with prescribed and prohibited actions
- `available_actions` — Valid action strings
- `step`, `max_steps`, `done`, `score`, `last_reward`, `reward_breakdown`, `feedback`

## Reward Function

Two axes, combined 50/50, normalized to 0.0–1.0:

- **Safety**: Did the agent follow the Runbook? Dangerous actions = 0.0, optimal = 1.0
- **Efficiency**: Did the agent fix it fast? Single-step resolution = 1.0

## Tasks

### `task_easy` — The Obvious Fix
503 errors from a bad canary deployment. Both chat and runbook agree: rollback.

### `task_medium` — The Conflicting Signals
Redis OOM + CROSSSLOT errors. Logs suggest flushing the cache. Runbook explicitly prohibits flushing during peak hours. Chat expert explains the real cause is a key-hashing bug from a bad deploy. Correct action: `rollback_deployment`.

### `task_hard` — The Cascading Blackout
Total system failure. A successful deploy 40 minutes ago is a red herring. Panicked engineers push for rollback/scale/flush. The root cause is a credential rotation cron job whose sidecar sync failed — buried in INFO/WARN-level vault-agent logs before the first ERROR. Correct action: `rotate_db_credentials`.

## Quick Start

```bash
# Local
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t incident-response-detective .
docker run --rm -p 7860:7860 incident-response-detective

# Run inference (deterministic fallback, no LLM needed)
python inference.py

# Run inference with LLM
HF_TOKEN=your_key MODEL_NAME=your_model API_BASE_URL=your_endpoint python inference.py
```

## Project Layout

```
.
├── Dockerfile
├── README.md
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── task_definitions.py
└── server/
    ├── __init__.py
    ├── app.py
    └── environment.py
```
