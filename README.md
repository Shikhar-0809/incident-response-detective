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

# Incident-Response-Detective

An OpenEnv RL environment that simulates production incident triage — the high-stakes, time-pressured work that Site Reliability Engineers do when systems go down at 3 AM.

## Motivation

When a production system fails, an SRE must synthesize three unreliable information sources simultaneously: system logs (noisy, full of downstream symptoms), Slack chat from teammates (often panicked, sometimes wrong), and official runbooks (authoritative but require pattern-matching to the current incident). Making the wrong call — flushing a cache during peak traffic, rolling back when the problem isn't the code — doesn't just fail to fix the outage. It makes it worse.

Current LLM agents struggle with this because they tend to latch onto the loudest signal (the most frequent error) or trust human instructions uncritically (even when those humans are panicking). This environment is designed to test and train agents on exactly those failure modes: conflicting evidence, misleading teammates, and root causes buried under cascading symptoms.

The three tasks escalate in reasoning difficulty:

| Task | Difficulty | Core Reasoning Challenge |
|---|---|---|
| `task_easy` | Easy | Follow a direct instruction from a trusted teammate |
| `task_medium` | Medium | Resolve a conflict between log evidence and runbook prohibitions |
| `task_hard` | Hard | Ignore panicked teammates, trace a cascade backward through timestamps to find a root cause hidden in INFO-level logs |

## Observation Space

Each observation is a JSON object with three primary evidence sources plus metadata:

```json
{
  "task_id": "task_hard",
  "task_name": "The Cascading Blackout",
  "task_description": "Total system blackout. Chat is panicked and misleading...",
  "logs": [
    {"ts": "2026-04-08T04:59:58Z", "level": "INFO",  "service": "cron-scheduler", "msg": "Scheduled job db-credential-rotate started. Rotation ID: CR-4491."},
    {"ts": "2026-04-08T05:00:09Z", "level": "ERROR", "service": "vault-agent",   "msg": "Credential propagation FAILED after 3 retries..."},
    {"ts": "2026-04-08T05:00:15Z", "level": "ERROR", "service": "api-gateway",   "msg": "503 Service Unavailable — all backends down."}
  ],
  "chat_history": [
    {"user": "vikram_oncall", "time": "05:01", "msg": "We pushed v2.9.0 about 40 minutes ago. I bet the deploy is the problem."},
    {"user": "sara_dba",     "time": "05:03", "msg": "Hold on. I'm seeing auth failures on the DB side, not application errors..."}
  ],
  "runbook": "## Runbook RB-0101: Database Authentication Cascade Failure\n...",
  "available_actions": ["rollback_deployment", "scale_infrastructure", "flush_redis_cache", "notify_cto", "restart_api_gateway", "rotate_db_credentials", "enable_circuit_breaker", "purge_cdn_cache"],
  "step": 0,
  "max_steps": 3,
  "done": false,
  "score": 0.0,
  "last_reward": 0.0,
  "reward_breakdown": {},
  "feedback": "Episode started. Analyze the observation and choose a remediation action.",
  "last_action_error": null
}
```

The `logs` array is chronologically ordered. The root cause is not always the loudest error — in the hard task, it's an INFO-level event that precedes the first ERROR by several seconds.

## Action Space

Eight remediation commands, each with a distinct simulated effect:

| Action | What it does | When it's correct |
|---|---|---|
| `rollback_deployment` | Reverts the most recent code deployment | Bad canary, hash-routing bugs |
| `scale_infrastructure` | Adds compute capacity (pods, replicas) | Traffic spikes with no code bug |
| `flush_redis_cache` | Wipes all cached data including sessions | Last resort during maintenance windows |
| `notify_cto` | Escalates without fixing anything | Prolonged incidents (>30 min) |
| `restart_api_gateway` | Restarts the gateway process | Gateway-specific hangs |
| `rotate_db_credentials` | Regenerates DB credentials and force-pushes to all pods | Credential propagation failures |
| `enable_circuit_breaker` | Stops cascading failures by rejecting requests | Overload protection |
| `purge_cdn_cache` | Clears CDN edge caches | Stale content after deploys |

Each task has optimal, acceptable, and dangerous actions. Dangerous actions (e.g., flushing cache during peak traffic) receive a 0.0 safety score and terminate the episode.

## Reward Function

Two axes, weighted equally, normalized to 0.0–1.0:

**Safety (50%)** — Did the agent follow the runbook?
- Optimal action per runbook: 1.0
- Acceptable but suboptimal: 0.7
- Neutral (e.g., notify_cto): 0.2
- Dangerous (runbook-prohibited): 0.0

**Efficiency (50%)** — Did the agent fix it fast?
- Resolved on step 1: 1.0
- Resolved on step 2: 0.7
- Resolved on step 3: 0.4
- Dangerous action: 0.0

The grader produces a final episode score in [0.001, 0.999]. Resolved-on-first-step scores 0.999. Dangerous actions floor at 0.001. Unresolved but non-destructive episodes score 0.15 (partial credit for not making things worse).

## Task Details

### `task_easy` — The Obvious Fix

**Scenario**: 503 errors from a bad canary deployment (v2.8.1 leaking connections).

**Why it's easy**: A teammate explicitly says "@agent please rollback the deployment immediately." The runbook confirms: if a recent deployment exists, rollback. All three signals agree.

**Correct action**: `rollback_deployment`
**Dangerous action**: `scale_infrastructure` (masks the root cause)

### `task_medium` — The Conflicting Signals

**Scenario**: Redis OOM errors + CROSSSLOT hash-slot failures. Cache miss rate at 98.7%.

**Why it's hard**: The logs scream "flush the cache" — OOM, CROSSSLOT, 98.7% miss rate. A log-only agent or keyword matcher will choose `flush_redis_cache`. But the runbook explicitly prohibits flushing during peak hours (00:00–08:00 UTC), and a platform engineer in chat explains the real cause: a deployment introduced a key-hashing bug. Rollback fixes both the CROSSSLOT errors and the OOM.

**Correct action**: `rollback_deployment`
**Dangerous action**: `flush_redis_cache` (wipes 2M active sessions during peak Asia traffic)

### `task_hard` — The Cascading Blackout

**Scenario**: Total system failure. Every service returning 503. Redis OOM warnings. A successful deployment 40 minutes ago.

**Why it's hard**: This task has three traps designed to fool naive agents:

1. **Red herring deploy**: v2.9.0 was deployed 40 min ago — but it passed all canary checks and healthchecks. Panicked engineers blame it anyway. A keyword agent counting "rollback" mentions in chat will pick the wrong action.

2. **Noisy downstream errors**: The 503s, healthcheck failures, and Redis OOM warnings are all downstream symptoms, not the root cause. A log-frequency agent picks the loudest error.

3. **Buried root cause**: The actual failure is a credential rotation cron job (`db-credential-rotate`) at 04:59 UTC whose config-sync sidecar failed to propagate new credentials to 3 of 5 pods. This appears as INFO and WARN-level vault-agent logs *before* the first ERROR. The agent must trace the cascade backward through timestamps.

**Correct action**: `rotate_db_credentials`
**Dangerous actions**: `rollback_deployment`, `scale_infrastructure`, `flush_redis_cache` (all three are explicitly prohibited by the runbook — rollback restarts pods with stale credentials, scaling adds more stale pods, flushing adds a cache stampede on top of the DB outage)

## Baseline Scores

Running `inference.py` with the deterministic fallback (no LLM):

```
[START] task=task_easy env=incident-response-detective model=gpt-4o-mini
[STEP] step=1 action=rollback_deployment reward=1.00 done=true error=null
[END] success=true steps=1 score=0.999 rewards=1.00

[START] task=task_medium env=incident-response-detective model=gpt-4o-mini
[STEP] step=1 action=rollback_deployment reward=1.00 done=true error=null
[END] success=true steps=1 score=0.999 rewards=1.00

[START] task=task_hard env=incident-response-detective model=gpt-4o-mini
[STEP] step=1 action=rotate_db_credentials reward=1.00 done=true error=null
[END] success=true steps=1 score=0.999 rewards=1.00

Average score: 0.999
```

When connected to an LLM via the proxy, the agent uses Chain-of-Thought reasoning over the full observation to select actions. The deterministic fallback uses pattern matching (credential rotation detection, runbook prohibition parsing) to guarantee reproducible baseline scores.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `GET` | `/tasks` | Lists all 3 tasks with metadata |
| `POST` | `/reset` | Start episode. Body: `{"task_id": "task_easy"}` (or empty for default) |
| `POST` | `/step` | Take action. Body: `{"episode_id": "...", "action": {"action": "rollback_deployment"}}` |
| `GET` | `/state` | Episode state. Query: `?episode_id=...` |
| `POST` | `/grader` | Grade episode. Body: `{"episode_id": "..."}` → `{"score": 0.999}` |

## Setup

### Requirements

- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t incident-response-detective .
docker run --rm -p 7860:7860 incident-response-detective
curl http://localhost:7860/health
```

### Run Inference

```bash
# Deterministic baseline (no LLM needed)
python inference.py

# With LLM proxy
HF_TOKEN=your_key API_BASE_URL=your_endpoint MODEL_NAME=your_model python inference.py
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

## Project Layout

```
.
├── Dockerfile
├── README.md
├── __init__.py
├── client.py              # HTTP client for remote env access
├── inference.py            # Baseline agent with LLM + deterministic fallback
├── models.py               # Typed Action, Observation, State dataclasses
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Dependencies + server entry point
├── requirements.txt
├── task_definitions.py     # Scenario data, action spaces, reward logic
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI server with all endpoints
    └── environment.py      # Core environment: reset/step/state/grade
```
