"""
Incident-Response-Detective: Inference Script
===============================================
Uses the OpenAI-compatible client to call an LLM that performs Chain-of-Thought
reasoning over incident observations (logs, chat, runbook) and selects actions.

Emits structured [START]/[STEP]/[END] logs per hackathon requirements.

Required env vars:
  API_BASE_URL  — LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (default: gpt-4o-mini)
  HF_TOKEN      — API key for the LLM service

Optional:
  ENV_BASE_URL  — Running environment URL. If unset, uses embedded in-process env.
  TASK_IDS      — Comma-separated subset (default: task_easy,task_medium,task_hard)
  MAX_AGENT_STEPS — Max steps per task (default: 3)
  BENCHMARK_NAME  — Label for [START] line (default: incident-response-detective)
"""

import os
import sys
import json

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "")
TASK_IDS = os.environ.get("TASK_IDS", "task_easy,task_medium,task_hard").split(",")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "3"))
BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "incident-response-detective")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

# ── Ensure project root importable ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Environment Access ────────────────────────────────────────────────────────

def get_env():
    """Return either an HTTP client or an embedded environment.

    Embedded mode uses the root-level environment (step(episode_id, action_dict) signature).
    HTTP mode uses the client which wraps the deployed server.
    """
    if ENV_BASE_URL:
        from client import IncidentResponseClient
        return IncidentResponseClient(base_url=ENV_BASE_URL), "http"
    else:
        from environment import IncidentResponseEnvironment
        return IncidentResponseEnvironment(), "embedded"


# ── LLM Agent ─────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

GROQ_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) performing incident triage.

You will receive an incident observation containing:
1. logs: Raw system error messages with timestamps, levels, and services. Each line is prefixed with its index [N].
2. chat_history: Slack-style messages from on-call engineers. WARNING: engineers may panic and suggest wrong fixes.
3. runbook: Official procedures. Runbook prohibitions MUST be obeyed — they override chat suggestions.

Analyze ALL three sources. Identify the ROOT CAUSE (not downstream symptoms). Select the single best remediation action.

Respond with ONLY this JSON object and absolutely nothing else:
{"action": "<action_name>", "evidence": <log_index>, "reasoning": "<one sentence>"}

Where:
- action is exactly one of: rollback_deployment, scale_infrastructure, flush_redis_cache, notify_cto, restart_api_gateway, rotate_db_credentials, enable_circuit_breaker, purge_cdn_cache
- evidence is the integer index [N] of the single most diagnostic log line
- reasoning is one sentence explaining the root cause and your action choice"""

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) performing incident triage.

You will receive an incident observation containing:
1. logs: Raw system error messages with timestamps, levels, and services. Each line is prefixed with its index [N].
2. chat_history: Slack-style messages from on-call engineers. WARNING: engineers may panic and suggest wrong fixes.
3. runbook: Official procedures. Runbook prohibitions MUST be obeyed — they override chat suggestions.

Analyze ALL three sources. Identify the ROOT CAUSE (not downstream symptoms). Select the single best remediation action.

Respond with ONLY this JSON object and absolutely nothing else:
{"action": "<action_name>", "evidence": <log_index>, "reasoning": "<one sentence>"}

Where:
- action is exactly one of: rollback_deployment, scale_infrastructure, flush_redis_cache, notify_cto, restart_api_gateway, rotate_db_credentials, enable_circuit_breaker, purge_cdn_cache
- evidence is the integer index [N] of the single most diagnostic log line
- reasoning is one sentence explaining the root cause and your action choice"""


def build_user_prompt(observation: dict) -> str:
    """Format the observation into a structured prompt for the LLM."""
    logs_str = "\n".join(
        f"  [{i}] [{l['ts']}] [{l['level']}] {l['service']}: {l['msg']}"
        for i, l in enumerate(observation.get("logs", []))
    )
    chat_str = "\n".join(
        f"  [{m['time']}] {m['user']}: {m['msg']}"
        for m in observation.get("chat_history", [])
    )
    runbook_str = observation.get("runbook", "No runbook provided.")

    return f"""== INCIDENT OBSERVATION ==

SYSTEM LOGS (each prefixed with index [N]):
{logs_str}

SLACK CHAT:
{chat_str}

RUNBOOK:
{runbook_str}

== YOUR TASK ==
Analyze the above. Identify the root cause. Select ONE action.
Respond with JSON only: {{"action": "...", "evidence": <log_index>, "reasoning": "one sentence"}}"""


def call_llm(observation: dict) -> dict:
    """Call the LLM via OpenAI-compatible client. Returns {action, evidence, reasoning}."""
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    user_prompt = build_user_prompt(observation)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        try:
            evidence = int(result.get("evidence", 0))
        except (TypeError, ValueError):
            evidence = 0
        return {
            "action": result.get("action", "notify_cto"),
            "evidence": evidence,
            "reasoning": result.get("reasoning", ""),
        }
    except Exception:
        return deterministic_fallback(observation)


def _first_log_index(logs: list, msg_predicate) -> int:
    """Return index of the first log whose msg satisfies msg_predicate, else 0."""
    for i, log in enumerate(logs):
        if msg_predicate(log.get("msg", "")):
            return i
    return 0


def deterministic_fallback(observation: dict) -> dict:
    """Rule-based fallback when LLM is unavailable. Implements CoT heuristics."""
    logs = observation.get("logs", [])
    runbook = observation.get("runbook", "")

    # Detect patterns
    has_credential_rotation = any("credential" in l.get("msg", "").lower() and "rotat" in l.get("msg", "").lower() for l in logs)
    has_propagation_fail = any("propagat" in l.get("msg", "").lower() and "fail" in l.get("msg", "").lower() for l in logs)
    has_sidecar_fail = any("sidecar" in l.get("msg", "").lower() and ("not responding" in l.get("msg", "").lower() or "retry" in l.get("msg", "").lower()) for l in logs)
    has_oom = any("oom" in l.get("msg", "").lower() or "maxmemory" in l.get("msg", "").lower() for l in logs)
    has_crossslot = any("crossslot" in l.get("msg", "").lower() for l in logs)
    has_503 = any("503" in l.get("msg", "") for l in logs)
    has_deploy_success = any("completed successfully" in l.get("msg", "").lower() and "deploy" in l.get("msg", "").lower() for l in logs)

    # Runbook prohibitions
    runbook_lower = runbook.lower()
    prohibit_rollback = "do not" in runbook_lower and "rollback" in runbook_lower
    prohibit_flush = "do not" in runbook_lower and "flush" in runbook_lower
    prohibit_scale = "do not" in runbook_lower and "scale" in runbook_lower

    # Decision tree
    if has_credential_rotation and (has_propagation_fail or has_sidecar_fail):
        evidence = _first_log_index(
            logs,
            lambda m: "propagat" in m.lower() and "fail" in m.lower(),
        )
        if evidence == 0 and not has_propagation_fail:
            evidence = _first_log_index(
                logs,
                lambda m: "credential" in m.lower() and "rotat" in m.lower(),
            )
        return {
            "action": "rotate_db_credentials",
            "evidence": evidence,
            "reasoning": "Credential rotation + propagation failure detected. Root cause is stale credentials. Runbook prescribes rotate_db_credentials.",
        }

    if has_oom and has_crossslot and prohibit_flush:
        return {
            "action": "rollback_deployment",
            "evidence": _first_log_index(logs, lambda m: "crossslot" in m.lower()),
            "reasoning": "Cache OOM with CROSSSLOT errors. Runbook prohibits flushing during peak. Root cause is a bad deploy introducing hash-slot bug. Rollback is safe.",
        }

    if has_503 and not prohibit_rollback:
        return {
            "action": "rollback_deployment",
            "evidence": _first_log_index(logs, lambda m: "503" in m),
            "reasoning": "503 upstream errors detected. Runbook allows rollback for recent deployments.",
        }

    if has_503 and prohibit_rollback and not prohibit_flush:
        evidence = _first_log_index(
            logs,
            lambda m: "password authentication failed" in m.lower(),
        )
        if evidence == 0:
            evidence = _first_log_index(
                logs,
                lambda m: "propagat" in m.lower() and "fail" in m.lower(),
            )
        return {
            "action": "rotate_db_credentials",
            "evidence": evidence,
            "reasoning": "503s present but rollback prohibited. Likely credential issue.",
        }

    return {
        "action": "notify_cto",
        "evidence": _first_log_index(logs, lambda m: "blackout" in m.lower() or "503" in m),
        "reasoning": "Unable to determine root cause with confidence. Escalating.",
    }


# ── Groq Agent ────────────────────────────────────────────────────────────────

def build_groq_prompt(observation: dict) -> str:
    """Format observation with indexed log lines so the model can cite evidence by index."""
    logs_str = "\n".join(
        f"  [{i}] [{l['ts']}] [{l['level']}] {l['service']}: {l['msg']}"
        for i, l in enumerate(observation.get("logs", []))
    )
    chat_str = "\n".join(
        f"  [{m['time']}] {m['user']}: {m['msg']}"
        for m in observation.get("chat_history", [])
    )
    return (
        "== INCIDENT OBSERVATION ==\n\n"
        f"SYSTEM LOGS (each prefixed with index [N]):\n{logs_str}\n\n"
        f"SLACK CHAT:\n{chat_str}\n\n"
        f"RUNBOOK:\n{observation.get('runbook', 'No runbook provided.')}\n\n"
        'Respond with JSON only: {"action": "...", "evidence": <log_index>, "reasoning": "one sentence"}'
    )


def run_groq_agent(task_id: str, adversarial: bool = False) -> float:
    """Call Groq API for one episode. Returns grade score (0.0–1.0).

    Falls back to deterministic_fallback if GROQ_API_KEY is not set or the
    API call fails.
    """
    import requests as _requests
    from environment import IncidentResponseEnvironment

    groq_key = os.environ.get("GROQ_API_KEY", "")

    env = IncidentResponseEnvironment()
    episode_id, obs = env.reset(task_id=task_id, adversarial=adversarial)
    log_count = len(obs.get("logs", []))

    action, evidence = "notify_cto", 0

    if groq_key:
        try:
            resp = _requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user",   "content": build_groq_prompt(obs)},
                    ],
                    "temperature": 0,
                    "max_tokens": 256,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences if the model wraps its output
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            action = parsed.get("action", "notify_cto")
            try:
                evidence = int(parsed.get("evidence", 0))
            except (TypeError, ValueError):
                evidence = 0

        except Exception:
            fb = deterministic_fallback(obs)
            action, evidence = fb["action"], fb.get("evidence", 0)
    else:
        fb = deterministic_fallback(obs)
        action, evidence = fb["action"], fb.get("evidence", 0)

    # Clamp to valid log index range so we don't waste the evidence penalty
    if log_count:
        evidence = max(0, min(evidence, log_count - 1))

    env.step(episode_id, {"action": action, "evidence": evidence})
    return env.grade(episode_id)["score"]


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_task(env, env_mode: str, task_id: str) -> dict:
    """Run a single task. Returns {success, steps, score, rewards}."""

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")

    episode_id, observation = env.reset(task_id=task_id)

    rewards = []
    last_score = 0.0

    for step_num in range(1, MAX_AGENT_STEPS + 1):
        if observation.get("done", False):
            break

        # Call LLM (or fallback)
        if HF_TOKEN:
            agent_result = call_llm(observation)
        else:
            agent_result = deterministic_fallback(observation)

        action_str = agent_result["action"]
        evidence = int(agent_result.get("evidence", 0))
        log_count = len(observation.get("logs", []))
        evidence = max(0, min(evidence, log_count - 1)) if log_count > 0 else 0
        action_dict = {"action": action_str, "evidence": evidence}

        if env_mode == "http":
            observation = env.step(episode_id, action_str, evidence)
        else:
            observation = env.step(episode_id, action_dict)

        reward = observation.get("last_reward", 0.0)
        done = observation.get("done", False)
        score = observation.get("score", 0.0)
        error = observation.get("last_action_error", None)
        rewards.append(reward)
        last_score = score

        action_json = json.dumps(action_dict)
        print(f"[STEP] step={step_num} action={action_json} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}")

        if done:
            break

    grade_result = env.grade(episode_id)
    final_score = grade_result.get("score", last_score)
    success = final_score >= SUCCESS_SCORE_THRESHOLD
    total_steps = len(rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={total_steps} score={final_score:.3f} rewards={rewards_str}")

    return {
        "task_id": task_id,
        "success": success,
        "steps": total_steps,
        "score": final_score,
        "rewards": rewards,
    }


def main():
    env, env_mode = get_env()

    results = []
    for task_id in TASK_IDS:
        task_id = task_id.strip()
        if not task_id:
            continue
        result = run_task(env, env_mode, task_id)
        results.append(result)

    total_score = sum(r["score"] for r in results) / max(len(results), 1)
    all_success = all(r["success"] for r in results)
    print(f"\n# Average score: {total_score:.3f} | All passed: {all_success}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
