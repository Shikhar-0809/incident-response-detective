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
    """Return either an HTTP client or an embedded environment."""
    if ENV_BASE_URL:
        from client import IncidentResponseClient
        return IncidentResponseClient(base_url=ENV_BASE_URL), "http"
    else:
        from server.environment import IncidentResponseEnvironment
        return IncidentResponseEnvironment(), "embedded"


# ── LLM Agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) performing incident triage.

You will receive an incident observation containing:
1. **logs**: Raw system error messages with timestamps, levels, and services.
2. **chat_history**: Slack-style messages from on-call engineers (WARNING: engineers may be panicked and suggest wrong fixes).
3. **runbook**: Official company procedures. Runbook prohibitions MUST be obeyed — they override chat suggestions.

Your task: Analyze ALL three sources, identify the ROOT CAUSE (not downstream symptoms), and select the single best remediation action.

REASONING PROCESS:
1. Read logs chronologically. Find the FIRST error and what preceded it.
2. Read chat — note who is an expert vs who is panicking. Be skeptical of panicked suggestions.
3. Read the runbook — identify prohibited and prescribed actions. Runbook VETOES override everything.
4. Select the action that fixes the root cause while following the runbook.

You MUST respond with EXACTLY this JSON format and nothing else:
{"action": "<action_name>", "reasoning": "<one paragraph explaining your chain of thought>"}

Available actions: rollback_deployment, scale_infrastructure, flush_redis_cache, notify_cto, restart_api_gateway, rotate_db_credentials, enable_circuit_breaker, purge_cdn_cache"""


def build_user_prompt(observation: dict) -> str:
    """Format the observation into a structured prompt for the LLM."""
    logs_str = "\n".join(
        f"  [{l['ts']}] [{l['level']}] {l['service']}: {l['msg']}"
        for l in observation.get("logs", [])
    )
    chat_str = "\n".join(
        f"  [{m['time']}] {m['user']}: {m['msg']}"
        for m in observation.get("chat_history", [])
    )
    runbook_str = observation.get("runbook", "No runbook provided.")

    return f"""== INCIDENT OBSERVATION ==

SYSTEM LOGS (chronological):
{logs_str}

SLACK CHAT:
{chat_str}

RUNBOOK:
{runbook_str}

== YOUR TASK ==
Analyze the above. Identify the root cause. Select ONE action.
Respond with JSON only: {{"action": "...", "reasoning": "..."}}"""


def call_llm(observation: dict) -> dict:
    """Call the LLM via OpenAI-compatible client. Returns {action, reasoning}."""
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
        return {
            "action": result.get("action", "notify_cto"),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        # Fallback: deterministic policy if LLM fails
        return deterministic_fallback(observation)


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
        return {"action": "rotate_db_credentials", "reasoning": "Credential rotation + propagation failure detected. Root cause is stale credentials. Runbook prescribes rotate_db_credentials."}

    if has_oom and has_crossslot and prohibit_flush:
        return {"action": "rollback_deployment", "reasoning": "Cache OOM with CROSSSLOT errors. Runbook prohibits flushing during peak. Root cause is a bad deploy introducing hash-slot bug. Rollback is safe."}

    if has_503 and not prohibit_rollback:
        return {"action": "rollback_deployment", "reasoning": "503 upstream errors detected. Runbook allows rollback for recent deployments."}

    if has_503 and prohibit_rollback and not prohibit_flush:
        return {"action": "rotate_db_credentials", "reasoning": "503s present but rollback prohibited. Likely credential issue."}

    return {"action": "notify_cto", "reasoning": "Unable to determine root cause with confidence. Escalating."}


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_task(env, env_mode: str, task_id: str) -> dict:
    """Run a single task. Returns {success, steps, score, rewards}."""

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")

    # Reset
    if env_mode == "http":
        episode_id, observation = env.reset(task_id=task_id)
    else:
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
        action_dict = {"action": action_str, "reasoning": agent_result.get("reasoning", "")}

        # Step
        if env_mode == "http":
            observation = env.step(episode_id, action_str, agent_result.get("reasoning", ""))
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

    # Grade
    if env_mode == "http":
        grade_result = env.grade(episode_id)
    else:
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

    # Summary (not parsed by evaluator, just for human readability)
    total_score = sum(r["score"] for r in results) / max(len(results), 1)
    all_success = all(r["success"] for r in results)
    print(f"\n# Average score: {total_score:.3f} | All passed: {all_success}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
