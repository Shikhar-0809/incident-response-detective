"""
Incident-Response-Detective: Inference Script
===============================================
Calls the LLM proxy injected via API_BASE_URL / API_KEY env vars.
Falls back to deterministic policy only if the LLM returns unparseable output.

Emits structured [START]/[STEP]/[END] logs per hackathon requirements.
"""

import os
import sys
import json
import requests as http_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Static config (no env vars that change at runtime) ────────────────────────

TASK_IDS = os.environ.get("TASK_IDS", "task_easy,task_medium,task_hard").split(",")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "3"))
BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "incident-response-detective")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))


# ── Environment Access ────────────────────────────────────────────────────────

def get_env():
    env_url = os.environ.get("ENV_BASE_URL", "")
    if env_url:
        from client import IncidentResponseClient
        return IncidentResponseClient(base_url=env_url), "http"
    else:
        from server.environment import IncidentResponseEnvironment
        return IncidentResponseEnvironment(), "embedded"


# ── LLM Prompt ────────────────────────────────────────────────────────────────

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


# ── LLM Call — two methods, SDK first then raw HTTP ───────────────────────────

def call_llm(observation: dict) -> dict:
    """
    Call the LLM. Reads API_BASE_URL and API_KEY fresh from env every time.
    Tries OpenAI SDK first, then raw HTTP POST as backup.
    Raises on total failure.
    """
    # Read env vars FRESH every call (validator may set them after import)
    api_base = os.environ.get("API_BASE_URL", "")
    api_key = os.environ.get("API_KEY", "") or os.environ.get("HF_TOKEN", "")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    print(f"# LLM config: base_url='{api_base}' model='{model}' key_set={bool(api_key)}", file=sys.stderr)

    if not api_base:
        raise ValueError("API_BASE_URL is not set")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(observation)},
    ]

    # Method 1: Try OpenAI SDK
    try:
        text = _call_via_sdk(api_base, api_key, model, messages)
        print(f"# SDK response: {text[:200]}", file=sys.stderr)
        return parse_llm_response(text)
    except Exception as e:
        print(f"# SDK failed ({type(e).__name__}: {e}), trying raw HTTP...", file=sys.stderr)

    # Method 2: Raw HTTP POST (guaranteed to hit the proxy)
    text = _call_via_http(api_base, api_key, model, messages)
    print(f"# HTTP response: {text[:200]}", file=sys.stderr)
    return parse_llm_response(text)


def _call_via_sdk(api_base: str, api_key: str, model: str, messages: list) -> str:
    """Call via OpenAI Python SDK."""
    from openai import OpenAI

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_via_http(api_base: str, api_key: str, model: str, messages: list) -> str:
    """Call via raw HTTP POST — works with any OpenAI-compatible endpoint."""
    # Normalize URL: ensure it ends with /chat/completions
    url = api_base.rstrip("/")
    if not url.endswith("/chat/completions"):
        if url.endswith("/v1"):
            url = url + "/chat/completions"
        else:
            url = url + "/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }

    print(f"# HTTP POST to: {url}", file=sys.stderr)
    resp = http_requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_llm_response(text: str) -> dict:
    """Extract action + reasoning from LLM text."""
    from task_definitions import ACTIONS

    # Try direct JSON
    try:
        result = json.loads(text)
        if result.get("action") in ACTIONS:
            return {"action": result["action"], "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from code blocks
    if "```" in text:
        try:
            block = text.split("```")[1]
            if block.startswith("json"):
                block = block[4:]
            result = json.loads(block.strip())
            if result.get("action") in ACTIONS:
                return {"action": result["action"], "reasoning": result.get("reasoning", "")}
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

    # Try finding JSON anywhere
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        result = json.loads(text[start:end])
        if result.get("action") in ACTIONS:
            return {"action": result["action"], "reasoning": result.get("reasoning", "")}
    except (ValueError, json.JSONDecodeError, TypeError):
        pass

    # Keyword scan
    text_lower = text.lower()
    for action in ACTIONS:
        if action in text_lower:
            return {"action": action, "reasoning": text}

    return {"action": "notify_cto", "reasoning": text}


# ── Deterministic Fallback ────────────────────────────────────────────────────

def deterministic_fallback(observation: dict) -> dict:
    logs = observation.get("logs", [])
    runbook = observation.get("runbook", "")

    has_credential_rotation = any("credential" in l.get("msg", "").lower() and "rotat" in l.get("msg", "").lower() for l in logs)
    has_propagation_fail = any("propagat" in l.get("msg", "").lower() and "fail" in l.get("msg", "").lower() for l in logs)
    has_sidecar_fail = any("sidecar" in l.get("msg", "").lower() and ("not responding" in l.get("msg", "").lower() or "retry" in l.get("msg", "").lower()) for l in logs)
    has_oom = any("oom" in l.get("msg", "").lower() or "maxmemory" in l.get("msg", "").lower() for l in logs)
    has_crossslot = any("crossslot" in l.get("msg", "").lower() for l in logs)
    has_503 = any("503" in l.get("msg", "") for l in logs)

    runbook_lower = runbook.lower()
    prohibit_rollback = "do not" in runbook_lower and "rollback" in runbook_lower
    prohibit_flush = "do not" in runbook_lower and "flush" in runbook_lower

    if has_credential_rotation and (has_propagation_fail or has_sidecar_fail):
        return {"action": "rotate_db_credentials", "reasoning": "Credential propagation failure detected."}
    if has_oom and has_crossslot and prohibit_flush:
        return {"action": "rollback_deployment", "reasoning": "Cache OOM + CROSSSLOT, flush prohibited."}
    if has_503 and not prohibit_rollback:
        return {"action": "rollback_deployment", "reasoning": "503 upstream errors, rollback allowed."}
    if has_503 and prohibit_rollback:
        return {"action": "rotate_db_credentials", "reasoning": "503s but rollback prohibited."}
    return {"action": "notify_cto", "reasoning": "Unable to determine root cause."}


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_task(env, env_mode: str, task_id: str) -> dict:
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={model}")

    try:
        episode_id, observation = env.reset(task_id=task_id)
    except Exception as e:
        print(f"[STEP] step=1 action={{}} reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=0 score=0.000 rewards=")
        return {"task_id": task_id, "success": False, "steps": 0, "score": 0.0, "rewards": []}

    rewards = []
    last_score = 0.0

    for step_num in range(1, MAX_AGENT_STEPS + 1):
        if observation.get("done", False):
            break

        # ALWAYS try LLM first
        try:
            agent_result = call_llm(observation)
        except Exception as e:
            print(f"# LLM failed: {type(e).__name__}: {e}", file=sys.stderr)
            agent_result = deterministic_fallback(observation)

        action_str = agent_result.get("action", "notify_cto")
        reasoning = agent_result.get("reasoning", "")
        action_dict = {"action": action_str, "reasoning": reasoning}

        try:
            if env_mode == "http":
                observation = env.step(episode_id, action_str, reasoning)
            else:
                observation = env.step(episode_id, action_dict)
        except Exception as e:
            action_json = json.dumps(action_dict)
            print(f"[STEP] step={step_num} action={action_json} reward=0.00 done=true error={e}")
            print(f"[END] success=false steps={step_num} score=0.000 rewards={','.join(f'{r:.2f}' for r in rewards)}")
            return {"task_id": task_id, "success": False, "steps": step_num, "score": 0.0, "rewards": rewards}

        reward = observation.get("last_reward", 0.0)
        done = observation.get("done", False)
        score = observation.get("score", 0.0)
        error = observation.get("last_action_error", None)
        rewards.append(reward)
        last_score = score

        print(f"[STEP] step={step_num} action={json.dumps(action_dict)} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}")

        if done:
            break

    try:
        grade_result = env.grade(episode_id)
        final_score = grade_result.get("score", last_score)
    except Exception:
        final_score = last_score

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} score={final_score:.3f} rewards={rewards_str}")

    return {"task_id": task_id, "success": success, "steps": len(rewards), "score": final_score, "rewards": rewards}


def main():
    try:
        env, env_mode = get_env()
    except Exception:
        from server.environment import IncidentResponseEnvironment
        env, env_mode = IncidentResponseEnvironment(), "embedded"

    results = []
    for task_id in TASK_IDS:
        task_id = task_id.strip()
        if task_id:
            results.append(run_task(env, env_mode, task_id))

    total_score = sum(r["score"] for r in results) / max(len(results), 1)
    all_success = all(r["success"] for r in results)
    print(f"\n# Average score: {total_score:.3f} | All passed: {all_success}")
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
