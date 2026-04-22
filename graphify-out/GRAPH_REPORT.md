# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~5,780 words - fits in a single context window. You may not need a graph.

## Summary
- 106 nodes · 168 edges · 9 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 54 edges (avg confidence: 0.62)
- Token cost: 2,800 input · 1,950 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Environment Logic|Core Environment Logic]]
- [[_COMMUNITY_Client Interface Layer|Client Interface Layer]]
- [[_COMMUNITY_Scenario Design and Actions|Scenario Design and Actions]]
- [[_COMMUNITY_Inference and Agent Stack|Inference and Agent Stack]]
- [[_COMMUNITY_FastAPI Server Endpoints|FastAPI Server Endpoints]]
- [[_COMMUNITY_Reward and Task Logic|Reward and Task Logic]]
- [[_COMMUNITY_Reward Function Design|Reward Function Design]]
- [[_COMMUNITY_Server Package Init|Server Package Init]]
- [[_COMMUNITY_Root Package Init|Root Package Init]]

## God Nodes (most connected - your core abstractions)
1. `IncidentResponseEnvironment` - 23 edges
2. `IncidentResponseClient` - 16 edges
3. `Incident Response Detective` - 9 edges
4. `Action Space (8 Remediation Commands)` - 9 edges
5. `run_task()` - 8 edges
6. `IncidentAction` - 8 edges
7. `IncidentObservation` - 8 edges
8. `IncidentState` - 8 edges
9. `task_medium — The Conflicting Signals` - 6 edges
10. `task_hard — The Cascading Blackout` - 6 edges

## Surprising Connections (you probably didn't know these)
- `ResetRequest` --uses--> `IncidentResponseEnvironment`  [INFERRED]
  server\app (2).py → server\environment.py
- `StepRequest` --uses--> `IncidentResponseEnvironment`  [INFERRED]
  server\app (2).py → server\environment.py
- `GradeRequest` --uses--> `IncidentResponseEnvironment`  [INFERRED]
  server\app (2).py → server\environment.py
- `FastAPI server for Incident-Response-Detective OpenEnv environment.` --uses--> `IncidentResponseEnvironment`  [INFERRED]
  server\app (2).py → server\environment.py
- `Incident-Response-Detective: Inference Script =================================` --uses--> `IncidentResponseEnvironment`  [INFERRED]
  inference (2).py → server\environment.py

## Hyperedges (group relationships)
- **Task Difficulty Escalation (Easy to Medium to Hard)** — readme_task_easy, readme_task_medium, readme_task_hard [EXTRACTED 1.00]
- **Reward Function Composed of Safety and Efficiency** — readme_reward_function, readme_safety_score, readme_efficiency_score [EXTRACTED 1.00]
- **FastAPI Server Stack** — readme_server_app, readme_server_environment, readme_task_definitions_py, readme_models_py [INFERRED 0.85]

## Communities

### Community 0 - "Core Environment Logic"
Cohesion: 0.14
Nodes (16): IncidentResponseEnvironment, Core environment logic for Incident-Response-Detective., Grade an episode. Returns score in 0.0-1.0., Start a new episode. Returns (episode_id, observation_dict)., Execute an action. Returns observation dict., OpenEnv-compatible environment for incident triage.     Implements reset(), ste, ChatMessage, IncidentAction (+8 more)

### Community 1 - "Client Interface Layer"
Cohesion: 0.13
Nodes (15): IncidentResponseClient, Client for the Incident-Response-Detective OpenEnv environment., HTTP client for interacting with the Incident-Response-Detective environment., build_user_prompt(), call_llm(), deterministic_fallback(), get_env(), main() (+7 more)

### Community 2 - "Scenario Design and Actions"
Cohesion: 0.15
Nodes (20): Action Space (8 Remediation Commands), Cascading Failure Pattern, Rationale: Conflicting Evidence Design, Credential Rotation Root Cause, Dockerfile, enable_circuit_breaker Action, flush_redis_cache Action, Incident Response Detective (+12 more)

### Community 3 - "Inference and Agent Stack"
Cohesion: 0.16
Nodes (14): Chain-of-Thought Reasoning, client.py (HTTP Client), Deterministic Fallback (Pattern Matching), inference.py (Baseline Agent), LLM Agent (via Proxy), models.py (Typed Dataclasses), server/app.py (FastAPI Server), server/environment.py (Core Environment) (+6 more)

### Community 4 - "FastAPI Server Endpoints"
Cohesion: 0.29
Nodes (11): get_tasks(), grader(), GradeRequest, health(), FastAPI server for Incident-Response-Detective OpenEnv environment., reset(), ResetRequest, state() (+3 more)

### Community 5 - "Reward and Task Logic"
Cohesion: 0.5
Nodes (3): compute_reward(), Task definitions and reward logic for Incident-Response-Detective., Compute reward on two axes: Safety (did it follow the Runbook?) and     Efficie

### Community 6 - "Reward Function Design"
Cohesion: 1.0
Nodes (3): Efficiency Score (Reward Axis), Reward Function (Safety + Efficiency), Safety Score (Reward Axis)

### Community 7 - "Server Package Init"
Cohesion: 1.0
Nodes (1): Incident-Response-Detective OpenEnv Environment.

### Community 8 - "Root Package Init"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **27 isolated node(s):** `Client for the Incident-Response-Detective OpenEnv environment.`, `HTTP client for interacting with the Incident-Response-Detective environment.`, `LogEntry`, `ChatMessage`, `RewardBreakdown` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Server Package Init`** (2 nodes): `__init__.py`, `Incident-Response-Detective OpenEnv Environment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Root Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `IncidentResponseEnvironment` connect `Core Environment Logic` to `Client Interface Layer`, `FastAPI Server Endpoints`?**
  _High betweenness centrality (0.258) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `IncidentResponseEnvironment` (e.g. with `ResetRequest` and `StepRequest`) actually correct?**
  _`IncidentResponseEnvironment` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `IncidentResponseClient` (e.g. with `Incident-Response-Detective: Inference Script =================================` and `Return either an HTTP client or an embedded environment.`) actually correct?**
  _`IncidentResponseClient` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_task()` (e.g. with `.reset()` and `.step()`) actually correct?**
  _`run_task()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Client for the Incident-Response-Detective OpenEnv environment.`, `HTTP client for interacting with the Incident-Response-Detective environment.`, `LogEntry` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Core Environment Logic` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._
- **Should `Client Interface Layer` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._