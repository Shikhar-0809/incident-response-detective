"""FastAPI server for Incident-Response-Detective OpenEnv environment."""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from server.environment import IncidentResponseEnvironment

app = FastAPI(
    title="Incident-Response-Detective",
    description="OpenEnv RL environment for complex incident triage reasoning",
    version="1.0.0",
)

env = IncidentResponseEnvironment()


# ── Request/Response Models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"

class StepRequest(BaseModel):
    episode_id: str
    action: dict

class GradeRequest(BaseModel):
    episode_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "incident-response-detective"}


@app.get("/tasks")
def get_tasks():
    return {"tasks": env.get_tasks()}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        episode_id, observation = env.reset(task_id=req.task_id)
        return {"episode_id": episode_id, "observation": observation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        observation = env.step(episode_id=req.episode_id, action_dict=req.action)
        return {"observation": observation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(episode_id: str):
    try:
        return env.get_state(episode_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grader")
def grader(req: GradeRequest):
    try:
        return env.grade(req.episode_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
