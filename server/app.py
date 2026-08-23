"""FastAPI server for Incident-Response-Detective OpenEnv environment."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
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
    adversarial: bool = False

class StepRequest(BaseModel):
    episode_id: str
    action: dict

class GradeRequest(BaseModel):
    episode_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "environment": "Incident-Response-Detective"}


@app.get("/tasks")
def get_tasks() -> dict:
    return {"tasks": env.get_tasks()}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    try:
        episode_id, observation = env.reset(task_id=req.task_id, adversarial=req.adversarial)
        return {"episode_id": episode_id, "observation": observation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> dict:
    try:
        observation = env.step(req.action, episode_id=req.episode_id)
        return {"observation": observation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(episode_id: str) -> dict:
    try:
        return env.get_state(episode_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grader")
def grader(req: GradeRequest) -> dict:
    try:
        return env.grade(req.episode_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
