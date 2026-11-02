"""JobPilot API (FastAPI on Vercel). UI is served from public/."""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from jobpilot import graph
from jobpilot.llm import llm_available, model_name
from jobpilot.samples import SAMPLES
from jobpilot.state_token import InvalidToken, seal, signing_configured, unseal

app = FastAPI(title="JobPilot")
ROOT = Path(__file__).parent

# Size caps keep a public demo from sending huge prompts to the LLM (cost, latency, serverless time limits).
MAX_TEXT = 15_000


class StartReq(BaseModel):
    jd_text: str = Field(min_length=40, max_length=MAX_TEXT)
    resume_text: str = Field(min_length=40, max_length=MAX_TEXT)
    company: str = Field("", max_length=100)
    target_ats: int = Field(75, ge=0, le=100)


class ResumeReq(BaseModel):
    token: str                                   # signed paused state from the previous response
    decision: Literal["approve", "revise"]
    feedback: str = Field("", max_length=1_000)


def _respond(result: dict) -> dict:
    """Add a signed token when the graph is paused; the browser sends only that token back to resume."""
    if result["paused_at"]:
        result["token"] = seal({"state": result["state"], "paused_at": result["paused_at"]})
    return result


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm_available(), "model": model_name(), "state_signing": signing_configured()}


@app.get("/api/meta")
def meta():
    return {"samples": SAMPLES, "graph": graph.mermaid()}


@app.post("/api/run")
def run(req: StartReq):
    return _respond(graph.start(req.jd_text, req.resume_text, req.company, req.target_ats))


@app.post("/api/resume")
def resume(req: ResumeReq):
    try:
        paused = unseal(req.token)
    except InvalidToken as e:
        raise HTTPException(400, str(e))
    if req.decision == "revise" and not req.feedback.strip():
        raise HTTPException(422, "Say what to change in the cover note.")
    return _respond(graph.resume(paused["state"], req.decision, req.feedback))


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")
