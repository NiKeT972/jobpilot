"""JobPilot API (FastAPI on Vercel). UI is served from public/."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from jobpilot import graph
from jobpilot.llm import llm_available, model_name
from jobpilot.samples import SAMPLES

app = FastAPI(title="JobPilot")
ROOT = Path(__file__).parent


class StartReq(BaseModel):
    jd_text: str = Field(min_length=40)
    resume_text: str = Field(min_length=40)
    company: str = ""
    target_ats: int = 75


class ResumeReq(BaseModel):
    state: dict
    decision: str
    feedback: str = ""


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm_available(), "model": model_name()}


@app.get("/api/meta")
def meta():
    return {"samples": SAMPLES, "graph": graph.mermaid()}


@app.post("/api/run")
def run(req: StartReq):
    return graph.start(req.jd_text, req.resume_text, req.company, req.target_ats)


@app.post("/api/resume")
def resume(req: ResumeReq):
    return graph.resume(req.state, req.decision, req.feedback)


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")
