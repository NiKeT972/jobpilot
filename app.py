"""JobPilot API (FastAPI on Vercel). UI is served from public/."""
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from jobpilot import graph
from jobpilot.files import MAX_BYTES, UnsupportedFile, extract_resume_text, package_docx
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


@app.post("/api/resume-file")
async def resume_file(file: UploadFile = File(...)):
    """Upload a resume (PDF / DOCX / TXT) and get back plain text, one bullet per line, for the resume box."""
    data = await file.read(MAX_BYTES + 1)          # read at most 1 byte past the limit, never the whole upload
    try:
        text = extract_resume_text(file.filename or "", data)
    except UnsupportedFile as e:
        raise HTTPException(422, str(e))
    except Exception:                                # corrupt or password-protected files
        raise HTTPException(422, "Couldn't read this file. Try another format, or paste the text instead.")
    return {"filename": file.filename, "text": text[:MAX_TEXT]}


class Package(BaseModel):
    """The approved package from /api/resume. It's the user's own content, so it's validated but not signed."""
    title: str = Field(max_length=200)
    company: str = Field("", max_length=200)
    fit: int = 0
    ats_before: int = 0
    ats_after: int = 0
    resume: list[str] = Field(default_factory=list, max_length=80)
    cover_note: str = Field("", max_length=5_000)
    interview_questions: list[str] = Field(default_factory=list, max_length=20)


@app.post("/api/package.docx")
def package_download(pkg: Package):
    # "AI Engineer (GenAI)" + "Razorpay" -> "AI-Engineer-GenAI-Razorpay.docx": only letters/digits, single dashes
    safe = re.sub(r"[^A-Za-z0-9]+", "-", f"{pkg.title} {pkg.company}").strip("-")[:60] or "application"
    return Response(package_docx(pkg.model_dump()),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.docx"'})


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")
