"""JobPilot: a LangGraph agent that tailors a job application.

Flow:
    parse_jd -> research_company -> index_resume -> score_fit
        -> (fit < 35)                       gap_plan -> END          [early exit: not a good match]
        -> (ATS < target and attempts < 2)  tailor_bullets -> verify_claims -> score_fit   [improvement loop]
        -> otherwise                        draft_cover_note -> human_review [HITL]
                                                -> revise -> draft_cover_note
                                                -> approve -> finalize -> END

Two scores are tracked on purpose:
  * fit  – does the candidate actually have the skills? (synonym-aware, fixed by the resume's real content)
  * ats  – does the resume use the JD's exact words? (what keyword filters see; tailoring can raise this honestly)
verify_claims reverts any rewritten bullet that introduces a skill or number not in the original, so the loop
can never "improve" the score by inventing experience.
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import skills as SK
from .llm import get_llm, llm_failed
from .retriever import ResumeRetriever, split_bullets
from .tools import company_research


class AppState(TypedDict, total=False):
    jd_text: str
    company: str
    resume_text: str
    target_ats: int
    job: dict
    company_info: dict
    bullets: list            # original resume bullets
    evidence: dict           # skill -> list of supporting bullets
    fit: dict
    ats: dict
    tailored: list           # [{original, rewritten, skill, status}]
    verification: list
    attempts: int
    cover_note: str
    cover_version: int
    decision: str
    feedback: str
    plan: dict
    package: dict
    status: str
    trace: list


def _trace(s: AppState, node: str, detail: str, mode: str = "rules") -> list:
    return list(s.get("trace", [])) + [{"node": node, "detail": detail, "mode": mode}]


def current_bullets(s: AppState) -> list[str]:
    """The resume as it stands now: original bullets with accepted rewrites swapped in."""
    swap = {t["original"]: t["rewritten"] for t in s.get("tailored", []) if t.get("status") == "accepted"}
    return [swap.get(b, b) for b in s.get("bullets", [])]


# --------------------------------------------------------------------------- 1. parse JD


class JobSpec(BaseModel):
    title: str
    company: str = ""
    seniority: str = Field("", description="intern / junior / mid / senior")
    must_have: list[str] = Field(description="Required skills or tools, short names, max 10")
    nice_to_have: list[str] = Field(default_factory=list, description="Preferred skills, short names, max 8")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities, max 6, short")
    years_experience: Optional[float] = None


def parse_jd(s: AppState) -> dict:
    jd = s["jd_text"]
    llm = get_llm(0)
    if llm is not None:
        try:
            spec: JobSpec = llm.with_structured_output(JobSpec).invoke(f"Extract a structured job spec from this JD:\n\n{jd}")
            job = spec.model_dump()
            job["company"] = s.get("company") or job["company"]
            return {"job": job, "trace": _trace(s, "parse_jd", f"{job['title']}: {len(job['must_have'])} must-have, "
                                                f"{len(job['nice_to_have'])} nice-to-have skills", "llm")}
        except Exception as e:
            llm_failed("parse_jd", e)

    lines = [l.strip() for l in jd.splitlines() if l.strip()]
    lower = jd.lower()
    # split JD into required vs preferred sections by headings
    pref_idx = min([lower.find(k) for k in ("nice to have", "preferred", "good to have", "bonus", "plus:") if k in lower] or [len(jd)])
    must = list(SK.find_skills(jd[:pref_idx]).keys())
    nice = [k for k in SK.find_skills(jd[pref_idx:]).keys() if k not in must]
    years = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+\s*)?years?", lower)
    resp = [re.sub(r"^[-•*\d.)\s]+", "", l) for l in lines if re.match(r"^[-•*]", l)][:6]
    title = re.sub(r"^(job title|role|position)\s*[:\-]\s*", "", lines[0], flags=re.I) if lines else "Role"
    title = re.split(r"\s+[—–|-]\s+", title)[0]
    job = {"title": title[:80], "company": s.get("company", ""),
           "seniority": next((w for w in ("intern", "junior", "senior", "lead") if w in lower), ""),
           "must_have": must[:12], "nice_to_have": nice[:8], "responsibilities": resp,
           "years_experience": float(years.group(1)) if years else None}
    return {"job": job, "trace": _trace(s, "parse_jd", f"{job['title']}: {len(job['must_have'])} must-have, {len(job['nice_to_have'])} nice-to-have "
                                        "skills (lexicon match)")}


# --------------------------------------------------------------------------- 2. research (tool)


def research_company(s: AppState) -> dict:
    name = s["job"].get("company") or s.get("company") or ""
    if not name:
        return {"company_info": {"name": "", "summary": "", "source": ""},
                "trace": _trace(s, "research_company", "No company given; skipped")}
    info = company_research.invoke({"company": name})
    det = f"{info['source']}: {info['summary'][:90]}…" if info.get("summary") else f"No public summary found for {name}"
    return {"company_info": info, "trace": _trace(s, "research_company", det, "tool")}


# --------------------------------------------------------------------------- 3. RAG over resume


def index_resume(s: AppState) -> dict:
    bullets = split_bullets(s["resume_text"])
    retr = ResumeRetriever(bullets=bullets, k=2)
    evidence = {}
    for skill in s["job"]["must_have"] + s["job"]["nice_to_have"]:
        docs = retr.invoke(skill)
        canon = SK.canonical(skill)
        # keep only bullets that genuinely mention the skill (or a synonym)
        hits = [d.page_content for d in docs if canon in SK.find_skills(d.page_content) or SK.has_exact(d.page_content, skill)]
        if not hits:  # lexicon scan as a backstop
            hits = [b for b in bullets if canon in SK.find_skills(b)][:2]
        evidence[skill] = hits
    found = sum(1 for v in evidence.values() if v)
    return {"bullets": bullets, "evidence": evidence, "attempts": s.get("attempts", 0),
            "trace": _trace(s, "index_resume", f"{len(bullets)} resume bullets indexed (BM25); evidence found for "
                                               f"{found}/{len(evidence)} JD skills")}


# --------------------------------------------------------------------------- 4. scoring


class FitNote(BaseModel):
    summary: str = Field(description="2 sentences: overall fit and the biggest gap, addressed to the candidate")


def score_fit(s: AppState) -> dict:
    job, ev = s["job"], s["evidence"]
    resume_now = "\n".join(current_bullets(s))
    resume_skills = SK.find_skills("\n".join(s["bullets"]))
    fams = {SK.family(k) for k in resume_skills}

    def status(skill):
        if ev.get(skill):
            return "matched"
        if SK.family(SK.canonical(skill)) in fams:
            return "partial"
        return "missing"

    must = {k: status(k) for k in job["must_have"]}
    nice = {k: status(k) for k in job["nice_to_have"]}
    pts = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
    mcov = sum(pts[v] for v in must.values()) / max(len(must), 1)
    ncov = sum(pts[v] for v in nice.values()) / max(len(nice), 1) if nice else mcov
    fit = round(100 * (0.8 * mcov + 0.2 * ncov))

    terms = job["must_have"] + job["nice_to_have"]
    exact = [t for t in terms if SK.has_exact(resume_now, t)]
    ats = round(100 * len(exact) / max(len(terms), 1))

    missing = [k for k, v in must.items() if v == "missing"]
    summary = (f"You cover {sum(v == 'matched' for v in must.values())} of {len(must)} must-have skills. "
               + (f"Biggest gaps: {', '.join(missing[:3])}." if missing else "No must-have gaps."))
    mode = "rules"
    llm = get_llm(0)
    if llm is not None and s.get("attempts", 0) == 0:
        try:
            summary = llm.with_structured_output(FitNote).invoke(
                f"Job: {job}\nSkill coverage: must={must} nice={nice}\nFit score {fit}/100. Write the summary.").summary
            mode = "llm"
        except Exception as e:
            llm_failed("score_fit", e)
    elif s.get("fit", {}).get("summary"):
        summary = s["fit"]["summary"]

    attempt_note = f" (after tailoring round {s['attempts']})" if s.get("attempts") else ""
    return {"fit": {"score": fit, "must": must, "nice": nice, "summary": summary},
            "ats": {"score": ats, "exact": exact, "absent": [t for t in terms if t not in exact],
                    "history": list(s.get("ats", {}).get("history", [])) + [ats]},
            "trace": _trace(s, "score_fit", f"Fit {fit}/100 · ATS keyword match {ats}/100{attempt_note}", mode)}


def route_after_score(s: AppState) -> str:
    if s["fit"]["score"] < 35:
        return "gap_plan"
    if s["ats"]["score"] < s.get("target_ats", 75) and s.get("attempts", 0) < 2 and _tailorable(s):
        return "tailor_bullets"
    return "draft_cover_note"


def _tailorable(s: AppState) -> list[tuple[str, str]]:
    """(JD term, evidence bullet) pairs where the skill is evidenced but the JD's exact word is absent."""
    resume_now = "\n".join(current_bullets(s))
    out = []
    for term in s["ats"]["absent"]:
        for b in s["evidence"].get(term, []):
            if not SK.has_exact(resume_now, term):
                out.append((term, b))
                break
    return out


# --------------------------------------------------------------------------- 5. tailor + verify (loop)


def _group(pairs):
    g: dict[str, list[str]] = {}
    for t, b in pairs:
        g.setdefault(b, []).append(t)
    return g


class Rewrite(BaseModel):
    original: str
    rewritten: str
    skill: str


class Rewrites(BaseModel):
    items: list[Rewrite]


def tailor_bullets(s: AppState) -> dict:
    pairs = _tailorable(s)
    attempts = s.get("attempts", 0) + 1
    prev = list(s.get("tailored", []))
    cur = {b: b for b in s["bullets"]}
    for t in prev:
        if t["status"] == "accepted":
            cur[t["original"]] = t["rewritten"]

    new: list[dict] = []
    mode = "rules"
    llm = get_llm(0.2)
    if llm is not None and pairs:
        try:
            r: Rewrites = llm.with_structured_output(Rewrites).invoke(
                "Rewrite each resume bullet so it uses the job description's exact term, keeping it truthful. Rules: "
                "keep every fact, tool and number from the original; do NOT add tools, skills, numbers or claims "
                "that are not in the original; start with a strong verb; max 30 words.\n\n"
                + "\n".join(f"- terms: {', '.join(ts)} | original: {cur[b]}" for b, ts in _group(pairs).items()))
            by_orig = {it.original.strip(): it for it in r.items}
            for b, ts in _group(pairs).items():
                it = by_orig.get(cur[b].strip())
                if it:
                    new.append({"original": b, "rewritten": it.rewritten.strip(), "skill": ", ".join(ts), "status": "proposed"})
            mode = "llm"
        except Exception as e:
            llm_failed("tailor_bullets", e)
            new = []
    if not new:
        grouped: dict[str, list[str]] = {}
        for t, b in pairs:
            grouped.setdefault(b, []).append(t)
        for b, terms in grouped.items():
            rew = cur[b]
            for t in terms:
                surface = SK.find_skills(rew).get(SK.canonical(t))
                m = re.search(r"(?i)(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])", rew) if surface else None
                if m:   # keep the candidate's own wording, add the JD's term right after it
                    rew = rew[:m.end()] + f" ({t})" + rew[m.end():]
                else:
                    rew = rew.rstrip(".") + f" ({t})"
            new.append({"original": b, "rewritten": rew, "skill": ", ".join(terms), "status": "proposed"})

    keep = [t for t in prev if t["original"] not in {n["original"] for n in new}]
    return {"tailored": keep + new, "attempts": attempts,
            "trace": _trace(s, "tailor_bullets", f"Round {attempts}: rewrote {len(new)} bullet(s) to use JD terms "
                            + ", ".join(sorted({n['skill'] for n in new})[:5]), mode)}


def verify_claims(s: AppState) -> dict:
    out, problems = [], []
    for t in s["tailored"]:
        if t["status"] != "proposed":
            out.append(t)
            continue
        orig_sk = set(SK.find_skills(t["original"]))
        new_sk = set(SK.find_skills(t["rewritten"]))
        # the targeted JD term is allowed only if it maps to a skill already evidenced in the original
        targets = {SK.canonical(x.strip()) for x in t["skill"].split(",")}
        # a JD term is allowed only if the original bullet already evidences that skill (directly or by synonym)
        allowed = orig_sk | {x for x in targets if x in orig_sk}
        invented = sorted(new_sk - allowed)
        nums_o = set(re.findall(r"\d+(?:\.\d+)?", t["original"]))
        new_nums = sorted(set(re.findall(r"\d+(?:\.\d+)?", t["rewritten"])) - nums_o)
        if invented or new_nums:
            why = []
            if invented:
                why.append("adds " + ", ".join(invented))
            if new_nums:
                why.append("adds number(s) " + ", ".join(new_nums))
            problems.append({"bullet": t["rewritten"], "reason": "; ".join(why)})
            out.append({**t, "status": "reverted", "reason": "; ".join(why)})
        else:
            out.append({**t, "status": "accepted"})
    acc = sum(1 for t in out if t["status"] == "accepted")
    det = f"{acc} rewrite(s) verified against the original resume" + (f"; {len(problems)} reverted for unsupported claims" if problems else "")
    return {"tailored": out, "verification": list(s.get("verification", [])) + problems,
            "trace": _trace(s, "verify_claims", det)}


# --------------------------------------------------------------------------- 6. early exit


def gap_plan(s: AppState) -> dict:
    must = s["fit"]["must"]
    missing = [k for k, v in must.items() if v == "missing"]
    partial = [k for k, v in must.items() if v == "partial"]
    steps = [f"Build a small project that uses {m} and add it to your resume" for m in missing[:3]]
    steps += [f"Turn your related experience into direct {p} experience (e.g. port one project to {p})" for p in partial[:2]]
    plan = {"recommendation": "Not a strong match yet — apply to closer roles first, or close these gaps.",
            "missing": missing, "partial": partial, "steps": steps}
    return {"plan": plan, "status": "not_recommended",
            "trace": _trace(s, "gap_plan", f"Fit too low; {len(missing)} must-have gap(s), plan with {len(steps)} steps")}


# --------------------------------------------------------------------------- 7. cover note + HITL


class CoverNote(BaseModel):
    note: str = Field(description="Cover note, 120-180 words, 3 short paragraphs, no placeholders like [Name]")


def _template_note(s: AppState) -> str:
    job, info = s["job"], s.get("company_info", {})
    generic = {"language", "database", "devops", "soft"}
    matched = sorted([k for k, v in s["fit"]["must"].items() if v == "matched"],
                     key=lambda k: SK.family(SK.canonical(k)) in generic)[:4]
    ev = []
    for k in matched:
        for b in s["evidence"].get(k, []):
            if b not in ev and b.count(",") < 4:
                ev.append(b)
                break
    company = job.get("company") or "your team"
    why = ""
    if info.get("summary"):
        first = re.split(r"(?<=[.!?])\s+", info["summary"])[0]
        why = f" {first.rstrip('.')} — that scale of impact is exactly why this role stands out to me."
    p1 = f"Dear Hiring Team,\n\nI'm applying for the {job['title']} role at {company}.{why}"
    p2 = "Relevant work: " + " ".join(f"{e.rstrip('.')}." for e in ev[:2]) if ev else ""
    p2 += f" These map directly to the {', '.join(matched)} requirements in your description." if matched else ""
    p3 = "I'd welcome the chance to discuss how I can contribute. Thank you for your time."
    fb = re.sub(r"^(please\s+)?(mention|add|say|include|write|state)\s+(that\s+)?", "", s.get("feedback", "").strip(), flags=re.I)
    if fb:
        p3 = fb[0].upper() + fb[1:].rstrip(".") + ". " + p3
    return "\n\n".join(x for x in [p1, p2.strip(), p3] if x)


def draft_cover_note(s: AppState) -> dict:
    v = s.get("cover_version", 0) + 1
    llm = get_llm(0.5)
    if llm is not None:
        try:
            note = llm.with_structured_output(CoverNote).invoke(
                "Write a concise, specific cover note for this application. Use only facts from the resume bullets; "
                "reference one concrete thing about the company if the research has it.\n\n"
                f"Job: {s['job']}\nCompany research: {s.get('company_info')}\n"
                f"Resume (tailored): {current_bullets(s)}\nFit: {s['fit']['summary']}\n"
                + (f"Previous draft: {s.get('cover_note')}\nCandidate's requested changes: {s['feedback']}\n" if s.get("feedback") else "")
            ).note
            return {"cover_note": note, "cover_version": v, "feedback": "",
                    "trace": _trace(s, "draft_cover_note", f"Version {v} written" + (" with your changes" if s.get("feedback") else ""), "llm")}
        except Exception as e:
            llm_failed("draft_cover_note", e)
    return {"cover_note": _template_note(s), "cover_version": v, "feedback": "",
            "trace": _trace(s, "draft_cover_note", f"Version {v} from template" + (" with your changes" if s.get("feedback") else ""))}


def human_review(s: AppState) -> dict:
    d = s.get("decision")
    det = "Approved the application package" if d == "approve" else f"Asked for changes: “{s.get('feedback', '')}”"
    return {"trace": _trace(s, "human_review", det, "human")}


def route_after_review(s: AppState) -> str:
    return "finalize" if s.get("decision") == "approve" else "draft_cover_note"


# --------------------------------------------------------------------------- 8. finalize


class Prep(BaseModel):
    questions: list[str] = Field(description="6 likely interview questions for this candidate and role")


def finalize(s: AppState) -> dict:
    job = s["job"]
    gaps = [k for k, v in s["fit"]["must"].items() if v != "matched"]
    qs = [f"Walk me through a project where you used {k}." for k in [k for k, v in s["fit"]["must"].items() if v == "matched"][:3]]
    qs += [f"We use {g} heavily. How would you get up to speed?" for g in gaps[:2]]
    qs += [f"How would you approach: {r}?" for r in job.get("responsibilities", [])[:1]]
    mode = "rules"
    llm = get_llm(0.3)
    if llm is not None:
        try:
            qs = llm.with_structured_output(Prep).invoke(
                f"Job: {job}\nCandidate resume: {current_bullets(s)}\nGaps: {gaps}\nList the questions.").questions
            mode = "llm"
        except Exception as e:
            llm_failed("finalize", e)
    pkg = {"title": job["title"], "company": job.get("company", ""), "fit": s["fit"]["score"],
           "ats_before": s["ats"]["history"][0], "ats_after": s["ats"]["score"],
           "resume": current_bullets(s), "cover_note": s["cover_note"], "interview_questions": qs}
    return {"package": pkg, "status": "ready",
            "trace": _trace(s, "finalize", f"Package ready: tailored resume, cover note, {len(qs)} interview questions", mode)}


# --------------------------------------------------------------------------- graph


RESUME_FROM = {"human_review": "draft_cover_note"}


def build_graph():
    g = StateGraph(AppState)
    for n, f in [("parse_jd", parse_jd), ("research_company", research_company), ("index_resume", index_resume),
                 ("score_fit", score_fit), ("tailor_bullets", tailor_bullets), ("verify_claims", verify_claims),
                 ("gap_plan", gap_plan), ("draft_cover_note", draft_cover_note), ("human_review", human_review),
                 ("finalize", finalize)]:
        g.add_node(n, f)
    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "research_company")
    g.add_edge("research_company", "index_resume")
    g.add_edge("index_resume", "score_fit")
    g.add_conditional_edges("score_fit", route_after_score, ["gap_plan", "tailor_bullets", "draft_cover_note"])
    g.add_edge("tailor_bullets", "verify_claims")
    g.add_edge("verify_claims", "score_fit")
    g.add_edge("gap_plan", END)
    g.add_edge("draft_cover_note", "human_review")
    g.add_conditional_edges("human_review", route_after_review, ["finalize", "draft_cover_note"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])


def _finish(graph, cfg) -> dict:
    snap = graph.get_state(cfg)
    vals = dict(snap.values)
    paused = snap.next[0] if snap.next else None
    if paused:
        vals["status"] = "awaiting_review"
    return {"state": vals, "paused_at": paused}


def start(jd_text: str, resume_text: str, company: str = "", target_ats: int = 75) -> dict:
    graph, cfg = build_graph(), {"configurable": {"thread_id": "app"}}
    graph.invoke({"jd_text": jd_text, "resume_text": resume_text, "company": company, "target_ats": target_ats,
                  "trace": [], "attempts": 0}, cfg)
    return _finish(graph, cfg)


def resume(state: dict, decision: str, feedback: str = "") -> dict:
    graph, cfg = build_graph(), {"configurable": {"thread_id": "app"}}
    graph.update_state(cfg, {**state, "decision": decision, "feedback": feedback if decision != "approve" else ""},
                       as_node=RESUME_FROM["human_review"])
    graph.invoke(None, cfg)
    return _finish(graph, cfg)


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
