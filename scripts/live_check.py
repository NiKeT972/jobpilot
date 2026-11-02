"""End-to-end check against a running server, showing what each node produced.

    uvicorn app:app --port 8000          # terminal 1 (reads GROQ_API_KEY from .env)
    python scripts/live_check.py         # terminal 2

Runs both samples through the HTTP API: the good match goes run -> revise -> approve, the poor match exits early.
For every node it prints whether the LLM or the rule-based fallback ran, plus that node's output.
If a node shows "rules" while the key is set, the server log has a "LLM call in <node> failed" line saying why.
"""
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
c = httpx.Client(base_url=BASE, timeout=180)


def show_trace(state, seen):
    for t in state["trace"][seen:]:
        print(f"  [{t['mode']:>5}] {t['node']:<17} {t['detail']}")
    return len(state["trace"])


def dump(title, value):
    print(f"\n  --- {title} ---")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    print("  " + text.replace("\n", "\n  "))


health = c.get("/api/health").json()
print(f"Server: {BASE}  llm={health['llm']}  model={health['model']}")
if not health["llm"]:
    print("WARNING: no GROQ_API_KEY on the server, so everything below is the rule-based fallback.")
samples = c.get("/api/meta").json()["samples"]

# ---- good match: run, revise once, approve
ai = samples["ai"]
print("\n=== Sample 'ai' (good match): run ===")
r = c.post("/api/run", json={"jd_text": ai["jd"], "resume_text": ai["resume"], "company": ai["company"]}).json()
s = r["state"]
seen = show_trace(s, 0)
dump("parse_jd -> job", s["job"])
dump("research_company", s.get("company_info", {}).get("summary") or "(no summary)")
dump("score_fit -> fit summary", s["fit"]["summary"])
dump("tailor_bullets / verify_claims", [{k: t[k] for k in ("original", "rewritten", "status")} | ({"reason": t["reason"]} if "reason" in t else {})
                                        for t in s.get("tailored", [])])
dump(f"draft_cover_note v{s['cover_version']}", s["cover_note"])
assert r["paused_at"] == "human_review", r["paused_at"]

print("\n=== Human review: revise ===")
r = c.post("/api/resume", json={"state": s, "decision": "revise",
                                "feedback": "Mention I can join immediately and keep it under 150 words"}).json()
s = r["state"]
seen = show_trace(s, seen)
dump(f"draft_cover_note v{s['cover_version']}", s["cover_note"])

print("\n=== Human review: approve ===")
r = c.post("/api/resume", json={"state": s, "decision": "approve"}).json()
s = r["state"]
show_trace(s, seen)
dump("finalize -> interview questions", s["package"]["interview_questions"])
print(f"\n  status={s['status']}  ATS {s['package']['ats_before']} -> {s['package']['ats_after']}  fit={s['package']['fit']}")

# ---- poor match: early exit, no human review
de = samples["de"]
print("\n=== Sample 'de' (poor match): run ===")
r = c.post("/api/run", json={"jd_text": de["jd"], "resume_text": de["resume"], "company": de["company"]}).json()
s = r["state"]
show_trace(s, 0)
dump("parse_jd -> job", s["job"])
dump("gap_plan", s.get("plan"))
print(f"\n  status={s['status']}  paused_at={r['paused_at']}")
