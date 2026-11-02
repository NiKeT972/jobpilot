import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.pop("GROQ_API_KEY", None); os.environ.pop("TAVILY_API_KEY", None)
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import jobpilot.graph as G
import jobpilot.tools as T
from jobpilot.samples import SAMPLES
from jobpilot import skills as SK
from app import app

c = TestClient(app)
AI = SAMPLES["ai"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    class Fake:
        def invoke(self, args):
            return {"name": args["company"], "summary": f"{args['company']} is an Indian fintech company that builds payment products.",
                    "source": "Wikipedia", "links": []}
    monkeypatch.setattr(G, "company_research", Fake())


def test_good_match_loops_then_pauses_then_finalizes():
    r = c.post("/api/run", json={"jd_text": AI["jd"], "resume_text": AI["resume"], "company": AI["company"]}).json()
    s = r["state"]
    nodes = [t["node"] for t in s["trace"]]
    assert r["paused_at"] == "human_review"
    assert "tailor_bullets" in nodes and nodes.count("score_fit") >= 2
    assert s["ats"]["history"][-1] > s["ats"]["history"][0]
    assert s["fit"]["must"]["LangGraph"] == "missing"          # honest: sample resume has no LangGraph
    assert all(t["status"] in ("accepted", "reverted") for t in s["tailored"])
    assert "Indian fintech" in s["cover_note"]
    r2 = c.post("/api/resume", json={"state": s, "decision": "revise", "feedback": "mention I can join immediately"}).json()
    assert r2["paused_at"] == "human_review" and r2["state"]["cover_version"] == 2
    assert "join immediately" in r2["state"]["cover_note"].lower()
    r3 = c.post("/api/resume", json={"state": r2["state"], "decision": "approve"}).json()
    assert r3["paused_at"] is None and r3["state"]["status"] == "ready"
    assert len(r3["state"]["package"]["interview_questions"]) >= 4


def test_poor_match_exits_early():
    d = SAMPLES["de"]
    r = c.post("/api/run", json={"jd_text": d["jd"], "resume_text": d["resume"], "company": d["company"]}).json()
    assert r["state"]["status"] == "not_recommended" and r["paused_at"] is None
    assert "Spark" in r["state"]["plan"]["missing"]


def test_verify_reverts_invented_skills():
    s = {"tailored": [
        {"original": "Built a chatbot on GPT-4 with FastAPI", "rewritten": "Built a LangGraph agent on GPT-4 with FastAPI on Kubernetes", "skill": "LangGraph", "status": "proposed"},
        {"original": "Built a chatbot on GPT-4", "rewritten": "Built a chatbot on GPT-4 (LLMs) serving 5000 users", "skill": "LLMs", "status": "proposed"},
        {"original": "Built a chatbot on GPT-4", "rewritten": "Built an LLMs chatbot on GPT-4", "skill": "LLMs", "status": "proposed"}],
        "trace": []}
    out = G.verify_claims(s)["tailored"]
    assert out[0]["status"] == "reverted" and "LangGraph" in out[0]["reason"]
    assert out[1]["status"] == "reverted" and "5000" in out[1]["reason"]
    assert out[2]["status"] == "accepted"


def test_skill_canonical():
    assert SK.canonical("LLM Evaluation") == "LLM Evaluation"
    assert SK.canonical("gpt-4o") == "LLMs"
    assert "R" not in SK.find_skills("R&D team")


class Stub:
    def with_structured_output(self, schema):
        class R:
            def invoke(_, prompt):
                n = schema.__name__
                if n == "JobSpec":
                    return schema(title="AI Engineer", company="", must_have=["Python", "LLMs", "RAG", "LangGraph"],
                                  nice_to_have=["AWS"], responsibilities=["Ship agents"])
                if n == "FitNote":
                    return schema(summary="Good fit. Gap: LangGraph.")
                if n == "Rewrites":
                    import re
                    items = [schema.model_fields["items"].annotation.__args__[0](original=m.group(2), rewritten=m.group(2) + " using LLMs", skill=m.group(1))
                             for m in re.finditer(r"- terms: (.*?) \| original: (.*)", prompt)]
                    return schema(items=items)
                if n == "CoverNote":
                    return schema(note="Dear team, ...")
                if n == "Prep":
                    return schema(questions=["q1", "q2", "q3", "q4", "q5", "q6"])
                raise AssertionError(n)
        return R()


def test_llm_paths(monkeypatch):
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: Stub())
    r = G.start(AI["jd"], AI["resume"], "Razorpay", target_ats=95)
    modes = {t["node"]: t["mode"] for t in r["state"]["trace"]}
    assert modes["parse_jd"] == "llm" and modes["draft_cover_note"] == "llm"
    r = G.resume(r["state"], "approve")
    assert r["state"]["package"]["interview_questions"][0] == "q1"
