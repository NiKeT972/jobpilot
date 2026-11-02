import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Set (not pop) the keys: load_dotenv(override=False) never replaces an existing variable, so a real key in
# .env cannot leak into these offline tests. An empty string still means "no LLM".
os.environ["GROQ_API_KEY"] = ""; os.environ["TAVILY_API_KEY"] = ""
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
    r2 = c.post("/api/resume", json={"token": r["token"], "decision": "revise", "feedback": "mention I can join immediately"}).json()
    assert r2["paused_at"] == "human_review" and r2["state"]["cover_version"] == 2
    assert "join immediately" in r2["state"]["cover_note"].lower()
    r3 = c.post("/api/resume", json={"token": r2["token"], "decision": "approve"}).json()
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
                    items = [schema.model_fields["items"].annotation.__args__[0](id=int(m.group(1)), rewritten=m.group(2) + " using LLMs")
                             for m in re.finditer(r"\[(\d+)\] terms: .*? \| original: (.*)", prompt)]
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


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    from dotenv import load_dotenv
    import jobpilot.llm as L
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from-dotenv-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "")
    load_dotenv(env, override=False)            # same call llm.py makes at import
    assert os.environ["GROQ_API_KEY"] == ""     # the existing (empty) value wins
    assert not L.llm_available() and L.get_llm() is None


def test_llm_error_falls_back_to_rules_and_logs(monkeypatch, caplog):
    class Broken:
        def with_structured_output(self, schema):
            class R:
                def invoke(_, prompt):
                    raise ValueError("tool_use_failed")   # what Groq returns when the model emits bad JSON
            return R()
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: Broken())
    with caplog.at_level("WARNING", logger="jobpilot"):
        r = G.start(AI["jd"], AI["resume"], "Razorpay")
    assert r["paused_at"] == "human_review"
    assert all(t["mode"] != "llm" for t in r["state"]["trace"])
    assert "parse_jd failed" in caplog.text and "tool_use_failed" in caplog.text


def _start_ai():
    return c.post("/api/run", json={"jd_text": AI["jd"], "resume_text": AI["resume"], "company": AI["company"]}).json()


def test_tampered_token_is_rejected():
    import base64, json, zlib
    r = _start_ai()
    body, sig = r["token"].rsplit(".", 1)
    paused = json.loads(zlib.decompress(base64.urlsafe_b64decode(body)))
    paused["state"]["tailored"][0]["rewritten"] = "Led a 50-person LangGraph team at Google"   # invented claim
    forged = base64.urlsafe_b64encode(zlib.compress(json.dumps(paused).encode())).decode() + "." + sig
    res = c.post("/api/resume", json={"token": forged, "decision": "approve"})
    assert res.status_code == 400 and "no longer valid" in res.json()["detail"]
    assert c.post("/api/resume", json={"token": "garbage", "decision": "approve"}).status_code == 400


def test_token_signed_with_other_key_is_rejected(monkeypatch):
    r = _start_ai()
    monkeypatch.setenv("STATE_SIGNING_KEY", "rotated-key")
    assert c.post("/api/resume", json={"token": r["token"], "decision": "approve"}).status_code == 400


def test_resume_input_validation():
    r = _start_ai()
    assert c.post("/api/resume", json={"token": r["token"], "decision": "publish"}).status_code == 422
    assert c.post("/api/resume", json={"token": r["token"], "decision": "revise", "feedback": "  "}).status_code == 422
    assert c.post("/api/run", json={"jd_text": "x" * 20_000, "resume_text": AI["resume"]}).status_code == 422


def test_tailor_matches_llm_output_by_id(monkeypatch):
    """The model may return items out of order and re-type the original text; ids still line up."""
    class IdStub:
        def with_structured_output(self, schema):
            class R:
                def invoke(_, prompt):
                    import re
                    if schema.__name__ != "Rewrites":
                        raise RuntimeError("only tailoring is stubbed")
                    found = re.findall(r"\[(\d+)\] terms: .*? \| original: (.*)", prompt)
                    Item = schema.model_fields["items"].annotation.__args__[0]
                    return schema(items=[Item(id=int(i), rewritten=o.replace("Built", "Developed")) for i, o in reversed(found)]
                                  + [Item(id=99, rewritten="out-of-range id is ignored")])
            return R()
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: IdStub())
    r = G.start(AI["jd"], AI["resume"], "Razorpay")
    s = r["state"]
    tailor = next(t for t in s["trace"] if t["node"] == "tailor_bullets")
    assert tailor["mode"] == "llm"
    assert s["tailored"] and all(t["original"] in s["bullets"] for t in s["tailored"])
    assert not any("out-of-range" in t["rewritten"] for t in s["tailored"])


def test_model_setting(monkeypatch):
    import jobpilot.llm as L
    monkeypatch.setenv("GROQ_MODEL", "some/model")
    assert L.model_name() == "built-in rules"            # no key -> rules, even if a model is configured
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert L.model_name() == "some/model"
    monkeypatch.delenv("GROQ_MODEL")
    assert L.model_name() == L.DEFAULT_MODEL != "llama-3.3-70b-versatile"   # retired by Groq (404)


def test_honesty_eval_does_not_regress():
    """scripts/eval_honesty.py over evals/honesty_cases.jsonl (28 hand-labelled rewrites)."""
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval_honesty.py"
    spec = importlib.util.spec_from_file_location("eval_honesty", path)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    r = ev.run()
    assert r["recall"] == 1.0, r["mistakes"]          # every known invented claim is reverted
    assert r["precision"] >= 0.9, r["mistakes"]       # at most ~1 in 10 reverts is a false alarm


def test_verify_catches_non_skill_claims_and_allows_implied_terms():
    s = {"trace": [], "tailored": [
        {"original": "Built a customer-support chatbot", "rewritten": "Built a customer-support chatbot at Google", "skill": "LLMs", "status": "proposed"},
        {"original": "Wrote FastAPI endpoints", "rewritten": "Led a team writing FastAPI endpoints", "skill": "FastAPI", "status": "proposed"},
        {"original": "Classifier with scikit-learn and XGBoost", "rewritten": "Classifier (Machine Learning) with scikit-learn and XGBoost", "skill": "Machine Learning", "status": "proposed"},
        {"original": "Classifier with scikit-learn", "rewritten": "Classifier with scikit-learn and PyTorch", "skill": "Machine Learning", "status": "proposed"}]}
    out = G.verify_claims(s)["tailored"]
    assert out[0]["status"] == "reverted" and "Google" in out[0]["reason"]
    assert out[1]["status"] == "reverted" and "led" in out[1]["reason"] and "team" in out[1]["reason"]
    assert out[2]["status"] == "accepted"                                  # scikit-learn/XGBoost imply ML
    assert out[3]["status"] == "reverted" and "PyTorch" in out[3]["reason"]  # implication never adds new tools
