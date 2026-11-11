# JobPilot: notes for Claude Code

LangGraph agent that tailors a job application: parse JD → company research (tool) → RAG over resume → fit + ATS scores → tailor/verify loop → cover note → human approval → final package. Portfolio project by Niket Korgaonkar; he must be able to explain every line in interviews, so keep code simple and explain changes.

Live: https://jobpilot-agent-nine.vercel.app (Vercel **auto-deploys on push to `main`**; there is no vercel.json, Vercel detects the FastAPI `app` in `app.py`).

## Layout
- `app.py`: FastAPI. `GET /api/health`, `GET /api/meta` (samples + mermaid), `POST /api/run`, `POST /api/resume`, `/` serves `public/index.html`.
- `jobpilot/graph.py`: all nodes, routers, `build_graph()`, `start()`, `resume()`. The module docstring has the flow.
- `jobpilot/llm.py`: `get_llm()` returns `ChatGroq` or `None` when `GROQ_API_KEY` is unset. Loads `.env` with `override=False`. `llm_failed()` logs why a node fell back.
- `jobpilot/skills.py`: skill lexicon (canonical → synonyms, family). Drives matching, ATS scoring and the honesty check.
- `jobpilot/retriever.py`: custom `BaseRetriever` (BM25 + synonym query expansion) over resume bullets.
- `jobpilot/tools.py`: `@tool company_research` (Tavily if `TAVILY_API_KEY`, else Wikipedia API, else empty).
- `jobpilot/files.py`: resume upload → text (`pypdf`, `python-docx`; `normalise()` re-joins wrapped PDF bullets) and `package_docx()` for the Word download. Endpoints `POST /api/resume-file` (multipart, 2 MB cap) and `POST /api/package.docx`.
- `evals/honesty_cases.jsonl` + `scripts/eval_honesty.py`: labelled rewrites → recall/precision of `verify_claims`. `test_honesty_eval_does_not_regress` enforces recall 1.0, precision ≥ 0.9. Add a case for every honesty bug you fix, labelled before looking at the result.
- `evals/fit_cases.json` + `scripts/eval.py`: end-to-end eval (8 JDs × 4 resumes = 18 labelled pairs + a 6-pair `holdout` block) → `evals/report_*.json`, served by `GET /api/eval` and shown in the UI. `test_end_to_end_eval_does_not_regress` enforces it. Company research is stubbed offline in the eval. Don't tune rules on the holdout block.
- `jobpilot/samples.py`: two demo cases: `ai` (good match → review) and `de` (poor match → `gap_plan` early exit).
- `public/index.html`: vanilla JS UI. It renders `state` and sends only the signed `token` back on resume.
- `scripts/live_check.py`: drives both samples and every review path through a running server and prints each node's output.

CI: `.github/workflows/tests.yml` runs `pytest -q` on push/PR (no keys needed).

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --reload            # http://127.0.0.1:8000
.venv/bin/python scripts/live_check.py        # in a 2nd terminal, against the running server
```
Key: put `GROQ_API_KEY=...` in `.env` (gitignored; template in `.env.example`). Never commit it or ask for it in chat. With no key, everything runs on rules.

## Design rules / gotchas
- **Model:** `DEFAULT_MODEL = "openai/gpt-oss-120b"` in `llm.py`, overridable with `GROQ_MODEL`. Groq retired `llama-3.3-70b-versatile` (2026-08-16); it returned 404 `model_not_found` and every node silently fell back to rules. If all nodes show `rules` with a key set, check the logs for `model_not_found` first.
- **Every LLM node has a rule-based fallback.** Pattern: `llm = get_llm(); if llm: try: ... return llm result; except Exception as e: llm_failed(node, e)` → then the rules path. Keep both paths working and tag trace entries with `mode` (`llm`/`rules`/`tool`/`human`); the UI shows it.
- **Stateless human-in-the-loop:** compiled with `MemorySaver` + `interrupt_before=["human_review"]`, but each request builds a fresh graph. `resume()` re-seeds with `update_state(cfg, state, as_node="draft_cover_note")` so the next node is `human_review`. Don't rely on the checkpointer persisting between requests (serverless).
- **Signed paused state:** `/api/run` and `/api/resume` return `token` (HMAC-SHA256 over zlib+base64 JSON, `jobpilot/state_token.py`) when paused; `/api/resume` takes `{token, decision: approve|revise, feedback}` and never accepts raw state. Key from `STATE_SIGNING_KEY`; without it a public dev key is used (`/api/health` shows `state_signing: false`). The UI sends `CUR.token`.
- **Honesty check** (`verify_claims`) reverts rewrites that add a skill (per `skills.find_skills`), a number, a `CLAIM_WORDS` word (led, team, production, customers, award…) or a new capitalised name not in the original bullet. A targeted JD term is allowed if a more specific skill in the original implies it (`skills.IMPLIED_BY`, e.g. scikit-learn → Machine Learning). JD terms outside the skill lexicon look like new names, so they get reverted (conservative). Fit score depends only on the original resume; only ATS can rise with tailoring.
- **Partial credit** (`partial` = 0.5) only within `skills.TRANSFERABLE` families. Broad families (language, devops, database, soft) give no partial credit.
- **JD parsing (rules):** if `REQ_HEADING` (Requirements / Must have / What we need / Key skills…) is found, only that section gives must-haves; title and "what you'll do" skills become nice-to-haves.
- `skills.find_skills()` drops a match that sits inside a longer match of another skill ("GitHub Actions" is CI/CD only).
- **LangSmith:** opt-in via `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`; run names/tags from `llm.run_config()`. Tests set `LANGSMITH_TRACING=false`.
- Tailor loop: max 2 rounds (`attempts`), routed in `route_after_score`; fit < 35 → `gap_plan`.
- `tailor_bullets` LLM path numbers the bullets `[id]` in the prompt and the model returns `{id, rewritten}`; unknown ids are ignored, and if nothing usable comes back the rules path runs.
- Structured output uses `llm.with_structured_output(PydanticModel)` (Groq tool calling). Groq returns `tool_use_failed` (400) when the model emits invalid arguments; that triggers the fallback and logs a warning.
- **Tests must never hit the network:** `tests/test_jobpilot.py` sets `GROQ_API_KEY=""` *before* importing (not `pop`), so `load_dotenv(override=False)` can't pull a real key from `.env`. `company_research` is monkeypatched. LLM paths are tested with a stub model.
- `uvicorn` and `pytest` live in `requirements-dev.txt`; `requirements.txt` is what Vercel installs.
- The cloud Claude Code container blocks `api.groq.com` and `en.wikipedia.org` by default; allow them in the environment's network settings to test the live LLM there.
