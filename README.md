# JobPilot — a LangGraph agent that tailors job applications honestly

Paste a job description and your resume. JobPilot parses the JD, researches the company with a tool call, retrieves evidence from your resume (RAG), scores your fit, rewrites bullets to use the JD's language in a loop, **checks every rewrite against your original resume so it never invents experience**, drafts a cover note, and pauses for your approval before building the final package: tailored bullets, cover note and likely interview questions.

**Live demo:** https://jobpilot-agent-nine.vercel.app

## The agent graph

```mermaid
graph TD
    START([start]) --> parse_jd --> research_company --> index_resume --> score_fit
    score_fit -. "fit < 35" .-> gap_plan --> END([end])
    score_fit -. "ATS < target & rounds < 2" .-> tailor_bullets --> verify_claims --> score_fit
    score_fit -. "good enough" .-> draft_cover_note --> human_review
    human_review:::human -. revise .-> draft_cover_note
    human_review -. approve .-> finalize --> END
    classDef human fill:#efebff,stroke:#5b3fd1
```

| Node | Type | What it does |
|---|---|---|
| `parse_jd` | LLM structured output | JD → Pydantic `JobSpec` (title, must-have / nice-to-have skills, responsibilities, years) |
| `research_company` | **LangChain tool** | `@tool company_research`: Tavily web search if a key is set, otherwise the Wikipedia API |
| `index_resume` | **RAG** | Splits the resume into bullets and uses a custom LangChain `BaseRetriever` (BM25 + skill-synonym query expansion) to find evidence for every JD skill |
| `score_fit` | Rules + LLM | Two scores: **real fit** (synonym-aware: "GPT-4" counts as LLMs) and **ATS keyword match** (the JD's exact words) |
| `tailor_bullets` | LLM | Rewrites evidenced bullets to use the JD's exact terms, keeping every fact |
| `verify_claims` | Rules | Honesty check: reverts any rewrite that adds a skill or number not in the original bullet |
| `gap_plan` | Rules | Early exit for poor matches, with a concrete plan to close the gaps |
| `draft_cover_note` | LLM | Specific cover note grounded in your bullets and the company research |
| `human_review` | **Human-in-the-loop** | `interrupt_before`: you approve or ask for changes, which loop back into the draft |
| `finalize` | LLM | Final package plus interview questions aimed at your gaps |

## Why two scores?

Most "resume optimisers" push a single match score up by stuffing keywords, which is how candidates end up claiming tools they've never used. JobPilot separates the two:

- **Real fit** is fixed by what's actually in your resume. Tailoring cannot change it.
- **ATS match** measures whether keyword filters will see your experience. Tailoring raises it honestly, because `verify_claims` only allows a JD term onto a bullet that already shows that skill (e.g. "Azure OpenAI" → "Azure OpenAI (LLMs)").

## Engineering notes

- **Loops with a hard stop:** `score_fit → tailor_bullets → verify_claims → score_fit` runs at most 2 rounds and stops early when nothing more can be tailored honestly.
- **Stateless human-in-the-loop on serverless:** the paused state goes back to the browser; on resume the API rebuilds the graph and calls `update_state(..., as_node="draft_cover_note")`, so LangGraph continues from the interrupt.
- **Graceful degradation:** every LLM node has a deterministic fallback, so the demo works with no API keys; the UI tags each step as LLM, tool, rules or human.
- **Tests:** every graph path, the honesty checker, skill matching and the LLM paths (with a stub model): `pytest -q`.

## Stack

LangGraph 1.x · LangChain Core (tools, retrievers, structured output) · langchain-groq (Llama 3.3 70B) · rank-bm25 · httpx · FastAPI · vanilla JS · Vercel

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...      # optional
export TAVILY_API_KEY=...    # optional, better company research
uvicorn app:app --reload
pytest -q
```

---
Built by **Niket Korgaonkar** · [GitHub](https://github.com/NiKeT972)
