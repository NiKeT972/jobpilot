"""LangChain tools the agent calls."""
import os

import httpx
from langchain_core.tools import tool


@tool
def company_research(company: str) -> dict:
    """Look up a short public summary of a company (Tavily web search if configured, otherwise Wikipedia)."""
    key = os.getenv("TAVILY_API_KEY")
    if key:
        try:
            r = httpx.post("https://api.tavily.com/search", timeout=12,
                           json={"api_key": key, "query": f"{company} company what they do products recent news",
                                 "max_results": 3, "include_answer": True})
            d = r.json()
            if d.get("answer"):
                return {"name": company, "summary": d["answer"], "source": "Tavily web search",
                        "links": [x["url"] for x in d.get("results", [])][:3]}
        except Exception:
            pass
    try:
        h = {"User-Agent": "JobPilot/1.0 (portfolio project)"}
        s = httpx.get("https://en.wikipedia.org/w/api.php", timeout=8, headers=h, params={
            "action": "query", "list": "search", "srsearch": f"{company} company", "format": "json", "srlimit": 1}).json()
        hits = s.get("query", {}).get("search", [])
        if hits:
            title = hits[0]["title"]
            p = httpx.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
                          timeout=8, headers=h).json()
            if p.get("extract") and company.split()[0].lower() in (p.get("title", "") + p["extract"]).lower():
                return {"name": company, "summary": p["extract"], "source": "Wikipedia",
                        "links": [p.get("content_urls", {}).get("desktop", {}).get("page", "")]}
    except Exception:
        pass
    return {"name": company, "summary": "", "source": "none", "links": []}
