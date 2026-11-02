"""Resume RAG: split a resume into bullets and retrieve evidence for a skill with BM25 + skill-synonym expansion."""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from . import skills as SK


def split_bullets(text: str) -> list[str]:
    lines = text.splitlines()
    marked = [l for l in lines if re.match(r"^\s*[-•*▪●◦·]\s+", l)]
    if len(marked) >= 3:            # resume uses bullet markers: ignore headings, titles, dates
        lines = marked
    out = []
    for line in lines:
        line = re.sub(r"^[\s\-•*▪●◦·]+", "", line).strip()
        if len(line) < 25:          # skip headings, names, dates
            continue
        out.append(line)
    return out


def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9+#./-]+", t.lower())


class ResumeRetriever(BaseRetriever):
    bullets: list[str]
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if not self.bullets:
            return []
        canon = SK.canonical(query)
        syns = SK.SKILLS.get(canon, ([], ""))[0]
        q = _tok(query) + [w for s in syns for w in _tok(s)]
        bm = BM25Okapi([_tok(b) for b in self.bullets])
        scores = bm.get_scores(q)
        ranked = sorted(range(len(self.bullets)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [Document(page_content=self.bullets[i], metadata={"score": float(scores[i])}) for i in ranked if scores[i] > 0]
