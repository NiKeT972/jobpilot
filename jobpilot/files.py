"""Turn an uploaded resume (PDF, DOCX or plain text) into plain text, one bullet per line.

The agent splits the resume into bullets line by line (retriever.split_bullets), so the main job here is
undoing what PDF layout does to text: bullets drawn as glyphs (•, ▪, ●) and long bullets wrapped over
several lines. Everything runs in memory; the file is never written to disk.
"""
from __future__ import annotations

import io
import re

MAX_BYTES = 2 * 1024 * 1024          # 2 MB is plenty for a resume and keeps the serverless function fast
SUPPORTED = (".pdf", ".docx", ".txt", ".md")

# Characters resumes use as bullet points; all are normalised to "- " so split_bullets recognises them.
_BULLET = re.compile(r"^\s*(?:[•●▪◦‣∙·■□➢➤►▶✓✔\-–—*]||)\s*")


class UnsupportedFile(ValueError):
    pass


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            lines.append("")
            continue
        # Word list items ("List Bullet" style) don't contain a bullet character in their text.
        list_style = p.style is not None and "List" in (p.style.name or "")
        numbered = p._p.pPr is not None and p._p.pPr.numPr is not None      # bullets applied as numbering
        lines.append(f"- {text}" if (list_style or numbered) and not _BULLET.match(text) else text)
    for table in doc.tables:                       # some resume templates put everything in tables
        for row in table.rows:
            for cell in row.cells:
                lines.extend(cell.text.splitlines())
    return "\n".join(lines)


def normalise(text: str) -> str:
    """One bullet per line: unify bullet markers and re-join bullets that the PDF wrapped onto new lines."""
    out: list[str] = []
    for raw in text.replace("\r", "").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if _BULLET.match(line):
            out.append("- " + _BULLET.sub("", line))
        elif out and out[-1].startswith("- ") and not out[-1].endswith((".", ":", ";")) and line[:1].islower():
            out[-1] += " " + line           # continuation of a wrapped bullet ("...with a retrieval" / "pipeline over...")
        else:
            out.append(line)
    return "\n".join(out).strip()


def extract_resume_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if len(data) > MAX_BYTES:
        raise UnsupportedFile("File is larger than 2 MB.")
    if name.endswith(".pdf"):
        text = _pdf_text(data)
    elif name.endswith(".docx"):
        text = _docx_text(data)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFile(f"Unsupported file type. Upload one of: {', '.join(SUPPORTED)}")
    text = normalise(text)
    if len(text) < 40:
        raise UnsupportedFile("Couldn't read any text from this file (a scanned/image-only PDF needs OCR). Paste the text instead.")
    return text


def package_docx(pkg: dict) -> bytes:
    """The approved application package (tailored bullets, cover note, interview prep) as a Word file."""
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    title = pkg.get("title", "Application")
    company = pkg.get("company") or ""
    doc.add_heading(f"{title}" + (f" at {company}" if company else ""), level=1)
    doc.add_paragraph(f"Fit {pkg.get('fit', '?')}/100 · ATS keyword match {pkg.get('ats_before', '?')} → {pkg.get('ats_after', '?')}/100")

    doc.add_heading("Cover note", level=2)
    for para in str(pkg.get("cover_note", "")).split("\n\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_heading("Tailored resume bullets", level=2)
    for b in pkg.get("resume", []):
        doc.add_paragraph(str(b), style="List Bullet")

    doc.add_heading("Interview prep", level=2)
    for q in pkg.get("interview_questions", []):
        doc.add_paragraph(str(q), style="List Number")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
