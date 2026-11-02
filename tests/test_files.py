"""Resume upload (PDF / DOCX / TXT) and .docx package download. Files are built in memory, no fixtures."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = ""; os.environ["TAVILY_API_KEY"] = ""

from docx import Document
from fastapi.testclient import TestClient

from app import app
from jobpilot.files import normalise
from jobpilot.retriever import split_bullets

c = TestClient(app)
LONG = "Built a customer-support chatbot on Azure OpenAI GPT-4 with a retrieval pipeline over 1,200 policy PDFs"


def make_pdf(lines):
    """Smallest valid PDF with one text line per entry (Helvetica + WinAnsi, i.e. Windows-1252: • is byte 0x95)."""
    ops = ["BT /F1 11 Tf 50 750 Td 14 TL"]
    for ln in lines:
        esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({esc}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("cp1252")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell()); out.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets))
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref))
    return out.getvalue()


def upload(name, data):
    return c.post("/api/resume-file", files={"file": (name, data)})


def test_pdf_upload_joins_wrapped_bullets():
    pdf = make_pdf(["Aarav Shah", "Experience", "• Built a customer-support chatbot on Azure OpenAI GPT-4 with a retrieval",
                    "pipeline over 1,200 policy PDFs", "• Wrote FastAPI endpoints and containerised the service with Docker"])
    r = upload("resume.pdf", pdf)
    assert r.status_code == 200, r.text
    bullets = split_bullets(r.json()["text"])
    assert LONG in bullets                      # the wrapped bullet came back as one line
    assert any(b.startswith("Wrote FastAPI") for b in bullets)


def test_docx_upload_keeps_list_bullets():
    doc = Document()
    doc.add_paragraph("Aarav Shah")
    doc.add_paragraph(LONG, style="List Bullet")
    doc.add_paragraph("Wrote FastAPI endpoints and containerised the service with Docker", style="List Bullet")
    doc.add_paragraph("Built an MCP server exposing internal tools to the assistant", style="List Bullet")
    buf = io.BytesIO(); doc.save(buf)
    r = upload("resume.docx", buf.getvalue())
    assert r.status_code == 200, r.text
    assert LONG in split_bullets(r.json()["text"])


def test_upload_rejects_bad_files():
    assert upload("resume.png", b"\x89PNG....").status_code == 422
    assert upload("resume.pdf", b"not really a pdf").status_code == 422
    assert upload("resume.txt", b"x" * (2 * 1024 * 1024 + 10)).status_code == 422
    assert "OCR" in upload("scan.pdf", make_pdf([])).json()["detail"]      # image-only PDF: no text


def test_normalise_bullet_glyphs():
    assert normalise("▪ Led a team\n● Shipped it\n  - Kept it") == "- Led a team\n- Shipped it\n- Kept it"


def test_package_docx_download():
    pkg = {"title": "AI Engineer", "company": "Razorpay", "fit": 75, "ats_before": 24, "ats_after": 65,
           "resume": ["Built a chatbot (LLMs)"], "cover_note": "Dear team,\n\nHello.", "interview_questions": ["Why RAG?"]}
    r = c.post("/api/package.docx", json=pkg)
    assert r.status_code == 200 and 'filename="AI-Engineer-Razorpay.docx"' in r.headers["content-disposition"]
    text = "\n".join(p.text for p in Document(io.BytesIO(r.content)).paragraphs)
    assert "AI Engineer at Razorpay" in text and "Built a chatbot (LLMs)" in text and "Why RAG?" in text
