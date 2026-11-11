"""End-to-end evaluation of JobPilot on hand-labelled job descriptions and resumes.

    python scripts/eval.py            # uses the LLM if GROQ_API_KEY is set, else the built-in rules
    python scripts/eval.py --rules    # force rules mode

Writes evals/report_<mode>.json (served by GET /api/eval and shown in the UI's Evaluation panel).

evals/fit_cases.json: 8 JDs x 4 fictional resumes = 18 pairs (+ a 6-pair held-out block written after the fixes), labelled by hand before running the agent with
  * each JD's must-have skills            -> did parse_jd find the requirements?
  * each resume's real skills             -> did score_fit mark the right skills as matched?
  * should this candidate apply (yes/no)  -> did the agent recommend applying, or send a gap plan?

Task success for one pair:
  should apply     -> agent recommends applying, the approved package is produced, the tailored resume contains
                      no skill the original resume didn't have, ATS score didn't go down, and the cover note names
                      the role and the company
  should not apply -> agent stops early with a gap plan

Company research is replaced with an offline stub so the eval is deterministic and never calls the network.
The honesty check has its own eval: scripts/eval_honesty.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pct(a, b):
    return round(100 * a / b, 1) if b else None


class OfflineResearch:
    """Stands in for the company_research tool: same shape, no network."""

    def invoke(self, args):
        return {"name": args["company"], "summary": "", "source": "none", "links": []}


def run(force_rules: bool = False) -> dict:
    if force_rules:
        os.environ["GROQ_API_KEY"] = ""
    from jobpilot import graph as G
    from jobpilot import skills as SK
    from jobpilot.llm import model_name

    G.company_research = OfflineResearch()
    data = json.loads((ROOT / "evals" / "fit_cases.json").read_text(encoding="utf-8"))
    jds, resumes = data["jds"], data["resumes"]

    # 1. Requirement extraction, once per JD.
    jd_rows = []
    for name, jd in jds.items():
        job = G.parse_jd({"jd_text": jd["text"], "company": jd["company"], "trace": []})["job"]
        got = {SK.canonical(x) for x in job["must_have"]}
        gold = set(jd["must_have"])
        jd_rows.append({"jd": name, "tp": len(got & gold), "fp": sorted(got - gold), "fn": sorted(gold - got)})

    # 2. Full agent run per (JD, resume) pair.
    rows = run_pairs(G, SK, jds, resumes, data["pairs"])
    hold = data.get("holdout")
    hrows = run_pairs(G, SK, {**jds, **hold["jds"]}, {**resumes, **hold["resumes"]}, hold["pairs"]) if hold else []
    return report(rows, jd_rows, hrows, model_name(), len(jds), len(resumes))


def run_pairs(G, SK, jds, resumes, pairs) -> list[dict]:
    rows = []
    for p in pairs:
        jd, cv = jds[p["jd"]], resumes[p["resume"]]
        t0 = time.perf_counter()
        out = G.start(jd["text"], cv["text"], jd["company"])
        st = out["state"]
        applied = out["paused_at"] == "human_review"
        pkg, fabricated, note_ok = None, [], None
        if applied:
            pkg = G.resume(st, "approve")["state"].get("package")
            before = set(SK.find_skills(cv["text"]))
            after = set(SK.find_skills("\n".join(pkg["resume"]))) if pkg else set()
            fabricated = sorted(after - before)
            note = (pkg or {}).get("cover_note", "").lower()
            note_ok = bool(pkg) and jd["company"].lower() in note and st["job"]["title"].split()[0].lower() in note
        ms = (time.perf_counter() - t0) * 1000

        gold_match = set(jd["must_have"]) & set(cv["skills"])
        got_match = {SK.canonical(k) for k, v in st["fit"]["must"].items() if v == "matched"}
        hist = st["ats"]["history"]
        rows.append({"id": p["id"], "jd": p["jd"], "resume": p["resume"], "should_apply": p["should_apply"],
                     "applied": applied, "fit": st["fit"]["score"], "ats_before": hist[0], "ats_after": hist[-1],
                     "rounds": st.get("attempts", 0), "package": bool(pkg), "fabricated": fabricated, "note_ok": note_ok,
                     "match_tp": len(got_match & gold_match), "match_fp": sorted(got_match - gold_match),
                     "match_fn": sorted(gold_match - got_match), "ms": ms})
    return rows


def success(r) -> bool:
    if not r["should_apply"]:
        return not r["applied"]
    return bool(r["applied"] and r["package"] and not r["fabricated"] and r["ats_after"] >= r["ats_before"] and r["note_ok"])


def failed(rows) -> list[dict]:
    return [{"id": r["id"], "jd": r["jd"], "resume": r["resume"], "should_apply": r["should_apply"],
             "applied": r["applied"], "fit": r["fit"], "fabricated": r["fabricated"], "note_ok": r["note_ok"]}
            for r in rows if not success(r)]


def report(rows, jd_rows, hrows, mode, n_jds, n_resumes) -> dict:
    n = len(rows)
    tp = sum(r["tp"] for r in jd_rows); fp = sum(len(r["fp"]) for r in jd_rows); fn = sum(len(r["fn"]) for r in jd_rows)
    mtp = sum(r["match_tp"] for r in rows); mfp = sum(len(r["match_fp"]) for r in rows); mfn = sum(len(r["match_fn"]) for r in rows)
    applied = [r for r in rows if r["applied"]]
    lifts = [r["ats_after"] - r["ats_before"] for r in applied]

    return {
        "mode": mode, "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "jds": n_jds, "resumes": n_resumes, "pairs": n,
        "metrics": {
            "task_success": pct(sum(success(r) for r in rows), n),
            "recommendation_accuracy": pct(sum(r["applied"] == r["should_apply"] for r in rows), n),
            "good_matches_recommended": pct(sum(r["applied"] for r in rows if r["should_apply"]), sum(r["should_apply"] for r in rows)),
            "poor_matches_stopped": pct(sum(not r["applied"] for r in rows if not r["should_apply"]), sum(not r["should_apply"] for r in rows)),
            "requirement_precision": pct(tp, tp + fp),
            "requirement_recall": pct(tp, tp + fn),
            "skill_match_precision": pct(mtp, mtp + mfp),
            "skill_match_recall": pct(mtp, mtp + mfn),
            "fabricated_skills_in_packages": sum(len(r["fabricated"]) for r in rows),
            "avg_ats_lift_points": round(sum(lifts) / len(lifts), 1) if lifts else None,
            "cover_notes_name_role_and_company": pct(sum(bool(r["note_ok"]) for r in applied), len(applied)),
            "avg_ms_per_pair": round(sum(r["ms"] for r in rows) / n),
        },
        "pairs_detail": [{k: r[k] for k in ("id", "jd", "resume", "should_apply", "applied", "fit", "ats_before", "ats_after", "rounds")}
                         for r in rows],
        "failed_pairs": failed(rows),
        # Written after the rule fixes and never tuned on: the honest estimate of how the rules generalise.
        "holdout": {"pairs": len(hrows), "task_success": pct(sum(success(r) for r in hrows), len(hrows)),
                    "recommendation_accuracy": pct(sum(r["applied"] == r["should_apply"] for r in hrows), len(hrows)),
                    "fabricated_skills_in_packages": sum(len(r["fabricated"]) for r in hrows),
                    "failed_pairs": failed(hrows)} if hrows else None,
        "requirement_misses": [{"jd": r["jd"], "extra": r["fp"], "missed": r["fn"]} for r in jd_rows if r["fp"] or r["fn"]],
        "skill_match_misses": [{"id": r["id"], "wrongly_matched": r["match_fp"], "missed": r["match_fn"]}
                               for r in rows if r["match_fp"] or r["match_fn"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="force rule-based mode")
    ap.add_argument("--out", help="report file name (default report_<mode>.json)")
    args = ap.parse_args()
    rep = run(args.rules)
    name = args.out or f"report_{'rules' if rep['mode'] == 'built-in rules' else 'llm'}.json"
    (ROOT / "evals" / name).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"JobPilot eval · {rep['mode']} · {rep['jds']} JDs x {rep['resumes']} resumes = {rep['pairs']} pairs")
    for k, v in rep["metrics"].items():
        print(f"  {k:36} {v}")
    for f in rep["failed_pairs"]:
        print(f"  FAIL {f['id']} {f['jd']} x {f['resume']}: should_apply={f['should_apply']} applied={f['applied']} "
              f"fit={f['fit']} fabricated={f['fabricated']} note_ok={f['note_ok']}")
    h = rep.get("holdout")
    if h:
        print(f"  held-out ({h['pairs']} pairs, never tuned on): task_success {h['task_success']}, "
              f"recommendation {h['recommendation_accuracy']}, fabricated {h['fabricated_skills_in_packages']}")
        for f in h["failed_pairs"]:
            print(f"  HELD-OUT FAIL {f['id']} {f['jd']} x {f['resume']}: should_apply={f['should_apply']} applied={f['applied']} fit={f['fit']}")
    for m in rep["requirement_misses"]:
        print(f"  JD {m['jd']}: extra {m['extra']} missed {m['missed']}")
    for m in rep["skill_match_misses"]:
        print(f"  MATCH {m['id']}: wrongly matched {m['wrongly_matched']} missed {m['missed']}")
    print(f"wrote evals/{name}")


if __name__ == "__main__":
    main()
