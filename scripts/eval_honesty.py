"""Evaluate the honesty check (verify_claims) on hand-labelled rewrites.

    python scripts/eval_honesty.py            # prints metrics and every mistake
    python scripts/eval_honesty.py --json     # machine-readable (used by the test)

Each case in evals/honesty_cases.jsonl is an (original bullet, rewritten bullet) pair labelled
"honest" (only re-words what's there, or adds a JD term the original already evidences) or
"invented" (adds a skill, number or claim the candidate never made).

"Positive" = the checker reverts the rewrite. So:
  recall    = invented rewrites caught / all invented rewrites       (how much fabrication gets through)
  precision = invented among everything reverted                     (how often honest rewrites are wrongly blocked)
No LLM or network: verify_claims is deterministic.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobpilot.graph import verify_claims  # noqa: E402

CASES = ROOT / "evals" / "honesty_cases.jsonl"


def run() -> dict:
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    state = {"tailored": [{"original": c["original"], "rewritten": c["rewritten"], "skill": c["skill"], "status": "proposed"}
                          for c in cases], "trace": []}
    results = verify_claims(state)["tailored"]

    tp = fp = fn = tn = 0
    mistakes = []
    for case, res in zip(cases, results):
        reverted = res["status"] == "reverted"
        invented = case["label"] == "invented"
        if invented and reverted:
            tp += 1
        elif invented and not reverted:
            fn += 1
            mistakes.append({"id": case["id"], "error": "missed (invented claim accepted)", "rewritten": case["rewritten"]})
        elif reverted:
            fp += 1
            mistakes.append({"id": case["id"], "error": "false alarm (honest rewrite reverted)",
                             "rewritten": case["rewritten"], "reason": res.get("reason", "")})
        else:
            tn += 1
    return {
        "cases": len(cases), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": round(tp / max(tp + fn, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "accuracy": round((tp + tn) / max(len(cases), 1), 3),
        "mistakes": mistakes,
    }


if __name__ == "__main__":
    r = run()
    if "--json" in sys.argv:
        print(json.dumps(r, indent=2))
        sys.exit(0)
    print(f"Honesty check on {r['cases']} labelled rewrites")
    print(f"  caught invented claims (recall) : {r['tp']}/{r['tp'] + r['fn']} = {r['recall']:.0%}")
    print(f"  reverts that were right (precision): {r['tp']}/{r['tp'] + r['fp']} = {r['precision']:.0%}")
    print(f"  overall accuracy                : {r['accuracy']:.0%}")
    for m in r["mistakes"]:
        print(f"  - {m['id']}: {m['error']}: {m['rewritten']}" + (f"  [{m['reason']}]" if m.get("reason") else ""))
