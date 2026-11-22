"""Pick the answerability cut point on a question set we will NOT report on.

    python scripts/select_answerability_threshold.py

The signal (`question_coverage`) separates answerable from unanswerable at
AUROC 0.83 on the open-corpus questions. Choosing its threshold on those same
questions and then reporting refusal rates over them would be fitting the
operating point to the test set -- the exact mistake that made the detector's
0.5 threshold meaningless until thresholds were moved to the train split.

So this builds a DEVELOPMENT set from a different seed and a disjoint slice of
RAGTruth, selects there, and prints the number to put in the config. The
evaluation set is never touched.

The objective is stated up front and not changed afterwards: maximise
(correct refusals) - (over refusals), i.e. treat one needless refusal as
costing exactly one correct refusal. That is a choice, not a fact; it is the
neutral one, and the full curve is printed so a different trade can be read
off it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.detect.baselines import question_coverage  # noqa: E402
from truthforge.retrieve.base import BM25Index  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "eoc", ROOT / "scripts" / "eval_open_corpus.py"
)
eoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eoc)

EVAL_SEED = 20260920          # the seed the reported run used -- avoid it
DEV_SEED = 777                # development only


def build(seed: int, split: str, limit: int, n_ans: int, n_una: int):
    examples = load_ragtruth(
        root=str(ROOT / "data/raw/ragtruth"), split=split,
        task_types={"QA"}, limit=limit,
    )
    texts, ids, questions = eoc.build_corpus(examples, n_ans, n_una, seed)
    bm = BM25Index(texts, ids)
    rows = []
    for q in questions:
        hits = bm.retrieve(q["question"], k=5)
        rows.append({
            "answerable": q["answerable"],
            "qc": question_coverage(q["question"], hits),
        })
    return rows, len(texts)


def sweep(rows):
    out = []
    for i in range(0, 21):
        tau = i / 20
        # refuse when question coverage is BELOW tau
        correct = [r for r in rows if not r["answerable"] and r["qc"] < tau]
        over = [r for r in rows if r["answerable"] and r["qc"] < tau]
        n_una = sum(1 for r in rows if not r["answerable"])
        n_ans = sum(1 for r in rows if r["answerable"])
        cr = len(correct) / n_una if n_una else 0.0
        orf = len(over) / n_ans if n_ans else 0.0
        out.append({"tau": round(tau, 2), "correct_refusal": round(cr, 3),
                    "over_refusal": round(orf, 3), "objective": round(cr - orf, 3)})
    return out


def main() -> int:
    # Development slice: different seed AND a different draw from the corpus.
    rows, n_passages = build(DEV_SEED, "train", 400, 40, 40)
    print(f"development set: {len(rows)} questions over {n_passages} passages "
          f"(split=train, seed={DEV_SEED})")
    print(f"the reported evaluation uses split=test, seed={EVAL_SEED} -- disjoint\n")

    table = sweep(rows)
    head = f"{'tau':>6}{'correct_ref':>13}{'over_ref':>11}{'objective':>12}"
    print(head)
    print("-" * len(head))
    for r in table:
        mark = ""
        print(f"{r['tau']:>6}{r['correct_refusal']:>13}{r['over_refusal']:>11}"
              f"{r['objective']:>12}{mark}")

    best = max(table, key=lambda r: (r["objective"], -r["tau"]))
    print(f"\nselected tau = {best['tau']}  "
          f"(correct {best['correct_refusal']}, over {best['over_refusal']}, "
          f"objective {best['objective']})")
    print("\nPut this in RouterConfig.min_question_coverage. It was chosen on the")
    print("development set above and must not be re-tuned after seeing the")
    print("evaluation numbers -- that is what the DECISIONS.md change log is for.")

    out = ROOT / "runs" / "answerability_threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dev_split": "train", "dev_seed": DEV_SEED, "eval_seed": EVAL_SEED,
        "n_questions": len(rows), "n_passages": n_passages,
        "objective": "correct_refusal - over_refusal",
        "selected_tau": best["tau"], "sweep": table,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
