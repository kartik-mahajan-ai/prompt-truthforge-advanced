"""Does requiring two verifiers to agree make a flag worth acting on?

    python scripts/verifier_agreement.py --split train --limit 300 --out runs/agreement_train.json

Every abstention TruthForge makes inherits the precision of the flag behind
it, and on RAGTruth QA both verifiers are right about one flag in five. The
NLI verifier (DeBERTa-v3) and MiniCheck (Flan-T5) are different model families
trained on different data, so their mistakes need not coincide. If they do
not, a claim both of them flag is more likely to be a real error than a claim
either flags alone.

This measures that on a split chosen for SELECTION -- run it on train, decide
whether agreement is worth using, and only then look at test. Each verifier is
used at the operating point the system actually runs it at: NLI flags below
0.5, MiniCheck at its calibrated threshold selected on the training split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.config import Config  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.detect.minicheck import build_detector  # noqa: E402


def flags_for(name: str, examples, device: str):
    cfg = Config()
    cfg.detector.name = name
    cfg.detector.device = device
    det = build_detector(cfg)
    out, probs = [], []
    for ex, claims in examples:
        if not claims:
            continue
        for v in det.check(claims, ex.passages):
            out.append(v.is_flagged)
            probs.append(v.support_prob)
    thr = getattr(det, "flag_threshold", None)
    return out, probs, thr


def stats(flag, gold_unsup):
    tp = sum(1 for f, g in zip(flag, gold_unsup) if f and g)
    fp = sum(1 for f, g in zip(flag, gold_unsup) if f and not g)
    fn = sum(1 for f, g in zip(flag, gold_unsup) if not f and g)
    n_flag = tp + fp
    return {"flagged": n_flag, "tp": tp, "fp": fp,
            "precision": round(tp / n_flag, 4) if n_flag else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/agreement_train.json")
    args = ap.parse_args()

    exs = load_ragtruth(root=str(ROOT / "data/raw/ragtruth"), split=args.split,
                        task_types={"QA"}, limit=args.limit)
    splitter = RuleSplitter()
    examples, gold = [], []
    for ex in exs:
        claims = splitter.split(ex.response)
        examples.append((ex, claims))
        gold += [ex.overlaps_span(c.start, c.end) for c in claims]
    print(f"{args.split}: {len(exs)} responses, {len(gold)} claims, {sum(gold)} gold-unsupported")

    nli, nli_p, nli_t = flags_for("nli", examples, args.device)
    mc, mc_p, mc_t = flags_for("minicheck", examples, args.device)
    assert len(nli) == len(mc) == len(gold), "verifiers scored different claim sets"

    rows = {
        "nli": stats(nli, gold),
        "minicheck": stats(mc, gold),
        "both (AND)": stats([a and b for a, b in zip(nli, mc)], gold),
        "either (OR)": stats([a or b for a, b in zip(nli, mc)], gold),
    }
    # how often do they disagree, and is agreement better than chance overlap?
    both = sum(1 for a, b in zip(nli, mc) if a and b)
    exp = (sum(nli) / len(nli)) * (sum(mc) / len(mc)) * len(nli)
    print(f"\nthresholds: nli {nli_t}  minicheck {mc_t}")
    print(f"{'rule':14s}{'flagged':>9}{'precision':>11}{'recall':>9}")
    for k, r in rows.items():
        print(f"{k:14s}{r['flagged']:>9}{str(r['precision']):>11}{str(r['recall']):>9}")
    print(f"\nflagged by both: {both}  (expected if independent: {exp:.0f})")

    Path(ROOT / args.out).write_text(json.dumps({
        "split": args.split, "limit": args.limit, "n_claims": len(gold),
        "n_unsupported": int(sum(gold)), "thresholds": {"nli": nli_t, "minicheck": mc_t},
        "rules": rows, "both_flagged": both, "both_expected_if_independent": round(exp, 1),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
