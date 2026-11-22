"""GATE 1: does the detector beat a trivial baseline?

From DECISIONS.md: if it does not, the claim splitter is broken and building
the loop is wasted work. This script is the test.

It splits each RAGTruth response into claims, marks a claim gold-unsupported
when its character span overlaps an annotated hallucination span, then scores
every detector on the same claims.

    # runs anywhere, no GPU, no downloads
    python scripts/gate1_detector_vs_baseline.py --sample

    # the real thing
    python scripts/gate1_detector_vs_baseline.py --split test --task QA --limit 500
    python scripts/gate1_detector_vs_baseline.py --split test --detectors minicheck

Caveat that must travel with every number this prints: RAGTruth's original
spans under-annotate, so recall here is a LOWER bound and some "false
positives" are real catches the release missed. See RAGTRUTH_SPAN_CAVEAT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.data.ragtruth import (  # noqa: E402
    RAGTRUTH_SPAN_CAVEAT,
    load_ragtruth,
    load_sample,
)
from truthforge.detect.baselines import BASELINES  # noqa: E402
from truthforge.detect.calibration import select_threshold  # noqa: E402
from truthforge.types import Label  # noqa: E402
from truthforge.eval.metrics import (  # noqa: E402
    auroc,
    compare_to_baselines,
    detector_metrics,
    format_table,
)

DEFAULT_DETECTORS = ["always_supported", "always_unsupported", "random", "lexical"]


def build_claim_set(examples, splitter, require_offsets: bool = True):
    """Split every response and attach gold labels from the annotated spans.

    Claims with no recoverable character offsets are dropped rather than
    guessed at -- a mislabelled gold set would make Gate 1 meaningless.
    """
    rows = []
    dropped = 0
    for ex in examples:
        for claim in splitter.split(ex.response):
            if claim.start is None or claim.end is None:
                dropped += 1
                if require_offsets:
                    continue
            gold = (
                ex.overlaps_span(claim.start, claim.end)
                if claim.start is not None
                else False
            )
            rows.append({"example": ex, "claim": claim, "gold_unsupported": gold})
    return rows, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="use the synthetic fixture")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default=None, help="e.g. QA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--detectors",
        nargs="*",
        default=DEFAULT_DETECTORS,
        help=f"any of {sorted(BASELINES)} plus 'minicheck'",
    )
    ap.add_argument("--primary", default="lexical", help="detector to gate on")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tune-split", default="train",
                    help="split used to select each detector's threshold; "
                         "'' disables tuning and uses --threshold for all")
    ap.add_argument("--tune-limit", type=int, default=250)
    ap.add_argument("--objective", default="balanced_accuracy",
                    choices=["balanced_accuracy", "f1", "youden"])
    args = ap.parse_args()

    if args.sample:
        examples = load_sample()
        print(f"loaded {len(examples)} SYNTHETIC examples -- results are a smoke "
              f"test, not a measurement\n")
    else:
        examples = load_ragtruth(
            root=args.root,
            split=args.split,
            task_types={args.task} if args.task else None,
            limit=args.limit,
        )
        print(f"loaded {len(examples)} RAGTruth examples "
              f"(split={args.split}, task={args.task or 'all'})\n")

    rows, dropped = build_claim_set(examples, RuleSplitter())
    if not rows:
        print("no claims produced -- the splitter is the problem, which IS the "
              "Gate 1 answer")
        return 1

    gold = [r["gold_unsupported"] for r in rows]
    n_pos = sum(gold)
    print(f"{len(rows)} claims, {n_pos} gold-unsupported "
          f"({n_pos / len(rows):.1%}), {dropped} dropped for missing offsets")
    print(f"majority class = {'unsupported' if n_pos * 2 > len(rows) else 'supported'}\n")

    # Tuning split. Comparing a detector at a threshold chosen for it against
    # a baseline stuck at 0.5 would be rigged, so every detector gets its
    # operating point selected on the SAME held-out tuning data, and all are
    # then evaluated on the test rows.
    tune_rows = []
    if args.tune_split and not args.sample:
        tune_examples = load_ragtruth(
            root=args.root,
            split=args.tune_split,
            task_types={args.task} if args.task else None,
            limit=args.tune_limit,
        )
        tune_rows, _ = build_claim_set(tune_examples, RuleSplitter())
        print(f"tuning thresholds on {len(tune_rows)} claims from "
              f"split={args.tune_split} (never evaluated on)")
        print()

    def score_all(det, rs):
        out = []
        for r in rs:
            out.append(det.check([r["claim"]], r["example"].passages)[0].support_prob)
        return out

    results, table = {}, []
    for name in args.detectors:
        det = _build(name, args)
        if det is None:
            continue

        threshold, how = args.threshold, "default"
        if tune_rows:
            try:
                tp = score_all(det, tune_rows)
                chosen = select_threshold(
                    tp, [r["gold_unsupported"] for r in tune_rows],
                    objective=args.objective,
                )
                threshold, how = chosen["threshold"], f"tuned ({args.objective})"
            except ValueError as exc:
                print(f"  {name}: keeping default threshold -- {exc}")

        probs = score_all(det, rows)
        preds = [
            Label.UNSUPPORTED if p < threshold else Label.SUPPORTED for p in probs
        ]
        scores = [1.0 - p for p in probs]  # P(unsupported)

        m = detector_metrics(preds, gold)
        results[name] = m
        table.append({
            "detector": name,
            "threshold": round(threshold, 4),
            "how": how,
            **m.to_dict(),
            "auroc": round(auroc(scores, gold), 4),
        })

    print(format_table(table))
    print()

    if args.primary in results:
        verdict = compare_to_baselines(results, primary=args.primary)
        print(json.dumps(verdict, indent=2))
        status = "PASS" if verdict["passes_gate_1"] else "FAIL"
        print(f"\nGATE 1: {status}")
        if not verdict["passes_gate_1"]:
            print("  -> Stop. Fix the claim splitter before building the loop.")
    else:
        verdict = {"note": f"primary {args.primary!r} not among detectors run"}
        print(verdict["note"])

    print(f"\nCAVEAT: {RAGTRUTH_SPAN_CAVEAT}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "n_claims": len(rows),
                    "n_gold_unsupported": n_pos,
                    "detectors": table,
                    "gate_1": verdict,
                    "caveat": RAGTRUTH_SPAN_CAVEAT,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    return 0


def _build(name: str, args):
    """Resolve a detector name.

    Routed through the shared `build_detector` rather than a local if-chain,
    so this script cannot drift out of sync with the detector registry. It
    did: `nli` was silently skipped and `minicheck` was called with a kwarg
    that no longer exists, so a Gate 1 run quietly compared baselines only.
    """
    from truthforge.config import Config
    from truthforge.detect.minicheck import build_detector

    cfg = Config()
    cfg.detector.name = name
    cfg.detector.flag_threshold = args.threshold
    cfg.detector.device = args.device
    # Gate 1 asks whether the RAW detector separates the classes. Applying a
    # fitted scaler here would be circular: the scaler is fitted on labels
    # from this same corpus.
    cfg.detector.calibration_path = ""

    try:
        return build_detector(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"skipping {name}: {type(exc).__name__}: {exc}")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
