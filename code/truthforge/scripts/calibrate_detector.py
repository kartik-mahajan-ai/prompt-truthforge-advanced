"""Fit a temperature scaler so a detector's scores mean what the router thinks.

    python scripts/calibrate_detector.py --sample --detector minicheck
    python scripts/calibrate_detector.py --detector minicheck --split train --task QA

Why this is not optional: the rule router accepts above 0.70, rewrites below
0.30, and does nothing inside a band around 0.5. MiniCheck's raw output puts
*every* claim above 0.5 -- clearly-false ones land at 0.504. Feeding that to
the router means nothing is ever routed to REWRITE and the whole system
degenerates to `noop`, silently, while every metric still prints.

Fits on a held-out split and writes `runs/calibration_<detector>.json`, which
the CLI and UI load automatically.

Report both the before and after ECE in the paper. A detector whose scores were
uncalibrated at evaluation time is a detector whose router thresholds were
arbitrary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.config import Config  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth, load_sample  # noqa: E402
from truthforge.detect.calibration import (  # noqa: E402
    TemperatureScaler,
    reliability,
    select_threshold,
    spread,
)
from truthforge.detect.minicheck import build_detector  # noqa: E402


def collect(examples, detector, splitter):
    """Score every claim and pair it with its gold label.

    Gold: a claim is unsupported when its character span overlaps an annotated
    hallucination span. Claims without recoverable offsets are dropped rather
    than guessed at.
    """
    probs, labels = [], []
    for ex in examples:
        claims = [c for c in splitter.split(ex.response) if c.start is not None]
        if not claims:
            continue
        for claim, v in zip(claims, detector.check(claims, ex.passages)):
            probs.append(v.support_prob)
            labels.append(not ex.overlaps_span(claim.start, claim.end))
    return probs, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="train")
    ap.add_argument("--task", default="QA")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--detector", default="minicheck")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--objective", default="balanced_accuracy",
                    choices=["balanced_accuracy", "f1", "youden"])
    ap.add_argument("--min-precision", type=float, default=0.0,
                    help="reject thresholds below this precision; every flag "
                         "costs a correction attempt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.detector.name = args.detector
    cfg.detector.device = args.device
    # Fit on the RAW detector. build_detector applies any scaler already on
    # disk, so without this the script calibrates an already-calibrated
    # detector and each re-run compounds on the last.
    cfg.detector.calibration_path = ""

    examples = (
        load_sample()
        if args.sample
        else load_ragtruth(
            root=args.root,
            split=args.split,
            task_types={args.task} if args.task else None,
            limit=args.limit,
        )
    )
    print(f"{len(examples)} examples | detector={args.detector}")
    if args.sample:
        print("SYNTHETIC fixture -- this proves the machinery runs. A "
              "temperature fitted on 21 claims is NOT usable for results.")

    detector = build_detector(cfg)
    probs, labels = collect(examples, detector, RuleSplitter())
    print(f"{len(probs)} claims, {sum(labels)} supported "
          f"({sum(labels) / max(1, len(probs)):.1%})")

    if len(probs) < 20:
        print("\nnot enough claims to calibrate (need >= 20). Download "
              "RAGTruth: python scripts/download_ragtruth.py")
        return 1

    # Split by example, not by claim -- claims from one answer share evidence.
    idx = list(range(len(probs)))
    rng = random.Random(args.seed)
    rng.shuffle(idx)
    n_hold = int(len(idx) * args.holdout)
    hold, fit = set(idx[:n_hold]), idx[n_hold:]

    fit_probs = [probs[i] for i in fit]
    fit_labels = [labels[i] for i in fit]
    hold_probs = [probs[i] for i in sorted(hold)]
    hold_labels = [labels[i] for i in sorted(hold)]

    print("\nBEFORE calibration")
    _report(probs, labels)

    try:
        scaler = TemperatureScaler.fit(fit_probs, fit_labels)
    except ValueError as exc:
        print(f"\ncannot calibrate: {exc}")
        return 1

    print(f"\nfitted on {scaler.fitted_on} claims: "
          f"T={scaler.temperature:.3f}  bias={scaler.bias:+.3f}")
    print(f"  NLL {scaler.nll_before:.4f} -> {scaler.nll_after:.4f}")
    print(f"  ECE {scaler.ece_before:.4f} -> {scaler.ece_after:.4f}")

    print("\nAFTER calibration (all claims)")
    _report(scaler.transform(probs), labels)

    if hold_probs:
        print(f"\nHELD-OUT check ({len(hold_probs)} claims, never seen by the fit)")
        before = reliability(hold_probs, hold_labels)["ece"]
        after = reliability(scaler.transform(hold_probs), hold_labels)["ece"]
        acc_b = sum(1 for p, y in zip(hold_probs, hold_labels) if (p >= 0.5) == y) / len(hold_probs)
        acc_a = sum(
            1 for p, y in zip(scaler.transform(hold_probs), hold_labels) if (p >= 0.5) == y
        ) / len(hold_probs)
        print(f"  ECE      {before:.4f} -> {after:.4f}")
        print(f"  accuracy {acc_b:.3f} -> {acc_a:.3f}")
        if after > before:
            print("  WARNING: calibration made held-out ECE WORSE. The fitting "
                  "split is probably too small or unrepresentative. Do not "
                  "ship this scaler.")

    # Calibration makes the probabilities honest; it does NOT decide where to
    # flag. On a corpus that is ~90% supported, an honest probability is
    # usually high, so a 0.5 threshold flags nothing. Pick the operating point
    # from the same fitting split.
    print()
    print("OPERATING POINT")
    try:
        fit_cal = scaler.transform(fit_probs)
        chosen = select_threshold(
            fit_cal, [not y for y in fit_labels], objective=args.objective,
            min_precision=args.min_precision,
        )
        print(f"  threshold {chosen['threshold']:.4f} selected on the fitting split "
              f"({args.objective} {chosen['score']:.4f})")
        print(f"  flags {chosen['flag_rate']:.1%} of claims | "
              f"precision {chosen['precision']:.3f} | recall {chosen['recall']:.3f}")

        if hold_probs:
            hold_cal = scaler.transform(hold_probs)
            t = chosen["threshold"]
            hp = [not y for y in hold_labels]
            tp = sum(1 for p, y in zip(hold_cal, hp) if p < t and y)
            fp = sum(1 for p, y in zip(hold_cal, hp) if p < t and not y)
            fn = sum(1 for p, y in zip(hold_cal, hp) if p >= t and y)
            tn = sum(1 for p, y in zip(hold_cal, hp) if p >= t and not y)
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            print(f"  held out: balanced accuracy {(tpr + tnr) / 2:.4f} "
                  f"(vs {chosen['score']:.4f} on the fit -- a large gap means overfitting)")
    except ValueError as exc:
        print(f"  could not select a threshold: {exc}")
        chosen = None

    out = Path(args.out or f"runs/calibration_{args.detector}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **scaler.to_dict(),
        "detector": args.detector,
        "dataset": "sample" if args.sample else f"ragtruth:{args.split}:{args.task}",
        "n_claims_total": len(probs),
        "synthetic": bool(args.sample),
        "threshold": chosen,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if args.sample:
        print("marked synthetic=true -- the CLI will refuse to use it for a "
              "non-sample run.")
    return 0


def _report(probs, labels) -> None:
    rel = reliability(probs, labels)
    sp = spread(probs)
    acc = sum(1 for p, y in zip(probs, labels) if (p >= 0.5) == y) / len(probs)
    print(f"  ECE {rel['ece']:.4f} | accuracy@0.5 {acc:.3f} | "
          f"mean confidence {rel['mean_confidence']:.3f} vs base rate "
          f"{rel['base_rate']:.3f}")
    print(f"  spread {sp['min']:.3f}-{sp['max']:.3f} | "
          f"{sp['frac_saturated']:.0%} saturated | "
          f"{sp['frac_in_uncertain_band']:.0%} in the router's uncertain band | "
          f"{sp['frac_actionable_rewrite']:.0%} below the rewrite threshold")


if __name__ == "__main__":
    raise SystemExit(main())
