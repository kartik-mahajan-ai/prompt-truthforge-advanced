"""Train the learned router on auto-labelled claims.

    python scripts/build_router_labels.py --split train --out runs/router_labels.jsonl
    python scripts/train_router.py runs/router_labels.jsonl --out runs/router_policy.json

A multinomial logistic regression over ten features. Deliberately small: the
router's decision rule has to be inspectable, and with this many features
anything bigger memorises the dev set.

The printed coefficients are a result, not a diagnostic -- "the router learned
to skip low-coverage claims" is a sentence the paper needs, and this is where
it comes from.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.router.learned import FEATURE_ORDER, train_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", help="jsonl from build_router_labels.py")
    ap.add_argument("--out", default="runs/router_policy.json")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print("no labelled rows")
        return 1

    dist = Counter(r["label"] for r in rows)
    print(f"{len(rows)} rows, labels: {dict(dist)}")
    if len(dist) < 2:
        print("only one label present -- nothing to learn")
        return 1

    # Split by example_id, not by row: claims from the same answer share
    # evidence and leak across a random row split.
    groups = sorted({r.get("example_id") or r["claim_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_frac))
    val_groups = set(groups[:n_val])

    train = [r for r in rows if (r.get("example_id") or r["claim_id"]) not in val_groups]
    val = [r for r in rows if (r.get("example_id") or r["claim_id"]) in val_groups]
    print(f"train {len(train)} / val {len(val)} (split by example)")

    if not train:
        print("empty training split -- lower --val-frac")
        return 1

    policy = train_policy(
        [r["features"] for r in train],
        [r["label"] for r in train],
        C=args.C,
        seed=args.seed,
    )

    report = {"n_rows": len(rows), "label_counts": dict(dist),
              "n_examples": len(groups), "split": {}}
    for name, rows_ in (("train", train), ("val", val)):
        if not rows_:
            continue
        correct = sum(
            1
            for r in rows_
            if max(policy.predict_proba(r["features"]).items(), key=lambda kv: kv[1])[0]
            == r["label"]
        )
        majority = Counter(r["label"] for r in rows_).most_common(1)[0][1]
        report["split"][name] = {"n": len(rows_), "accuracy": round(correct / len(rows_), 4),
                                 "majority_baseline": round(majority / len(rows_), 4)}
        print(
            f"{name}: accuracy {correct / len(rows_):.3f} "
            f"(majority-class baseline {majority / len(rows_):.3f})"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    print(f"\nwrote {out}")

    # Training metrics beside the policy, so reports quote them from a file
    # rather than from a console log nobody kept.
    if "rewrite" in policy.action_order:
        j = policy.action_order.index("rewrite")
        report["rewrite_coefficients"] = {
            f: round(policy.weights[j][i], 3) for i, f in enumerate(FEATURE_ORDER)
        }
    out.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\ncoefficients (standardised; sign = direction, magnitude = strength):")
    width = max(len(f) for f in FEATURE_ORDER)
    print("  " + " " * width + "  " + "  ".join(f"{a:>12}" for a in policy.action_order))
    for i, feat in enumerate(FEATURE_ORDER):
        cells = "  ".join(f"{policy.weights[j][i]:>12.3f}" for j in range(len(policy.action_order)))
        print(f"  {feat.ljust(width)}  {cells}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
