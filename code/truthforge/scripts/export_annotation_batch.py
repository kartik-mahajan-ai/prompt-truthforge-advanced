"""Sample ~300 correction attempts and write the annotation worksheet.

    python scripts/export_annotation_batch.py runs/always --n 300

Sampling is stratified by (action, heuristic outcome) so that rare cells --
especially WORSE, which is the whole point -- are represented, and seeded so
the batch is reproducible. Stratification means the raw label distribution in
the sample is NOT the population distribution; reweight before quoting a
worse-rate from human labels alone, or report it as "worse-rate among sampled
attempts" and say how they were sampled.

Writes two identical blind worksheets, one per annotator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.eval.outcomes import export_for_annotation  # noqa: E402
from truthforge.types import Action, CorrectionAttempt, Outcome, Passage  # noqa: E402


def load_attempts(run_dir: Path):
    attempts, passages = [], {}
    for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        turn = json.loads(line)
        ps = [
            Passage(doc_id=p["doc_id"], text=p["text"], source=p.get("source", "given"))
            for p in turn.get("passages", [])
        ]
        for a in turn.get("attempts", []):
            attempts.append(
                CorrectionAttempt(
                    claim_id=a["claim_id"],
                    action=Action(a["action"]),
                    iteration=a["iteration"],
                    before_text=a["before_text"],
                    after_text=a["after_text"],
                    before_support_prob=a["before_support_prob"],
                    after_support_prob=a.get("after_support_prob"),
                    outcome=Outcome(a["outcome"]) if a.get("outcome") else None,
                )
            )
            passages[a["claim_id"]] = ps
    return attempts, passages


def stratified_sample(attempts, n: int, seed: int):
    strata = defaultdict(list)
    for a in attempts:
        strata[(a.action.value, a.outcome.value if a.outcome else "unlabelled")].append(a)

    rng = random.Random(seed)
    for cell in strata.values():
        rng.shuffle(cell)

    # Round-robin across cells so small strata are not swamped by large ones.
    picked, keys = [], sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    rng.shuffle(picked)  # presentation order must not encode the stratum
    return picked, {k: len(v) for k, v in strata.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out-dir", default="annotation")
    ap.add_argument("--annotators", nargs="*", default=["A", "B"])
    args = ap.parse_args()

    attempts, passages = load_attempts(Path(args.run_dir))
    print(f"{len(attempts)} attempts in {args.run_dir}")
    if not attempts:
        print("nothing to annotate")
        return 1

    picked, remaining = stratified_sample(attempts, args.n, args.seed)
    print(f"sampled {len(picked)} (stratified by action x heuristic outcome)")

    for who in args.annotators:
        path = Path(args.out_dir) / f"labels_{who}.csv"
        if path.exists():
            print(f"REFUSING to overwrite existing {path}")
            continue
        n = export_for_annotation(picked, passages, path, blind=True)
        print(f"wrote {n} blind rows to {path}")

    manifest = Path(args.out_dir) / "batch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_dir": args.run_dir,
                "n_requested": args.n,
                "n_sampled": len(picked),
                "seed": args.seed,
                "stratum_sizes_before_sampling": {
                    "|".join(k): v for k, v in remaining.items()
                },
                "note": "stratified sample -- reweight before quoting a "
                        "population worse-rate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    print("\nNext: both annotators read annotation/GUIDELINES.md, label the "
          "first 30 rows, compare, THEN do the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
