"""Inter-annotator agreement on the correction-outcome labels.

    python scripts/agreement.py annotation/labels_A.csv annotation/labels_B.csv
    python scripts/agreement.py labels_A.csv labels_B.csv --run runs/always

With `--run`, the heuristic classifier's labels for the same attempts are
loaded from that run and compared against each annotator separately. A
heuristic that agrees with one annotator much more than the other is fitted to
a style, not validated.

Both kappa values go in the paper whatever they are. Run this BEFORE
adjudicating anything -- agreement computed after adjudication is meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.eval.outcomes import agreement_report, load_human_labels  # noqa: E402


def heuristic_labels_from_run(run_dir: str | Path) -> dict[str, str]:
    """attempt_key -> heuristic outcome, read from a run's turns.jsonl."""
    path = Path(run_dir) / "turns.jsonl"
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for a in json.loads(line).get("attempts", []):
            if a.get("outcome"):
                key = f"{a['claim_id']}:{a['iteration']}:{a['action']}"
                out[key] = a["outcome"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_a")
    ap.add_argument("labels_b")
    ap.add_argument("--run", default=None, help="run dir with heuristic labels")
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    args = ap.parse_args()

    a = load_human_labels(args.labels_a)
    b = load_human_labels(args.labels_b)
    print(f"annotator A: {len(a)} labelled rows")
    print(f"annotator B: {len(b)} labelled rows")

    heuristic = heuristic_labels_from_run(args.run) if args.run else None
    report = agreement_report(a, b, heuristic)
    print(json.dumps(report, indent=2))

    kappa = report.get("outcome_kappa")
    if isinstance(kappa, float):
        if kappa < 0.6:
            print("\nkappa < 0.6: the CODEBOOK is unclear, not an annotator. "
                  "Amend annotation/GUIDELINES.md and re-calibrate.")
        elif kappa < 0.8:
            print("\nkappa in [0.6, 0.8): substantial. Report it as-is.")
        else:
            print("\nkappa >= 0.8: strong agreement.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
