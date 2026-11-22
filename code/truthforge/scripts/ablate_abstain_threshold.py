"""What does the abstain threshold buy, and what does it cost?

    python scripts/ablate_abstain_threshold.py runs/rt_rule
    python scripts/ablate_abstain_threshold.py runs/rt_rule --json runs/ablation.json

**The finding this exists to quantify.** On 60 RAGTruth QA responses the NLI
detector flags 81 of 392 claims. RAGTruth's spans mark 12 as unsupported. The
detector catches 10 of those 12 -- recall 0.83 -- at precision 0.123. The rule
router then abstains on any flagged claim its corrector could not fix, which
discarded 63 claims: 8 genuinely unsupported and 55 that were fine.

That is not a bug in the router. It is the arithmetic of acting on a flag whose
precision is 0.12. Abstention is a sharp instrument and the detector is not
sharp enough to aim it, so the useful question is not "should we abstain?" but
"how confident must the flag be before abstaining pays?"

**Why this can be swept offline.** ABSTAIN is terminal and local: dropping a
claim does not change the text of any other claim, does not consume a corrector
call, and does not alter what the detector saw. So for any threshold tau, the
set of claims a tau-router would have dropped is exactly
`{claims it flagged with support_prob < tau}`, recoverable from the log. This
is off-policy evaluation over a decision whose outcome is fully observed, which
is the only kind that needs no new run.

**What it CANNOT tell you**, stated because the temptation is obvious: it says
nothing about REWRITE at any tau. A rewrite changes the answer, and the only
way to know what a different rewrite would have produced is to run it. Every
number here is about the abstain/keep decision alone, and the script refuses to
print a "best tau" -- the trade-off is the result, and which end of it to want
is a claim about the application, not about the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.cli import _turn_from_dict  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.types import Label  # noqa: E402


def load_claims(run_dir: Path, root: str, split: str, task: str) -> list[dict]:
    """One row per claim: its first support_prob, whether it was flagged, gold."""
    exs = {
        e.example_id: e
        for e in load_ragtruth(root=root, split=split, task_types={task})
    }
    rows = []
    for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = _turn_from_dict(json.loads(line))
        ex = exs.get(t.example_id)
        if ex is None:
            continue
        gold = {
            c.claim_id: not ex.overlaps_span(c.start, c.end)
            for c in t.claims
            if c.start is not None
        }
        seen = set()
        for v in t.verdicts:
            if v.claim_id in seen or v.claim_id not in gold:
                continue
            seen.add(v.claim_id)
            rows.append({
                "claim_id": v.claim_id,
                "support_prob": v.support_prob,
                "flagged": v.label is Label.UNSUPPORTED,
                "gold_supported": gold[v.claim_id],
            })
    return rows


def sweep(rows: list[dict], taus: list[float]) -> list[dict]:
    """For each tau: abstain on flagged claims with support_prob < tau."""
    n_bad = sum(1 for r in rows if not r["gold_supported"])
    n_good = sum(1 for r in rows if r["gold_supported"])
    out = []
    for tau in taus:
        dropped = [r for r in rows if r["flagged"] and r["support_prob"] < tau]
        bad_dropped = sum(1 for r in dropped if not r["gold_supported"])
        good_dropped = len(dropped) - bad_dropped
        out.append({
            "tau": round(tau, 3),
            "n_dropped": len(dropped),
            # of the real errors, how many did we remove
            "error_removal": round(bad_dropped / n_bad, 4) if n_bad else None,
            # of the good claims, how many survived
            "retention": round((n_good - good_dropped) / n_good, 4) if n_good else None,
            "errors_removed": bad_dropped,
            "good_claims_lost": good_dropped,
            # how many good claims thrown away per real error removed
            "cost_per_error": (
                round(good_dropped / bad_dropped, 2) if bad_dropped else None
            ),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default="QA")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load_claims(Path(args.run_dir), args.root, args.split, args.task)
    if not rows:
        raise SystemExit("no claims joined to gold -- check split/task")

    n_bad = sum(1 for r in rows if not r["gold_supported"])
    n_flag = sum(1 for r in rows if r["flagged"])
    tp = sum(1 for r in rows if r["flagged"] and not r["gold_supported"])
    print(f"{len(rows)} claims | {n_bad} gold-unsupported | {n_flag} flagged")
    print(f"detector precision {tp / max(1, n_flag):.3f}  "
          f"recall {tp / max(1, n_bad):.3f}")
    print()

    taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    table = sweep(rows, taus)

    head = (f"{'tau':>6}{'dropped':>9}{'err_removal':>13}{'retention':>11}"
            f"{'errors_rm':>11}{'good_lost':>11}{'good/err':>10}")
    print(head)
    print("-" * len(head))
    for r in table:
        cpe = "-" if r["cost_per_error"] is None else f"{r['cost_per_error']}"
        print(f"{r['tau']:>6}{r['n_dropped']:>9}{r['error_removal']:>13}"
              f"{r['retention']:>11}{r['errors_removed']:>11}"
              f"{r['good_claims_lost']:>11}{cpe:>10}")

    print()
    print("HOW TO READ THIS")
    print("  tau = 1.0 is the shipped router: abstain on ANY flag it could not fix.")
    print("  tau = 0.0 never abstains, i.e. the always-correct and Self-RAG arms.")
    print("  good/err is the exchange rate: supported claims destroyed per real")
    print("  error removed. There is no threshold at which that rate is free, and")
    print("  this script does not nominate a best tau -- which end of the trade-off")
    print("  you want is a property of the deployment, not of the data. A medical")
    print("  summariser and a search snippet do not want the same column.")
    print()
    print("  Scope: ABSTAIN only. Nothing here is evidence about REWRITE at any")
    print("  tau -- a rewrite changes the text, so it cannot be replayed from a log.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "run": args.run_dir,
            "n_claims": len(rows),
            "n_gold_unsupported": n_bad,
            "n_flagged": n_flag,
            "precision": round(tp / max(1, n_flag), 4),
            "recall": round(tp / max(1, n_bad), 4),
            "sweep": table,
            "scope": "abstain decision only; rewrite cannot be replayed offline",
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
