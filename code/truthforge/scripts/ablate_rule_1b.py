"""What does rule 1b cost, and what does it buy?

    python scripts/ablate_rule_1b.py runs/rt300_rule

Rule 1b abstains when a rewrite returned the claim unchanged, on the reasoning
that decoding is greedy so a retry is deterministic -- it cannot produce
anything new, and leaving a claim we KNOW is wrong standing as fact is worse
than saying we could not verify it.

That reasoning has a load-bearing premise: that the claim is known wrong. The
detector's precision on this data is 0.184, so "known wrong" is wrong about
four times in five. Rule 1b therefore converts a detector false positive into
a guaranteed information loss, and it fired on 223 of the roughly 299
abstentions in the n=300 rule arm -- the largest single source of withheld
content in the system.

**This counterfactual is exact, not simulated.** ABSTAIN is terminal and local:
making it an ACCEPT leaves the claim in place and changes nothing else -- no
other claim's text, no corrector call, no detector input. So the ledger under
"rule 1b accepts instead" can be computed from the log by moving each affected
claim from a DROPPED cell to a KEPT one. Nothing is being predicted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.cli import _turn_from_dict  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.eval.comparison import _rates, claim_fates  # noqa: E402
from truthforge.types import Action  # noqa: E402

TARGET = "corrector_could_not_fix_it"


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/rt300_rule")
    exs = {
        e.example_id: e
        for e in load_ragtruth(root="data/raw/ragtruth", split="test",
                               task_types={"QA"})
    }

    actual = {c: 0 for c in ("true_keep", "false_keep", "true_fix",
                             "false_fix", "true_drop", "false_drop")}
    counter = dict(actual)
    moved_good = moved_bad = 0

    for line in (run / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = _turn_from_dict(json.loads(line))
        ex = exs.get(t.example_id)
        if ex is None:
            continue
        gold = {c.claim_id: not ex.overlaps_span(c.start, c.end)
                for c in t.claims if c.start is not None}
        dropped = set(t.meta.get("abstained_claim_ids", []))
        by_1b = {
            d.claim_id for d in t.decisions
            if d.action is Action.ABSTAIN and d.reason == TARGET
        }
        # Use the real ledger for the shipped row rather than re-deriving it.
        # A hand-rolled approximation here produced 0.5920 error removal where
        # compare_arms reports 0.5840 -- a reader comparing the two tables
        # would have no way to tell which was wrong.
        for f in claim_fates(t, gold):
            actual[f.cell.value] += 1
            if f.claim_id in dropped and f.claim_id in by_1b:
                # The counterfactual: the claim stays in the answer untouched,
                # so it lands in a KEPT cell decided by its gold status.
                counter["true_keep" if f.was_supported else "false_keep"] += 1
                if f.was_supported:
                    moved_good += 1
                else:
                    moved_bad += 1
            else:
                counter[f.cell.value] += 1

    def show(name, cells):
        r, t, n = _rates(cells)
        net = (cells["true_fix"] + cells["true_drop"]) - (
            cells["false_fix"] + cells["false_drop"])
        print(f"\n{name}")
        print(f"  error removal {r:.4f}   retention {t:.4f}   net quality {n:.4f}")
        print(f"  errors removed {cells['true_fix'] + cells['true_drop']:4d}   "
              f"correct claims damaged/withheld "
              f"{cells['false_fix'] + cells['false_drop']:4d}   "
              f"net claims {net:+d}")
        print(f"  missed {cells['false_keep']}")

    print(f"run: {run}")
    print(f"rule 1b fired on {moved_good + moved_bad} claims "
          f"({moved_bad} genuinely unsupported, {moved_good} actually fine)")
    if moved_good + moved_bad:
        print(f"  precision of rule 1b's abstentions: "
              f"{moved_bad / (moved_good + moved_bad):.3f}")
    show("AS SHIPPED (rule 1b abstains)", actual)
    show("COUNTERFACTUAL (rule 1b accepts)", counter)
    print("\nThe counterfactual is exact: ABSTAIN is terminal and local, so")
    print("turning it into ACCEPT moves the claim between ledger cells and")
    print("changes nothing else. No corrector call or detector input differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
