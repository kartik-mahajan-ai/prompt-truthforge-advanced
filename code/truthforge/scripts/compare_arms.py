"""Head-to-head evaluation report: on what basis is one arm better?

    python scripts/compare_arms.py runs/vanilla runs/always runs/rule
    python scripts/compare_arms.py runs/always runs/rule --ours rule --baseline always
    python scripts/compare_arms.py runs/* --json runs/comparison.json

Produces the claim-fate ledger for each arm, the two rates that cannot both be
gamed, a paired-bootstrap significance test between the chosen arms, the Pareto
front, and a plain-language verdict that refuses to declare a win on a
non-significant difference.

Read `truthforge/eval/comparison.py` for why these metrics and not accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.cli import _turn_from_dict  # noqa: E402
from truthforge.eval.comparison import (  # noqa: E402
    compare_arms,
    format_ledger,
    pareto_front,
    score_arm,
)
from truthforge.eval.metrics import format_table  # noqa: E402


def load(run_dir: Path):
    path = run_dir / "turns.jsonl"
    if not path.exists():
        return None
    return [
        _turn_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_gold(turns, root: str, split: str, task: str | None) -> dict[str, bool]:
    """claim_id -> was_supported, from RAGTruth's annotated spans.

    Without this the ledger's "was this claim true to begin with?" column is
    the detector's own first verdict, which means the detector grades the work
    its own flags caused: a claim it wrongly flagged can only ever be scored as
    a true positive. RAGTruth ships character-level hallucination spans and our
    claims carry offsets, so the join is exact and there is no reason to accept
    the circular version.

    A claim is gold-unsupported when its span overlaps any annotated span --
    the same permissive criterion Gate 1 uses, so the two are comparable.

    **It has to key off the run's own claims.** The obvious implementation --
    re-split every response and key on `claim_id` -- joins at exactly 0%,
    silently, because claim ids are per-process UUIDs (`types._new_id`) and
    carry no information about the text. That version ran, printed a sensible
    "5312 claims, 6.7% unsupported", and would have scored every arm against
    an empty gold map while reporting `gold_source="annotated"`. So the lookup
    goes through (example_id, start, end), which is stable across processes,
    and `main` refuses to use a gold map that joined poorly.

    Caveat that must travel with these numbers: RAGTruth's spans
    UNDER-annotate. A claim marked supported here may still be unsupported, so
    `false_keep` is a lower bound and `false_fix` an upper one.
    """
    from truthforge.data.ragtruth import load_ragtruth

    by_id = {
        ex.example_id: ex
        for ex in load_ragtruth(
            root=root, split=split, task_types={task} if task else None
        )
    }

    gold: dict[str, bool] = {}
    for t in turns:
        ex = by_id.get(t.example_id)
        if ex is None:
            continue
        for c in t.claims:
            if c.start is None or c.end is None:
                continue
            gold[c.claim_id] = not ex.overlaps_span(c.start, c.end)
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories under runs/")
    ap.add_argument("--ours", default=None, help="arm to argue for (default: rule)")
    ap.add_argument("--baseline", default=None, help="arm to beat (default: always)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--gold", action="store_true",
                    help="score against RAGTruth's annotated spans instead "
                         "of the detector's own first verdict")
    ap.add_argument(
        "--rewrite-neutral", action="store_true",
        help="score a rewritten claim by its gold status, not by the "
             "detector's verdict on the new text. Licensed by the human "
             "annotation: that verdict had 6.7%% precision on 30 labelled "
             "rewrites. See eval/comparison.claim_fates.")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default="QA")
    args = ap.parse_args()

    loaded = {}
    for d in args.runs:
        p = Path(d)
        turns = load(p)
        if not turns:
            print(f"skipping {d}: no turns.jsonl")
            continue
        loaded[p.name] = turns

    scores = {}
    for name, turns in loaded.items():
        gold = None
        if args.gold:
            # Per arm: claim ids are per-run UUIDs, so one shared map cannot
            # cover them all.
            gold = build_gold(turns, args.root, args.split, args.task)
            want = {v.claim_id for t in turns for v in t.verdicts}
            hit = len(want & set(gold))
            frac = hit / max(1, len(want))
            n_unsup = sum(1 for v in gold.values() if not v)
            print(f"gold[{name}]: {hit}/{len(want)} claims joined ({frac:.1%}), "
                  f"{n_unsup} gold-unsupported")
            if frac < 0.9:
                # A partial join is not a smaller experiment, it is a biased
                # one: unjoined claims fall back to the detector's own verdict
                # and the table would mix two gold sources under one label.
                print(f"  REFUSING --gold for {name}: join rate {frac:.1%} < 90%. "
                      f"Check example_ids and split/task match the run.")
                gold = None
        scores[name] = score_arm(name, turns, gold,
                                 trust_post_rewrite=not args.rewrite_neutral)
        if scores[name].n_unscorable:
            print(f"  {scores[name].n_unscorable} claim(s) excluded from "
                  f"{name}: no gold label (missing character offsets)")

    if not scores:
        print("nothing to compare")
        return 1

    gold_sources = {s.gold_source for s in scores.values()}
    print("=" * 78)
    print("CLAIM-FATE LEDGER")
    print("=" * 78)
    if gold_sources == {"detector_on_original"}:
        print(
            "\n  NOTE: no annotated gold labels, so 'was supported' is the\n"
            "  detector's own verdict on the original answer. The judge is\n"
            "  grading work its own flags caused. Numbers below are indicative\n"
            "  only -- rerun on RAGTruth before quoting any of them.\n"
        )
    if gold_sources <= {"annotated", "annotated_rewrite_neutral"}:
        print(
            "\n  Gold labels are RAGTruth's annotated spans. Those spans\n"
            "  UNDER-annotate -- a re-annotation pass found hallucinations the\n"
            "  release marks clean -- so 'missed' is a LOWER bound.\n"
            "\n"
            "  'over-corr' and 'info loss' are UPPER bounds for a second,\n"
            "  separate reason: gold labels describe the ORIGINAL response, so\n"
            "  'was this claim fine to begin with' is annotated, but 'is it\n"
            "  fine NOW' is the detector's own judgement, at precision 0.184.\n"
            "  A rewrite the detector flags afterwards counts as damage\n"
            "  whether or not it did any.\n"
        )
    for s in scores.values():
        print()
        print(format_ledger(s))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    print(format_table([
        {
            "arm": s.arm,
            "net quality": round(s.net_quality, 4),
            "error removal": round(s.error_removal, 4),
            "retention": round(s.retention, 4),
            "over-corr": s.cells["false_fix"],
            "info loss": s.cells["false_drop"],
            "missed": s.cells["false_keep"],
            # Raw and countable by hand. Stays interpretable in the regime
            # where net quality does not -- see net_quality_is_unstable.
            "net claims": s.net_claims_improved,
            "tokens": round(s.tokens_per_turn, 1),
            "latency": round(s.latency_per_turn, 3),
        }
        for s in scores.values()
    ]))

    unstable = [s.arm for s in scores.values() if s.net_quality_unstable]
    if unstable:
        print(
            "\n  READ 'net quality' WITH CARE for: " + ", ".join(unstable) + "\n"
            "  It is the harmonic mean of error removal and retention, which is\n"
            "  pinned near zero whenever either term is. Once error removal is\n"
            "  tiny, the gap between 'exactly zero' and 'slightly above zero'\n"
            "  swamps anything retention can say, so an arm that fixes one claim\n"
            "  and breaks five hundred outranks one that changes nothing:\n"
            "      removal 0.0000, retention 1.0000  ->  0.0000\n"
            "      removal 0.0082, retention 0.5000  ->  0.0161\n"
            "  The 'net claims' column (errors removed minus good claims damaged)\n"
            "  stays interpretable here and is countable by hand."
        )

    front = pareto_front(list(scores.values()))
    print(f"\nPareto front (nothing beats these on both quality and cost): "
          f"{', '.join(front)}")

    # -- head to head ----------------------------------------------------
    ours = args.ours or ("rule" if "rule" in scores else None)
    base = args.baseline or ("always" if "always" in scores else None)
    report = None

    if ours in scores and base in scores:
        print()
        print("=" * 78)
        print(f"HEAD TO HEAD: {ours} vs {base}")
        print("=" * 78)
        report = compare_arms(scores[ours], scores[base], args.resamples)

        q, c = report["quality_delta"], report["cost_delta"]
        qp = report["quality_delta_pooled"]
        print(f"\n  quality  {q['mean_diff']:+.4f}  "
              f"95% CI [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}]  "
              f"p={q['p_value']:.4f}  {'SIGNIFICANT' if q['significant'] else 'not significant'}"
              f"   <- pre-registered (per-turn mean)")
        print(f"  quality* {qp['mean_diff']:+.4f}  "
              f"95% CI [{qp['ci_low']:+.4f}, {qp['ci_high']:+.4f}]  "
              f"p={qp['p_value']:.4f}  "
              f"{'SIGNIFICANT' if qp['significant'] else 'not significant'}"
              f"   <- pooled, NOT pre-registered")
        print(f"  tokens   {c['mean_diff']:+.1f}  "
              f"95% CI [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]  "
              f"p={c['p_value']:.4f}  {'SIGNIFICANT' if c['significant'] else 'not significant'}")
        print(f"\n  over-correction : {report['over_correction_ours']} "
              f"vs {report['over_correction_baseline']} (baseline)")
        print(f"  information loss: {report['info_loss_ours']} "
              f"vs {report['info_loss_baseline']} (baseline)")
        print(f"\n  VERDICT: {report['verdict']}   (follows the pre-registered test)")
        print(f"\n  What you may claim:\n    {report['claim']}")
        if q["significant"] != qp["significant"] or (
            q["mean_diff"] > 0
        ) != (qp["mean_diff"] > 0):
            print(
                "\n  NOTE: the two quality tests disagree. They measure different\n"
                "  things. The table's net quality is POOLED over all claims; the\n"
                "  pre-registered test averages PER-TURN net quality, and a turn\n"
                "  with no unsupported claims scores error-removal 1.0 by default,\n"
                "  so clean turns reward doing nothing. Most RAGTruth turns are\n"
                "  clean. Report both, and say which one the verdict follows."
            )
    else:
        print(f"\n(skipping head-to-head: need --ours and --baseline among "
              f"{sorted(scores)})")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "arms": {k: v.to_dict() for k, v in scores.items()},
                    "pareto_front": front,
                    "head_to_head": report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
