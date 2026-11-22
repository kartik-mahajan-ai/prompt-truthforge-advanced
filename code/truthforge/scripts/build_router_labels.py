"""Auto-label the router's training data (the Adaptive-RAG trick).

For every flagged claim, run each available action from the SAME starting
state, classify what it did, and record the action that should have been
chosen. No human annotation required.

    python scripts/build_router_labels.py --sample --out runs/router_labels.jsonl

Labelling policy, stated up front because it defines what the learned router
is being taught:

  1. Among actions whose outcome is BETTER, take the CHEAPEST (fewest tokens).
     Two actions that both fix the claim are not equally good.
  2. If no action is BETTER but at least one is SAME, label ACCEPT. Spending
     compute to change nothing is exactly the behaviour we want trained out.
  3. If every action is WORSE, label ABSTAIN. Saying nothing beats saying
     something damaged.

Note the asymmetry with the ORACLE router in `router/rules.py`: the oracle is
allowed to know the best action and is an upper bound; the learned router only
sees features. Do not report the oracle as achievable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.config import Config  # noqa: E402
from truthforge.correct.rewrite import Rewriter, splice  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth, load_sample  # noqa: E402
from truthforge.detect.baselines import contradiction_signal, coverage  # noqa: E402
from truthforge.detect.minicheck import build_detector  # noqa: E402
from truthforge.eval.outcomes import classify_attempt  # noqa: E402
from truthforge.models import build_generator  # noqa: E402
from truthforge.retrieve.base import GivenRetriever  # noqa: E402
from truthforge.router.base import ClaimState  # noqa: E402
from truthforge.types import Action, CorrectionAttempt, Outcome  # noqa: E402


def candidate_actions(retriever) -> list[Action]:
    acts = [Action.ACCEPT, Action.REWRITE, Action.ABSTAIN]
    if getattr(retriever, "supports_reretrieval", False):
        acts.insert(2, Action.RERETRIEVE)
    return acts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="train")
    ap.add_argument("--task", default="QA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detector", default="lexical")
    ap.add_argument("--generator", default="echo")
    ap.add_argument("--out", default="runs/router_labels.jsonl")
    args = ap.parse_args()

    cfg = Config()
    cfg.detector.name = args.detector

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
    print(f"{len(examples)} examples")
    if args.sample:
        print("SYNTHETIC fixture -- a policy trained on this is meaningless")

    detector = build_detector(cfg)
    generate_fn = build_generator(args.generator, cfg.generator)
    rewriter = Rewriter(generate_fn, abstain_message=cfg.loop.abstain_message)
    splitter = RuleSplitter()

    rows, skipped = [], 0
    for ex in examples:
        retriever = GivenRetriever(ex.passages)
        claims = splitter.split(ex.response)
        if not claims:
            continue
        verdicts = {v.claim_id: v for v in detector.check(claims, ex.passages)}

        for claim in claims:
            v = verdicts[claim.claim_id]
            if not v.is_flagged:
                continue  # the router only sees flagged claims

            state = ClaimState(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                verdict=v,
                coverage=coverage(claim.text, ex.passages),
                contradiction=contradiction_signal(claim.text, ex.passages),
                iteration=0,
            )

            trials = {}
            for action in candidate_actions(retriever):
                if action in (Action.ACCEPT, Action.ABSTAIN):
                    # Terminal actions change no text and cost nothing, so
                    # they are never BETTER by construction. They exist in the
                    # trial set so the fallbacks in the policy are grounded.
                    trials[action] = {"outcome": Outcome.SAME.value, "tokens": 0}
                    continue

                if action is Action.RERETRIEVE:
                    widened = retriever.retrieve(
                        f"{ex.question} {claim.text}", cfg.retrieval.reretrieve_top_k
                    )
                    merged = list({p.doc_id: p for p in [*ex.passages, *widened]}.values())
                    after = detector.check([claim], merged)[0]
                    attempt = CorrectionAttempt(
                        claim_id=claim.claim_id,
                        action=action,
                        iteration=0,
                        before_text=claim.text,
                        after_text=claim.text,
                        before_support_prob=v.support_prob,
                        after_support_prob=after.support_prob,
                    )
                    classify_attempt(attempt, merged)
                    # Re-retrieval changes no text, so the text-diff heuristics
                    # cannot fire; judge it purely on the support delta.
                    delta = after.support_prob - v.support_prob
                    outcome = (
                        Outcome.BETTER.value if delta >= 0.10 else Outcome.SAME.value
                    )
                    trials[action] = {"outcome": outcome, "tokens": 0, "delta": delta}
                    continue

                # REWRITE
                rw = rewriter.rewrite(claim, ex.response, ex.passages)
                spliced = splice(ex.response, claim, rw.rewritten)
                after = detector.check(
                    [_as_claim(claim, rw.rewritten, spliced)], ex.passages
                )[0]
                attempt = CorrectionAttempt(
                    claim_id=claim.claim_id,
                    action=action,
                    iteration=0,
                    before_text=claim.text,
                    after_text=rw.rewritten,
                    before_support_prob=v.support_prob,
                    after_support_prob=after.support_prob,
                )
                classify_attempt(attempt, ex.passages)
                trials[action] = {
                    "outcome": attempt.outcome.value,
                    "worse_kind": attempt.worse_kind.value if attempt.worse_kind else None,
                    "tokens": rw.cost.total_tokens,
                }

            label = pick_label(trials)
            if label is None:
                skipped += 1
                continue

            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "example_id": ex.example_id,
                    "claim_text": claim.text,
                    "features": state.features(),
                    "label": label.value,
                    "trials": {a.value: t for a, t in trials.items()},
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    dist = Counter(r["label"] for r in rows)
    print(f"\nwrote {len(rows)} labelled claims to {out} ({skipped} skipped)")
    print("label distribution:", dict(dist))
    if len(dist) < 2:
        print("WARNING: one label dominates completely. A policy trained on "
              "this collapses to a constant -- fix the detector or the "
              "corrector before training.")
    return 0


def pick_label(trials: dict[Action, dict]):
    """Apply the labelling policy documented at the top of this file."""
    better = {a: t for a, t in trials.items() if t["outcome"] == Outcome.BETTER.value}
    if better:
        return min(better, key=lambda a: better[a]["tokens"])
    if any(t["outcome"] == Outcome.SAME.value for t in trials.values()):
        return Action.ACCEPT
    return Action.ABSTAIN


def _as_claim(claim, new_text: str, new_answer: str):
    from truthforge.types import Claim

    pos = new_answer.find(new_text)
    return Claim(
        text=new_text,
        claim_id=claim.claim_id,
        start=pos if pos >= 0 else None,
        end=pos + len(new_text) if pos >= 0 else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
