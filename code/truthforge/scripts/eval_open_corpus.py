"""Evaluate the open-corpus setting -- the only one where RE-RETRIEVE is real.

    python scripts/eval_open_corpus.py --arms noop always_correct rule crag self_rag
    python scripts/eval_open_corpus.py --n-answerable 40 --n-unanswerable 40 --dense

**Why this script exists.** RAGTruth hands every example its evidence, so the
retriever has nothing to do and the loop downgrades RE-RETRIEVE to REWRITE
every time. One of the four actions has therefore never been measured, and a
paper claiming a four-way routing decision cannot report numbers for three of
them. This builds a setting where the fourth action can fire and fail.

**How the corpus is built, and why not a synthetic one.** Writing a handful of
fake documents would make every number a statement about documents I wrote. So
the corpus is assembled from RAGTruth's own source passages -- real retrieved
web text -- in the same way SQuAD 2.0 builds unanswerable questions:

    answerable    -- the question's own source passages ARE in the corpus
    unanswerable  -- the question's own source passages are HELD OUT, while
                     every other question's passages remain

That second group is the point. The question is well formed, the corpus is
large and topically plausible, and the answer is genuinely absent. A system
that answers it anyway is hallucinating, and no amount of rewriting can save
it -- the only correct outputs are ABSTAIN or RE-RETRIEVE-then-ABSTAIN.

**What is measured, and the trap it avoids.** Refusal rate alone is gameable:
a system that refuses everything scores 1.0. So the two rates are always
reported together and neither is reported alone:

    correct_refusal  -- refused when the answer was absent      (want high)
    over_refusal     -- refused when the answer was present     (want low)

A single number combining them would hide which way an arm fails, which is the
whole substance of the comparison, so there isn't one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.config import Config  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.detect.minicheck import build_detector  # noqa: E402
from truthforge.loop import LoopComponents, TruthForgeLoop  # noqa: E402
from truthforge.models import build_generator  # noqa: E402
from truthforge.correct.rewrite import Rewriter  # noqa: E402
from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.retrieve.base import BM25Index, HybridRetriever  # noqa: E402
from truthforge.router.rules import build_router  # noqa: E402
from truthforge.types import Action  # noqa: E402


# --------------------------------------------------------------------------


def build_corpus(examples, n_answerable: int, n_unanswerable: int, seed: int):
    """Return (corpus_texts, corpus_ids, questions).

    Every question's passages go into the corpus EXCEPT those of the questions
    marked unanswerable. Held-out passages are removed by source id, so a
    near-duplicate passage reached through another example would still be
    found -- which is the honest version of the test.
    """
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)

    need = n_answerable + n_unanswerable
    if len(pool) < need:
        raise SystemExit(
            f"need {need} examples, loaded {len(pool)}. Lower --n-answerable/"
            f"--n-unanswerable or raise --limit."
        )
    chosen = pool[:need]
    unanswerable = {ex.example_id for ex in chosen[n_answerable:]}

    # Deduplicate by source id: RAGTruth reuses one source across several
    # responses, so held-out text can sneak back in through a sibling example.
    held_out_sources = {
        p.doc_id.split(":")[0] for ex in chosen if ex.example_id in unanswerable
        for p in ex.passages
    }

    texts, ids, seen = [], [], set()
    for ex in examples:
        for p in ex.passages:
            src = p.doc_id.split(":")[0]
            if src in held_out_sources or p.doc_id in seen:
                continue
            seen.add(p.doc_id)
            texts.append(p.text)
            ids.append(p.doc_id)

    questions = [
        {
            "example_id": ex.example_id,
            "question": ex.question,
            "answerable": ex.example_id not in unanswerable,
        }
        for ex in chosen
    ]
    return texts, ids, questions


def score_turn(turn, answerable: bool) -> dict:
    """Did the arm answer, refuse, or do something in between?

    A system can decline in two ways and both count. The router can abstain on
    every claim, leaving nothing; or the generator can simply say so -- "The
    passages do not contain information about oil stains on concrete
    driveways". The first version of this counted only the router, reported
    0.0 correct refusals for every arm, and made it look as though nothing ever
    refused. The generator was declining on 56% of unanswerable questions in
    its own words the whole time.
    """
    from truthforge.claims.splitter import answer_is_refusal

    abstained = set(turn.meta.get("abstained_claim_ids", []))
    n_claims = len(turn.claims)
    # Ask the loop whether it refused; do not infer it from the text. The
    # answerability gate refuses BEFORE generating, so there are no claims to
    # abstain on and the answer is the system's own abstain message -- which
    # the refusal-phrase regex does not match. Inferring from text, this
    # scorer counted 0 of the gate's 29 refusals and reported that switching
    # the gate on cut correct refusals from 0.56 to 0.04. A refusal gate
    # cannot make a system refuse less; the measurement could not see it.
    router_refused = (
        bool(turn.abstained)
        or bool(turn.meta.get("refused_before_generating"))
        or (not turn.final_answer.strip())
        or (n_claims > 0 and len(abstained) == n_claims)
    )
    text_refused = answer_is_refusal(turn.final_answer)
    refused = router_refused or text_refused
    actions = [d.action for d in turn.decisions]
    return {
        "example_id": turn.example_id,
        "answerable": answerable,
        "refused": refused,
        # Split out, because "the router decided to withhold" and "the model
        # happened to say it did not know" are different capabilities and the
        # paper's claim is about the first one.
        "router_refused": router_refused,
        "text_refused": text_refused,
        "n_claims": n_claims,
        "n_abstained": len(abstained),
        "reretrieves": sum(1 for a in actions if a is Action.RERETRIEVE),
        "rewrites": sum(1 for a in actions if a is Action.REWRITE),
        "abstains": sum(1 for a in actions if a is Action.ABSTAIN),
        "accepts": sum(1 for a in actions if a is Action.ACCEPT),
        "reasons": [d.reason for d in turn.decisions],
        "latency_s": turn.wall_clock_s,
        "answer": turn.final_answer[:400],
    }


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]

    def rate(xs, key, default=0):
        if not xs:
            return None
        return round(sum(r.get(key, default) for r in xs) / len(xs), 4)

    reasons: dict[str, int] = {}
    for r in rows:
        for x in r["reasons"]:
            reasons[x] = reasons.get(x, 0) + 1

    return {
        "n": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        # The pair that must be read together. Either alone is gameable.
        "correct_refusal": rate(una, "refused"),
        "over_refusal": rate(ans, "refused"),
        # How much of the refusal behaviour the ROUTER is responsible for. If
        # this is ~0 while correct_refusal is high, the generator is doing the
        # work and the routing layer is contributing nothing here -- which is
        # a finding, not a detail.
        "correct_refusal_by_router": rate(una, "router_refused"),
        "over_refusal_by_router": rate(ans, "router_refused"),
        "reretrieve_per_turn": rate(rows, "reretrieves"),
        "rewrite_per_turn": rate(rows, "rewrites"),
        # Did RE-RETRIEVE ever fire at all? If this is 0 the setting failed to
        # exercise the action and the row proves nothing about it.
        "turns_with_reretrieve": sum(1 for r in rows if r["reretrieves"]),
        "latency_per_turn": rate(rows, "latency_s"),
        "routing_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300, help="examples to draw the corpus from")
    ap.add_argument("--n-answerable", type=int, default=25)
    ap.add_argument("--n-unanswerable", type=int, default=25)
    ap.add_argument("--arms", nargs="+",
                    default=["noop", "always_correct", "rule", "crag", "self_rag"])
    ap.add_argument("--detector", default="nli")
    ap.add_argument("--generator", default="local")
    ap.add_argument("--dense", action="store_true", help="BM25 + dense RRF instead of BM25")
    ap.add_argument("--answerability-gate", action="store_true",
                    help="refuse before generating when the question's terms are "
                         "largely absent from the retrieved evidence")
    ap.add_argument("--min-question-coverage", type=float, default=None,
                    help="override the gate threshold (default: the value "
                         "selected on the development set)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", default="runs/open_corpus.json")
    args = ap.parse_args()

    examples = load_ragtruth(
        root=args.root, split=args.split, task_types={"QA"}, limit=args.limit
    )
    texts, ids, questions = build_corpus(
        examples, args.n_answerable, args.n_unanswerable, args.seed
    )
    print(f"corpus: {len(texts)} passages | "
          f"{args.n_answerable} answerable + {args.n_unanswerable} unanswerable questions")
    print(f"retrieval: {'BM25 + dense (RRF)' if args.dense else 'BM25'}")

    retriever = (
        HybridRetriever(texts, ids, device=args.device, use_dense=True)
        if args.dense
        else BM25Index(texts, ids)
    )

    cfg = Config()
    cfg.router.answerability_gate = args.answerability_gate
    if args.min_question_coverage is not None:
        cfg.router.min_question_coverage = args.min_question_coverage
    if args.answerability_gate:
        print(f"answerability gate ON at {cfg.router.min_question_coverage}")
    cfg.detector.name = args.detector
    cfg.detector.device = args.device
    cfg.generator.device = args.device

    generate_fn = build_generator(args.generator, cfg.generator)
    detector = build_detector(cfg)
    rewriter = Rewriter(generate_fn, abstain_message=cfg.loop.abstain_message)

    contradiction_fn = None
    if args.detector in ("minicheck", "nli", "ensemble"):
        from truthforge.detect.nli import build_contradiction_fn

        contradiction_fn = build_contradiction_fn(cfg)

    results = {}
    for arm in args.arms:
        cfg.router.name = arm
        print(f"\n=== {arm} ===")
        rows, t0 = [], time.time()
        for i, q in enumerate(questions, 1):
            components = LoopComponents(
                detector=detector,
                # The evidence is NOT fixed here, so CRAG's web-search branch
                # has somewhere to go and RE-RETRIEVE is a real action.
                router=build_router(cfg, fixed_evidence=False),
                retriever=retriever,
                splitter=RuleSplitter(),
                rewriter=rewriter,
                generate_fn=generate_fn,
                contradiction_fn=contradiction_fn,
            )
            turn = TruthForgeLoop(components, cfg).run(
                question=q["question"],
                example_id=q["example_id"],
                dataset="open_corpus",
            )
            rows.append(score_turn(turn, q["answerable"]))
            if i % 10 == 0:
                print(f"  {i}/{len(questions)}  ({time.time() - t0:.0f}s)", flush=True)

        s = summarise(rows)
        results[arm] = {"summary": s, "rows": rows}
        print(f"  correct_refusal {s['correct_refusal']} "
              f"(router {s['correct_refusal_by_router']})  "
              f"over_refusal {s['over_refusal']}  "
              f"re-retrieve/turn {s['reretrieve_per_turn']}  "
              f"({s['turns_with_reretrieve']} turns used it)")

    # ---- table ----------------------------------------------------------
    print("\n" + "=" * 86)
    print("OPEN-CORPUS RESULTS   (read the two refusal columns together)")
    print("=" * 86)
    head = (f"{'arm':<16}{'correct_ref':>13}{'(router)':>10}{'over_ref':>11}"
            f"{'(router)':>10}{'reretr/turn':>13}{'lat_s':>9}")
    print(head)
    print("-" * len(head))
    for arm, r in results.items():
        s = r["summary"]
        print(f"{arm:<16}{s['correct_refusal']:>13}{s['correct_refusal_by_router']:>10}"
              f"{s['over_refusal']:>11}{s['over_refusal_by_router']:>10}"
              f"{s['reretrieve_per_turn']:>13}{s['latency_per_turn']:>9}")

    fired = {a: r["summary"]["turns_with_reretrieve"] for a, r in results.items()}
    print(f"\nRE-RETRIEVE fired in: {fired}")
    if not any(fired.values()):
        print("  WARNING: no arm used RE-RETRIEVE. This run says nothing about the")
        print("  fourth action -- check the retriever and the coverage threshold.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": {
            "split": args.split, "limit": args.limit, "seed": args.seed,
            "n_answerable": args.n_answerable, "n_unanswerable": args.n_unanswerable,
            "detector": args.detector, "generator": args.generator,
            "dense": args.dense, "corpus_passages": len(texts),
            "answerability_gate": args.answerability_gate,
            "min_question_coverage": cfg.router.min_question_coverage,
        },
        "arms": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
