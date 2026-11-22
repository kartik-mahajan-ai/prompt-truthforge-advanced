"""Run one arm of the experiment end to end and write a run log.

    # smoke test: no GPU, no downloads, no model weights
    python -m truthforge.cli --sample --detector lexical --router rule \
        --generator echo

    # compare arms on the same examples
    python -m truthforge.cli --sample --router noop           --run-name vanilla
    python -m truthforge.cli --sample --router always_correct --run-name always
    python -m truthforge.cli --sample --router rule           --run-name rule
    python -m truthforge.cli --compare runs/vanilla runs/always runs/rule

Each run writes `runs/<name>/{config.json,turns.jsonl,metrics.json}`. The
turns file is the primary artifact: every claim, verdict, routing decision,
correction attempt and cost record for every turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .claims.splitter import RuleSplitter
from .config import Config
from .correct.rewrite import Rewriter
from .data.ragtruth import load_ragtruth, load_sample
from .detect.minicheck import build_detector
from .eval.metrics import format_table, results_row, system_metrics
from .eval.outcomes import classify_attempt
from .loop import LoopComponents, TruthForgeLoop
from .models import build_generator
from .retrieve.base import GivenRetriever
from .router.rules import build_router


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="use the synthetic fixture")
    ap.add_argument("--root", default="data/raw/ragtruth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detector", default="lexical")
    ap.add_argument("--router", default="rule")
    ap.add_argument("--generator", default="echo", help="echo | local")
    ap.add_argument("--run-name", default="dev")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--config", default=None, help="load a Config json/yaml")
    ap.add_argument(
        "--confirm-abstain", default=None, choices=["minicheck", "nli"],
        help="withhold a claim only if this second verifier also flags it. "
             "Selected on the training split: requiring NLI and MiniCheck to "
             "agree raised flag precision from 0.228 to 0.308.",
    )
    ap.add_argument(
        "--router-labels", default=None,
        help="auto-label file for --router oracle. The oracle is an UPPER "
             "BOUND, not an achievable system: it is allowed to see which "
             "action worked. Never report it as a result.",
    )
    ap.add_argument(
        "--compare", nargs="*", default=None, help="run dirs to tabulate instead"
    )
    args = ap.parse_args(argv)

    if args.compare:
        return _compare(args.compare)

    cfg = Config.load(args.config) if args.config else Config()
    cfg.detector.name = args.detector
    cfg.router.name = args.router
    cfg.loop.max_iterations = args.max_iterations
    cfg.run_name = args.run_name
    cfg.out_dir = args.out_dir
    cfg.dataset = "ragtruth_sample" if args.sample else "ragtruth"
    cfg.split = "sample" if args.sample else args.split
    cfg.limit = args.limit

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
    if args.sample and args.limit:
        examples = examples[: args.limit]
    print(f"{len(examples)} examples | detector={args.detector} "
          f"router={args.router} generator={args.generator}")
    if args.sample:
        print("SYNTHETIC fixture -- smoke test only, do not report these numbers")

    # The oracle needs to know which action worked on each claim. That is
    # knowledge no deployable system has, which is the point: it bounds what
    # perfect routing could achieve with this detector and this corrector.
    # Keyed {example_id: {claim_text: action}}, never by claim_id: ids are
    # per-process UUIDs and the labels come from a different process, so an
    # id-keyed map joins nothing and the oracle silently becomes the rule
    # router. See OracleRouter.route.
    labels_by_example: dict[str, dict[str, str]] = {}
    if args.router_labels:
        import json as _json

        for line in Path(args.router_labels).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            # build_router_labels writes the chosen action under "label".
            if row.get("example_id") and row.get("claim_text") and row.get("label"):
                labels_by_example.setdefault(str(row["example_id"]), {})[
                    row["claim_text"]
                ] = row["label"]
        n = sum(len(v) for v in labels_by_example.values())
        print(f"oracle labels: {n} claims over {len(labels_by_example)} examples")
    elif cfg.router.name == "oracle":
        raise SystemExit("--router oracle needs --router-labels")
    oracle_routers = []

    generate_fn = build_generator(args.generator, cfg.generator)
    detector = build_detector(cfg)
    rewriter = Rewriter(generate_fn, abstain_message=cfg.loop.abstain_message)

    # The ABSTAIN branch needs a real source-conflict signal. Load the NLI
    # analyzer whenever we are already committed to running neural models;
    # with a pure-baseline detector, keep the whole run dependency-free.
    contradiction_fn = None
    if args.detector in ("minicheck", "nli", "ensemble"):
        from .detect.nli import build_contradiction_fn

        contradiction_fn = build_contradiction_fn(cfg)

    # Second verifier that must confirm a flag before a claim is withheld.
    confirm_fn = None
    if args.confirm_abstain:
        import copy

        ccfg = copy.deepcopy(cfg)
        ccfg.detector.name = args.confirm_abstain
        confirm_det = build_detector(ccfg)
        print(f"abstentions require confirmation by {args.confirm_abstain} "
              f"(flag threshold {getattr(confirm_det, 'flag_threshold', '?')})")

        def confirm_fn(claim, passages, _d=confirm_det):
            return _d.check([claim], passages)[0].is_flagged

    turns = []
    for i, ex in enumerate(examples, 1):
        router = build_router(
            cfg,
            labels_by_example.get(str(ex.example_id), {}) if args.router_labels else None,
            # RAGTruth freezes the evidence, so CRAG's web-search branch
            # has nowhere to go and must say so rather than silently
            # turning into a different policy.
            fixed_evidence=True,
        )
        if cfg.router.name == "oracle":
            oracle_routers.append(router)
        components = LoopComponents(
            detector=detector,
            router=router,
            # RAGTruth fixes the evidence, so RERETRIEVE is unavailable and the
            # loop downgrades it explicitly. See retrieve/base.py.
            retriever=GivenRetriever(ex.passages),
            splitter=RuleSplitter(),
            rewriter=rewriter,
            generate_fn=generate_fn,
            contradiction_fn=contradiction_fn,
            confirm_fn=confirm_fn,
        )
        loop = TruthForgeLoop(components, cfg)
        turn = loop.run(
            question=ex.question,
            answer=ex.response,
            passages=ex.passages,
            example_id=ex.example_id,
            dataset=cfg.dataset,
        )
        for attempt in turn.attempts:
            classify_attempt(attempt, ex.passages)
        turns.append(turn)
        if i % 25 == 0 or i == len(examples):
            print(f"  {i}/{len(examples)}", flush=True)

    out = Path(args.out_dir) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    with (out / "turns.jsonl").open("w", encoding="utf-8") as fh:
        for t in turns:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    metrics = system_metrics(turns)
    if oracle_routers:
        # How often the oracle actually knew the answer. A low hit rate means
        # the row is mostly the fallback rule router and cannot be presented
        # as an upper bound -- which is exactly what an id-keyed join produced.
        # Over FLAGGED claims: those are the only ones the auto-labeller
        # labels, so they are the only decisions the oracle could have made.
        decided = sum(r.flagged for r in oracle_routers)
        misses = sum(r.misses for r in oracle_routers)
        hit_rate = 1 - misses / decided if decided else 0.0
        metrics["oracle_flagged_decisions"] = decided
        metrics["oracle_misses"] = misses
        metrics["oracle_hit_rate"] = round(hit_rate, 4)
        print(f"\noracle hit rate: {hit_rate:.1%} ({decided - misses}/{decided} "
              f"flagged claims had a label; the rest fell back to the rule router)")
        if hit_rate < 0.5:
            print("  WARNING: this row is mostly the fallback rule router. It is not")
            print("  an upper bound and must not be reported as one.")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(json.dumps(
        {k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2
    ))
    print("\nrouting reasons:")
    for reason, n in metrics.get("routing_reasons", {}).items():
        print(f"  {n:>4}  {reason}")
    print(f"\nwrote {out}")
    return 0


def _compare(run_dirs: list[str]) -> int:
    """Tabulate several runs into the paper's main table."""
    rows = []
    for d in run_dirs:
        path = Path(d) / "turns.jsonl"
        if not path.exists():
            print(f"skipping {d}: no turns.jsonl")
            continue
        turns = [
            _turn_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.append(results_row(Path(d).name, turns))

    if not rows:
        print("nothing to compare")
        return 1
    print(format_table(rows))
    return 0


def _turn_from_dict(d: dict) -> "object":
    """Rehydrate just enough of a TurnResult for the metrics functions."""
    from .types import (
        Action, Claim, CorrectionAttempt, CostRecord, Label, Outcome, Passage,
        RoutingDecision, TurnResult, Verdict, WorseKind,
    )

    t = TurnResult(
        question=d["question"],
        final_answer=d["final_answer"],
        initial_answer=d["initial_answer"],
        turn_id=d["turn_id"],
        dataset=d.get("dataset", "unknown"),
        example_id=d.get("example_id"),
        abstained=d.get("abstained", False),
        iterations_used=d.get("iterations_used", 0),
        wall_clock_s=d.get("wall_clock_s", 0.0),
        meta=d.get("meta", {}),
    )
    # Evidence. Needed by anything that has to show a reader WHY a verdict was
    # reached -- the annotation workbook shipped with an empty EVIDENCE column
    # until this was added, which would have had a human judging rewrites
    # against nothing.
    t.passages = [
        Passage(
            doc_id=p["doc_id"],
            text=p["text"],
            source=p.get("source", "given"),
            score=p.get("score", 0.0),
            rank=p.get("rank", 0),
            meta=p.get("meta", {}),
        )
        for p in d.get("passages", [])
    ]
    # Claims carry the character offsets that RAGTruth's gold spans join on.
    # Leaving them out made `compare_arms --gold` join 0 of 392 claims and
    # silently fall back to the detector's own verdict -- the circular scoring
    # the gold labels exist to avoid.
    t.claims = [
        Claim(
            text=c["text"],
            claim_id=c["claim_id"],
            start=c.get("start"),
            end=c.get("end"),
            sent_index=c.get("sent_index", 0),
            meta=c.get("meta", {}),
        )
        for c in d.get("claims", [])
    ]
    # Routing decisions: needed by any report that counts actions per arm.
    t.decisions = [
        RoutingDecision(
            claim_id=x["claim_id"],
            action=Action(x["action"]),
            reason=x.get("reason", ""),
            router=x.get("router", ""),
            iteration=x.get("iteration", 0),
        )
        for x in d.get("decisions", [])
    ]
    t.verdicts = [
        Verdict(
            claim_id=v["claim_id"],
            label=Label(v["label"]),
            support_prob=v["support_prob"],
            detector=v["detector"],
        )
        for v in d.get("verdicts", [])
    ]
    t.attempts = [
        CorrectionAttempt(
            claim_id=a["claim_id"],
            action=Action(a["action"]),
            iteration=a["iteration"],
            before_text=a["before_text"],
            after_text=a["after_text"],
            before_support_prob=a["before_support_prob"],
            after_support_prob=a.get("after_support_prob"),
            outcome=Outcome(a["outcome"]) if a.get("outcome") else None,
            worse_kind=WorseKind(a["worse_kind"]) if a.get("worse_kind") else None,
            outcome_source=a.get("outcome_source"),
            # Must be restored, or `rejection_rate` reads 0 for every run
            # loaded from disk while the live run reports it correctly.
            rejection=a.get("rejection"),
            notes=a.get("notes", ""),
        )
        for a in d.get("attempts", [])
    ]
    t.costs = [
        CostRecord(
            stage=c["stage"],
            prompt_tokens=c.get("prompt_tokens", 0),
            completion_tokens=c.get("completion_tokens", 0),
            latency_s=c.get("latency_s", 0.0),
            retrieval_calls=c.get("retrieval_calls", 0),
            llm_calls=c.get("llm_calls", 0),
            detector_calls=c.get("detector_calls", 0),
        )
        for c in d.get("costs", [])
    ]
    return t


if __name__ == "__main__":
    sys.exit(main())
