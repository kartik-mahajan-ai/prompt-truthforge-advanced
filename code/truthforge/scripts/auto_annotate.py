"""Label correction outcomes automatically, and report what that is worth.

    python scripts/auto_annotate.py runs/always
    python scripts/auto_annotate.py runs/always --judge-model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/auto_annotate.py runs/always --human annotation/labels_always_K.csv

Produces three things:

  1. heuristic labels   -- the rule-based classifier (token counts, hedge
                           words, length ratios)
  2. LLM-judge labels   -- an instruct model reading before/after/evidence
  3. agreement between them, and against any human labels supplied

**On what the agreement means.** Heuristic-vs-LLM is agreement between two
METHODOLOGICALLY INDEPENDENT methods: one counts tokens, the other reads. When
those agree it is genuine evidence that the outcome categories are detectable.
Two LLMs agreeing would be mostly a statement about LLMs, so this script
reports the kind of every annotator and never averages across kinds.

A human anchor is still required to claim validity, but it does not have to be
300 items. Fifty, labelled by one person, is enough to say whether the
automatic labels track human judgement -- and the paper can report exactly
that, with the n stated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.eval.llm_judge import LLMJudge, label_batch  # noqa: E402
from truthforge.eval.outcomes import (  # noqa: E402
    classify_attempt,
    cohens_kappa,
    load_human_labels,
)
from truthforge.types import (  # noqa: E402
    Action,
    CorrectionAttempt,
    Outcome,
    Passage,
)


def load_attempts(run_dir: Path):
    attempts, passages = [], {}
    path = run_dir / "turns.jsonl"
    if not path.exists():
        raise SystemExit(f"no turns.jsonl in {run_dir}")
    for line in path.read_text(encoding="utf-8").splitlines():
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
                )
            )
            passages[a["claim_id"]] = ps
    return attempts, passages


def key_of(a: CorrectionAttempt) -> str:
    return f"{a.claim_id}:{a.iteration}:{a.action.value}"


def compare(name_a, labels_a, name_b, labels_b):
    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        return {"pair": f"{name_a} vs {name_b}", "n": 0, "note": "no overlap"}
    xa = [labels_a[k]["outcome"] for k in shared]
    xb = [labels_b[k]["outcome"] for k in shared]
    agree = sum(1 for p, q in zip(xa, xb) if p == q) / len(shared)
    k = cohens_kappa(xa, xb)

    both_worse = [
        s for s in shared
        if labels_a[s]["outcome"] == "worse" and labels_b[s]["outcome"] == "worse"
    ]
    out = {
        "pair": f"{name_a} vs {name_b}",
        "n": len(shared),
        "raw_agreement": round(agree, 4),
        "kappa": None if k != k else round(k, 4),
        "n_both_worse": len(both_worse),
    }
    if both_worse:
        ka = [labels_a[s]["worse_kind"] for s in both_worse]
        kb = [labels_b[s]["worse_kind"] for s in both_worse]
        wk = cohens_kappa(ka, kb)
        out["worse_kind_agreement"] = round(
            sum(1 for p, q in zip(ka, kb) if p == q) / len(both_worse), 4
        )
        out["worse_kind_kappa"] = None if wk != wk else round(wk, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--judge-model", default=None,
                    help="HF id for the judge; defaults to the configured generator")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--human", default=None, help="a completed worksheet CSV")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--only-keys", default=None,
        help="a labels CSV; judge only the attempts it covers. Use with a human "
             "labels file to compare the judge against the human on the SAME items.")
    args = ap.parse_args()

    run = Path(args.run_dir)
    attempts, passages = load_attempts(run)
    if args.only_keys:
        wanted = {
            row["attempt_key"]
            for row in csv.DictReader(open(args.only_keys, encoding="utf-8"))
            if row.get("attempt_key")
        }
        attempts = [a for a in attempts if key_of(a) in wanted]
        print(f"--only-keys: {len(attempts)} attempts match {len(wanted)} keys")
    if args.limit:
        attempts = attempts[: args.limit]
    if not attempts:
        print("no correction attempts in this run -- nothing to label")
        return 1
    print(f"{len(attempts)} correction attempts from {run}")

    # -- 1. heuristic ----------------------------------------------------
    heuristic = {}
    for a in attempts:
        classify_attempt(a, passages.get(a.claim_id, []))
        heuristic[key_of(a)] = {
            "outcome": a.outcome.value if a.outcome else "",
            "worse_kind": a.worse_kind.value if a.worse_kind else "",
        }
    print(f"heuristic labelled {len(heuristic)}")

    # -- 2. LLM judge ----------------------------------------------------
    from truthforge.config import GeneratorConfig
    from truthforge.models import LocalHFGenerator

    cfg = GeneratorConfig(device=args.device)
    if args.judge_model:
        cfg.model = args.judge_model
    print(f"loading judge: {cfg.model}")
    from truthforge.config import GeneratorConfig as _G
    if cfg.model == _G().model:
        # Trap #2: the judge and the corrector must be different model
        # families. This is the corrector. Its labels are reported as a
        # SAME-MODEL negative control -- "can a model grade its own edits?" --
        # and never as validation of anything.
        print("  WARNING: judge == corrector. Reporting as a same-model "
              "negative control, not as validation.")
    gen = LocalHFGenerator(cfg)
    judge = LLMJudge(gen, name=cfg.model.split("/")[-1])

    batch = label_batch(judge, attempts, passages)
    llm = batch["labels"]
    print(f"judge labelled {batch['n_labelled']}/{batch['n_attempts']}"
          f"  ({batch['n_unparseable']} unparseable)")

    annotators = {
        "heuristic": {"labels": heuristic, "kind": "automatic (rule-based)"},
        batch["annotator"]: {"labels": llm, "kind": "automatic (LLM)"},
    }
    if args.human:
        human = load_human_labels(args.human)
        annotators["human"] = {"labels": human, "kind": "human"}
        print(f"human labels loaded: {len(human)}")

    # -- 3. agreement ----------------------------------------------------
    print("\n" + "=" * 70)
    print("AGREEMENT")
    print("=" * 70)
    names = list(annotators)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = compare(a, annotators[a]["labels"], b, annotators[b]["labels"])
            pairs.append(r)
            if not r.get("n"):
                print(f"\n  {r['pair']}: {r.get('note')}")
                continue
            print(f"\n  {r['pair']}  (n={r['n']})")
            print(f"    outcome   : {r['raw_agreement']:.1%} raw, kappa {r['kappa']}")
            if "worse_kind_kappa" in r:
                print(f"    worse kind: {r['worse_kind_agreement']:.1%} raw, "
                      f"kappa {r['worse_kind_kappa']}  (n={r['n_both_worse']})")

    print("\n" + "=" * 70)
    print("WHAT THIS DOES AND DOES NOT SHOW")
    print("=" * 70)
    print("""
  heuristic vs LLM  -- two methodologically independent methods. One counts
                       tokens and hedge words; the other reads the evidence.
                       Agreement here IS evidence the categories are real.

  LLM vs LLM        -- mostly a statement about language models. Two models
                       can agree perfectly and both be wrong the same way.
                       Report it as automatic agreement, never as validation.

  vs human          -- the only comparison that establishes validity. It does
                       NOT need 300 items; 50 from one person is enough to say
                       whether the automatic labels track human judgement.
                       State the n.""")

    if "human" not in annotators:
        print("\n  NO HUMAN LABELS SUPPLIED. Every number above is automatic-only.")
        print("  Label ~50 attempts yourself and pass --human to anchor them.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "run": str(run),
            "n_attempts": len(attempts),
            "judge": batch["annotator"],
            "n_unparseable": batch["n_unparseable"],
            "annotator_kinds": {k: v["kind"] for k, v in annotators.items()},
            "agreement": pairs,
            "labels": {k: v["labels"] for k, v in annotators.items()},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
