"""Metrics, including the columns that make this project's table different.

Column definitions, fixed here so the paper and the code cannot drift:

  faithfulness         share of claims IN THE FINAL ANSWER judged supported;
                       abstained claims are excluded, so always report it
                       next to abstention_rate
  abstention_rate      share of turns whose final answer abstains on >=1 claim
  worse_rate           WORSE attempts / all correction attempts
                       (this is the headline "over-correction rate")
  futile_rate          (SAME + WORSE) / all attempts -- compute bought nothing
  rejection_rate       attempts whose corrector output was malformed and
                       discarded (e.g. it echoed the whole answer). These are
                       a corrector-capability confound, NOT evidence about
                       whether correcting was worth attempting -- report them
                       separately and never fold them into worse_rate.
  net_gain             (BETTER - WORSE) / all attempts
  tokens_per_turn      mean total tokens, ALL stages, including the detector
  latency_per_turn     mean wall clock, seconds

`worse_rate` is attempt-level, not turn-level, on purpose: a turn with five
attempts and one disaster should not score the same as a turn with one attempt
and one disaster. The turn-level variant is reported alongside as
`turns_harmed_rate`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from ..types import Action, CorrectionAttempt, Label, Outcome, TurnResult, WorseKind


# --------------------------------------------------------------------------
# Detector metrics -- Gate 1
# --------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        tpr = self.recall
        tnr = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
        return (tpr + tnr) / 2

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
        }


def detector_metrics(
    predicted: Sequence[Label], gold_unsupported: Sequence[bool]
) -> ClassificationMetrics:
    """Positive class = UNSUPPORTED (a hallucination the detector should catch)."""
    if len(predicted) != len(gold_unsupported):
        raise ValueError("predicted and gold must be the same length")
    tp = fp = tn = fn = 0
    for pred, gold in zip(predicted, gold_unsupported):
        flagged = pred is Label.UNSUPPORTED
        if flagged and gold:
            tp += 1
        elif flagged and not gold:
            fp += 1
        elif not flagged and gold:
            fn += 1
        else:
            tn += 1
    return ClassificationMetrics(len(predicted), tp, fp, tn, fn)


def auroc(scores: Sequence[float], gold_unsupported: Sequence[bool]) -> float:
    """Rank-based AUROC with tie handling. Score = P(unsupported), so pass
    `1 - support_prob`. Returns nan when one class is absent."""
    pairs = sorted(zip(scores, gold_unsupported), key=lambda x: x[0])
    n_pos = sum(1 for _, g in pairs if g)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie group
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    rank_sum_pos = sum(r for r, (_, g) in zip(ranks, pairs) if g)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compare_to_baselines(
    results: dict[str, ClassificationMetrics], primary: str = "minicheck"
) -> dict[str, object]:
    """Gate 1 verdict: does the primary detector clear every baseline?

    Uses balanced accuracy, because the class balance in RAGTruth makes plain
    accuracy flattering to a majority-class baseline.
    """
    if primary not in results:
        raise KeyError(f"{primary!r} not in results: {sorted(results)}")
    target = results[primary].balanced_accuracy
    others = {k: v.balanced_accuracy for k, v in results.items() if k != primary}
    best_other = max(others, key=others.__getitem__) if others else None
    margin = target - others[best_other] if best_other else float("nan")
    return {
        "primary": primary,
        "primary_balanced_acc": round(target, 4),
        "best_baseline": best_other,
        "best_baseline_balanced_acc": round(others[best_other], 4) if best_other else None,
        "margin": round(margin, 4) if best_other else None,
        "passes_gate_1": bool(best_other and margin >= 0.05),
    }


# --------------------------------------------------------------------------
# Correction-outcome metrics -- the contribution
# --------------------------------------------------------------------------


def correction_metrics(attempts: Sequence[CorrectionAttempt]) -> dict[str, object]:
    labelled = [a for a in attempts if a.outcome is not None]
    n = len(labelled)
    if n == 0:
        return {
            "n_attempts": 0,
            "worse_rate": 0.0,
            "futile_rate": 0.0,
            "net_gain": 0.0,
            "note": "no correction attempts (router never corrected)",
        }

    counts = Counter(a.outcome.value for a in labelled)
    kinds = Counter(
        a.worse_kind.value for a in labelled if a.worse_kind is not None
    )
    sources = Counter(a.outcome_source or "unknown" for a in labelled)
    by_action = {
        act.value: Counter(
            a.outcome.value for a in labelled if a.action is act
        )
        for act in (Action.REWRITE, Action.RERETRIEVE)
    }

    better = counts.get(Outcome.BETTER.value, 0)
    same = counts.get(Outcome.SAME.value, 0)
    worse = counts.get(Outcome.WORSE.value, 0)

    # Attempts the corrector botched badly enough that its output was thrown
    # away. These cost tokens and produce no change, so they inflate the
    # futile rate for a reason that is about the corrector rather than about
    # whether correcting was worth trying. Report them separately.
    rejected = [a for a in labelled if a.rejection]
    reject_reasons = Counter(
        str(a.rejection).split("(")[0] for a in rejected
    )

    return {
        "n_attempts": n,
        "n_rejected_outputs": len(rejected),
        "rejection_rate": round(len(rejected) / n, 4),
        "rejection_reasons": dict(reject_reasons),
        "better": better,
        "same": same,
        "worse": worse,
        "better_rate": round(better / n, 4),
        "same_rate": round(same / n, 4),
        "worse_rate": round(worse / n, 4),
        "futile_rate": round((same + worse) / n, 4),
        "net_gain": round((better - worse) / n, 4),
        "worse_kinds": {k: kinds.get(k, 0) for k in WorseKind.values()},
        "worse_kind_shares": {
            k: round(kinds.get(k, 0) / worse, 4) if worse else 0.0
            for k in WorseKind.values()
        },
        "outcome_by_action": {k: dict(v) for k, v in by_action.items()},
        "label_sources": dict(sources),
    }


# --------------------------------------------------------------------------
# System-level metrics -- one row of the results table
# --------------------------------------------------------------------------


def system_metrics(turns: Sequence[TurnResult]) -> dict[str, object]:
    if not turns:
        return {"n_turns": 0}

    n = len(turns)
    all_attempts = [a for t in turns for a in t.attempts]

    # Faithfulness is the share of claims IN THE FINAL ANSWER that the judge
    # marks supported -- so it uses the last verdict per claim, and excludes
    # claims the system abstained on, because those are no longer asserted.
    #
    # Counting an abstained claim as unsupported would penalise the system for
    # declining to assert something it could not verify, which is the exact
    # behaviour the abstain action exists to produce. `abstention_rate` below
    # reports how often that happened, so nothing is hidden: a system that
    # abstains on everything scores faithfulness 1.0 and abstention 1.0, and
    # the pair is obviously degenerate. Report them together, always.
    final_supported = final_total = 0
    for t in turns:
        abstained = set(t.meta.get("abstained_claim_ids", []))
        last: dict[str, bool] = {}
        for v in t.verdicts:
            if v.claim_id in abstained:
                continue
            last[v.claim_id] = v.label is Label.SUPPORTED
        final_supported += sum(1 for ok in last.values() if ok)
        final_total += len(last)

    actions = Counter(d.action.value for t in turns for d in t.decisions)
    reasons = Counter(d.reason for t in turns for d in t.decisions)

    turns_harmed = sum(
        1 for t in turns if any(a.outcome is Outcome.WORSE for a in t.attempts)
    )

    tokens = [sum(c.total_tokens for c in t.costs) for t in turns]
    latency = [t.wall_clock_s or sum(c.latency_s for c in t.costs) for t in turns]
    retrievals = [sum(c.retrieval_calls for c in t.costs) for t in turns]
    llm_calls = [sum(c.llm_calls for c in t.costs) for t in turns]

    return {
        "n_turns": n,
        "faithfulness": round(final_supported / final_total, 4) if final_total else 0.0,
        "abstention_rate": round(sum(1 for t in turns if t.abstained) / n, 4),
        "turns_harmed_rate": round(turns_harmed / n, 4),
        "mean_iterations": round(sum(t.iterations_used for t in turns) / n, 3),
        "tokens_per_turn": round(_mean(tokens), 1),
        "latency_per_turn": round(_mean(latency), 4),
        "retrieval_calls_per_turn": round(_mean(retrievals), 3),
        "llm_calls_per_turn": round(_mean(llm_calls), 3),
        "action_counts": dict(actions),
        "routing_reasons": dict(reasons.most_common()),
        **{f"correction_{k}": v for k, v in correction_metrics(all_attempts).items()},
    }


def results_row(
    arm: str, turns: Sequence[TurnResult], accuracy: Optional[float] = None
) -> dict[str, object]:
    """One row of the paper's main table.

    `accuracy` is task accuracy (EM/F1 against gold answers) and must be
    supplied by the dataset-specific evaluator -- RAGTruth has no gold answer
    to score against, so it is None there and the row is faithfulness-only.
    """
    m = system_metrics(turns)
    return {
        "arm": arm,
        "accuracy": accuracy,
        "faithfulness": m.get("faithfulness"),
        "over_correction_rate": m.get("correction_worse_rate"),
        "futile_rate": m.get("correction_futile_rate"),
        # A high rejection rate means the corrector could not follow the
        # single-sentence instruction, which inflates futile_rate for a reason
        # that has nothing to do with whether correcting was worth trying.
        # It belongs beside the other rates, not in a footnote.
        "rejection_rate": m.get("correction_rejection_rate"),
        "abstention_rate": m.get("abstention_rate"),
        "tokens": m.get("tokens_per_turn"),
        "latency_s": m.get("latency_per_turn"),
        "retrieval_calls": m.get("retrieval_calls_per_turn"),
        "n_turns": m.get("n_turns"),
    }


def format_table(rows: Sequence[dict[str, object]]) -> str:
    """Markdown table, ready to paste into the paper draft."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {
        c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols
    }
    head = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    body = [
        "| " + " | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
