"""Correction-outcome classification -- the contribution.

Every correction attempt is labelled BETTER / SAME / WORSE, and every WORSE is
given a kind: INFO_LOSS, HEDGING, or NEW_ERROR.

Two label sources, and they must never be mixed in one column of a results
table:

  * ``heuristic`` -- `classify_attempt` below. Cheap, runs on every attempt in
    every arm. It is a PROXY, and its only claim to validity is its agreement
    with the human labels, which `agreement_report` computes.
  * ``human``     -- the ~300 double-annotated attempts. This is the artifact
    the project is actually shipping. `annotation/GUIDELINES.md` is its
    codebook; `export_for_annotation` writes the worksheet.

If heuristic-vs-human agreement is poor, report human numbers only and say so.
Do not tune the heuristic on the human labels and then report their agreement.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..types import CorrectionAttempt, Outcome, Passage, WorseKind

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
# Capitalised tokens not at sentence start: a cheap proxy for named entities.
_ENTITY = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b")

HEDGES = frozenset(
    """may might could possibly perhaps probably likely unlikely seems appears
    suggests indicates approximately roughly around about somewhat generally
    typically often sometimes usually reportedly allegedly arguably unclear
    uncertain potentially presumably supposedly""".split()
)

HEDGE_PHRASES = (
    "it is not clear",
    "it is unclear",
    "cannot be determined",
    "does not specify",
    "is not specified",
    "may or may not",
    "some sources",
    "in general",
)

_STOP = frozenset(
    """a an and are as at be been being but by for from has have had he her his
    i in is it its of on or she that the their them they this to was were what
    when where which who will with would you your not no nor do does did""".split()
)


def _content(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


def _hedge_count(text: str) -> int:
    toks = _WORD.findall(text.lower())
    n = sum(1 for t in toks if t in HEDGES)
    low = text.lower()
    n += sum(1 for p in HEDGE_PHRASES if p in low)
    return n


def _specifics(text: str) -> set[str]:
    """Numbers and entity-ish tokens -- the details a hedging rewrite drops."""
    return set(_NUMBER.findall(text)) | set(_ENTITY.findall(text))


@dataclass
class OutcomeThresholds:
    """Every heuristic cut-off in one place, so ablating them is a config diff.

    Defaults were chosen a priori from the definitions, NOT fitted to the
    human labels. If you change them, re-run `agreement_report`.
    """

    # Minimum change in P(supported) to count as a real move.
    support_delta: float = 0.10
    # Length ratio below which a rewrite counts as having shrunk.
    shrink_ratio: float = 0.80
    # Share of the original's supported content tokens that may vanish.
    max_dropped_supported: float = 0.20
    # Extra hedge terms that turn a shortening into hedging.
    hedge_increase: int = 1
    # Share of new content tokens absent from the evidence that signals
    # a fresh hallucination.
    new_unsupported_ratio: float = 0.30


def classify_attempt(
    attempt: CorrectionAttempt,
    passages: Sequence[Passage],
    thresholds: Optional[OutcomeThresholds] = None,
) -> CorrectionAttempt:
    """Fill `outcome`, `worse_kind` and `outcome_source='heuristic'` in place.

    Order matters: the WORSE checks run before the support-probability check,
    because a rewrite can raise P(supported) precisely BY deleting the
    contested detail or hedging it away. A detector-only criterion would score
    that as a success, which is the exact blind spot this project is about.
    """
    t = thresholds or OutcomeThresholds()
    before, after = attempt.before_text.strip(), attempt.after_text.strip()
    evidence = " ".join(p.text for p in passages)
    ev_tokens = _content(evidence)

    if before == after:
        attempt.outcome = Outcome.SAME
        attempt.worse_kind = None
        attempt.outcome_source = "heuristic"
        attempt.notes = "no textual change"
        return attempt

    b_tokens, a_tokens = _content(before), _content(after)
    dropped = b_tokens - a_tokens
    added = a_tokens - b_tokens

    # Content that the evidence supported and the rewrite threw away.
    dropped_supported = dropped & ev_tokens
    supported_before = b_tokens & ev_tokens
    drop_share = (
        len(dropped_supported) / len(supported_before) if supported_before else 0.0
    )

    hedge_delta = _hedge_count(after) - _hedge_count(before)
    lost_specifics = _specifics(before) - _specifics(after)
    length_ratio = len(after) / len(before) if before else 1.0

    # New content words that appear nowhere in the evidence.
    new_unsupported = added - ev_tokens
    new_share = len(new_unsupported) / len(added) if added else 0.0

    delta = (
        (attempt.after_support_prob - attempt.before_support_prob)
        if attempt.after_support_prob is not None
        else 0.0
    )

    # --- WORSE, checked before any improvement claim -----------------------

    # 1. Hedging: got vaguer, and lost the specifics that made it useful.
    if hedge_delta >= t.hedge_increase and (lost_specifics or length_ratio < 1.2):
        return _mark_worse(
            attempt,
            WorseKind.HEDGING,
            f"hedge terms +{hedge_delta}; lost specifics {sorted(lost_specifics)[:4]}",
        )

    # 2. Info loss: shrank and dropped detail the evidence backed.
    if length_ratio < t.shrink_ratio and drop_share > t.max_dropped_supported:
        return _mark_worse(
            attempt,
            WorseKind.INFO_LOSS,
            f"len {length_ratio:.2f}x; dropped {drop_share:.0%} of supported content",
        )

    # 3. New error: invented content words with no basis in the evidence.
    if new_unsupported and new_share > t.new_unsupported_ratio:
        return _mark_worse(
            attempt,
            WorseKind.NEW_ERROR,
            f"{new_share:.0%} of added content absent from evidence: "
            f"{sorted(new_unsupported)[:4]}",
        )

    # 4. The detector got materially less happy.
    if delta <= -t.support_delta:
        return _mark_worse(
            attempt, WorseKind.NEW_ERROR, f"support_prob fell {delta:+.2f}"
        )

    # --- BETTER / SAME -----------------------------------------------------

    if delta >= t.support_delta:
        attempt.outcome = Outcome.BETTER
        attempt.notes = f"support_prob rose {delta:+.2f} with no detected damage"
    else:
        attempt.outcome = Outcome.SAME
        attempt.notes = f"text changed but support_prob moved only {delta:+.2f}"

    attempt.worse_kind = None
    attempt.outcome_source = "heuristic"
    return attempt


def _mark_worse(
    attempt: CorrectionAttempt, kind: WorseKind, note: str
) -> CorrectionAttempt:
    attempt.outcome = Outcome.WORSE
    attempt.worse_kind = kind
    attempt.outcome_source = "heuristic"
    attempt.notes = note
    return attempt


# --------------------------------------------------------------------------
# Human annotation: worksheet out, labels in, agreement report
# --------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "attempt_key",
    "claim_id",
    "action",
    "iteration",
    "before_text",
    "after_text",
    "evidence",
    "outcome",       # better | same | worse   <- annotator fills
    "worse_kind",    # info_loss | hedging | new_error  <- if worse
    "annotator_id",
    "notes",
]


def export_for_annotation(
    attempts: Iterable[CorrectionAttempt],
    passages_by_claim: dict[str, Sequence[Passage]],
    path: str | Path,
    evidence_chars: int = 3000,
    blind: bool = True,
) -> int:
    """Write the annotation worksheet.

    `blind=True` (the default) withholds the detector's probabilities and the
    heuristic label from the annotator. Showing them would anchor the human
    labels to the thing they are supposed to validate.
    """
    rows = []
    for i, a in enumerate(attempts):
        ev = " ".join(p.text for p in passages_by_claim.get(a.claim_id, []))
        row = {
            "attempt_key": f"{a.claim_id}:{a.iteration}:{a.action.value}",
            "claim_id": a.claim_id,
            "action": a.action.value,
            "iteration": a.iteration,
            "before_text": a.before_text,
            "after_text": a.after_text,
            "evidence": ev[:evidence_chars],
            "outcome": "" if blind else (a.outcome.value if a.outcome else ""),
            "worse_kind": "" if blind else (a.worse_kind.value if a.worse_kind else ""),
            "annotator_id": "",
            "notes": "",
        }
        rows.append(row)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ANNOTATION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def load_human_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Read a completed worksheet into {attempt_key: {outcome, worse_kind, ...}}."""
    out: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            outcome = (row.get("outcome") or "").strip().lower()
            if not outcome:
                continue
            if outcome not in {o.value for o in Outcome}:
                raise ValueError(
                    f"{path}: bad outcome {outcome!r} for {row.get('attempt_key')}"
                )
            kind = (row.get("worse_kind") or "").strip().lower() or None
            if outcome == Outcome.WORSE.value and kind not in set(WorseKind.values()):
                raise ValueError(
                    f"{path}: WORSE row {row.get('attempt_key')} needs a worse_kind "
                    f"in {WorseKind.values()}, got {kind!r}"
                )
            if outcome != Outcome.WORSE.value and kind:
                raise ValueError(
                    f"{path}: non-WORSE row {row.get('attempt_key')} must leave "
                    "worse_kind empty"
                )
            out[row["attempt_key"]] = {
                "outcome": outcome,
                "worse_kind": kind or "",
                "annotator_id": (row.get("annotator_id") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    return out


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted Cohen's kappa. Returns nan for fewer than two items."""
    if len(a) != len(b):
        raise ValueError("label sequences must be the same length")
    n = len(a)
    if n < 2:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def agreement_report(
    labels_a: dict[str, dict[str, str]],
    labels_b: dict[str, dict[str, str]],
    heuristic: Optional[dict[str, str]] = None,
) -> dict[str, object]:
    """Inter-annotator agreement, and heuristic validity against the humans.

    `heuristic` maps attempt_key -> outcome string. Heuristic agreement is
    reported against annotator A and annotator B separately; a heuristic that
    agrees with one annotator far more than the other is overfit to a style,
    not validated.
    """
    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        return {"n_shared": 0, "note": "no overlapping attempt_keys"}

    oa = [labels_a[k]["outcome"] for k in shared]
    ob = [labels_b[k]["outcome"] for k in shared]

    worse_keys = [
        k
        for k in shared
        if labels_a[k]["outcome"] == Outcome.WORSE.value
        and labels_b[k]["outcome"] == Outcome.WORSE.value
    ]

    report: dict[str, object] = {
        "n_shared": len(shared),
        "outcome_raw_agreement": sum(1 for x, y in zip(oa, ob) if x == y) / len(shared),
        "outcome_kappa": cohens_kappa(oa, ob),
        "outcome_dist_a": dict(Counter(oa)),
        "outcome_dist_b": dict(Counter(ob)),
        "n_both_worse": len(worse_keys),
    }

    if worse_keys:
        ka = [labels_a[k]["worse_kind"] for k in worse_keys]
        kb = [labels_b[k]["worse_kind"] for k in worse_keys]
        report["worse_kind_raw_agreement"] = sum(
            1 for x, y in zip(ka, kb) if x == y
        ) / len(worse_keys)
        report["worse_kind_kappa"] = cohens_kappa(ka, kb)
        report["worse_kind_dist_a"] = dict(Counter(ka))
        report["worse_kind_dist_b"] = dict(Counter(kb))

    if heuristic:
        for tag, labels in (("a", labels_a), ("b", labels_b)):
            keys = [k for k in shared if k in heuristic]
            if not keys:
                continue
            h = [heuristic[k] for k in keys]
            g = [labels[k]["outcome"] for k in keys]
            report[f"heuristic_vs_{tag}_agreement"] = sum(
                1 for x, y in zip(h, g) if x == y
            ) / len(keys)
            report[f"heuristic_vs_{tag}_kappa"] = cohens_kappa(h, g)

    return report
