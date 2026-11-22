"""Detector interface.

A detector maps (claim, evidence) -> Verdict with a calibrated-ish
P(supported). The router thresholds on that probability, so a detector that
only emits 0.0/1.0 collapses the router to two actions. Any new detector must
return a graded score.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..types import Claim, Label, Passage, Verdict


class Detector(ABC):
    """Base class. Subclasses implement `score_batch`; `check` handles the
    bookkeeping (timing, thresholding, Verdict construction)."""

    name: str = "base"

    def __init__(self, flag_threshold: float = 0.5) -> None:
        self.flag_threshold = flag_threshold

    @abstractmethod
    def score_batch(
        self, claims: list[Claim], passages: list[Passage]
    ) -> list[float]:
        """Return P(supported) in [0, 1] for each claim against the evidence."""

    def check(self, claims: list[Claim], passages: list[Passage]) -> list[Verdict]:
        if not claims:
            return []
        t0 = time.perf_counter()
        probs = self.score_batch(claims, passages)
        if len(probs) != len(claims):
            raise RuntimeError(
                f"{self.name}: got {len(probs)} scores for {len(claims)} claims"
            )
        per_claim_latency = (time.perf_counter() - t0) / len(claims)

        return [
            Verdict(
                claim_id=c.claim_id,
                label=(
                    Label.SUPPORTED if p >= self.flag_threshold else Label.UNSUPPORTED
                ),
                support_prob=float(p),
                detector=self.name,
                cited_doc_ids=[pas.doc_id for pas in passages],
                latency_s=per_claim_latency,
            )
            for c, p in zip(claims, probs)
        ]


def evidence_text(passages: list[Passage], max_chars: int | None = None) -> str:
    """Join passages into one evidence block, oldest-rank first."""
    ordered = sorted(passages, key=lambda p: (p.rank if p.rank is not None else 0))
    joined = "\n\n".join(p.text for p in ordered)
    return joined[:max_chars] if max_chars else joined
