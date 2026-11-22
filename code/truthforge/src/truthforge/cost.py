"""Cost accounting.

Cost is a headline result here, not a footnote, so every stage that spends
anything must go through `CostLedger.track`. If a stage is missing from the
ledger the results table is wrong, and nothing else will tell you.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .types import CostRecord


@dataclass
class CostLedger:
    """Accumulates `CostRecord`s for one turn (or one whole run)."""

    records: list[CostRecord] = field(default_factory=list)

    @contextmanager
    def track(self, stage: str) -> Iterator[CostRecord]:
        """Time a stage and append its record.

        The body mutates the yielded record with token counts and call counts;
        latency is filled in automatically.

            with ledger.track("generate") as c:
                out = llm(prompt)
                c.prompt_tokens, c.completion_tokens = out.usage
                c.llm_calls = 1
        """
        rec = CostRecord(stage=stage)
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec.latency_s = time.perf_counter() - t0
            self.records.append(rec)

    def add(self, record: CostRecord) -> None:
        self.records.append(record)

    def extend(self, records: list[CostRecord]) -> None:
        self.records.extend(records)

    # -- aggregation -------------------------------------------------------

    def total(self, stage: Optional[str] = None) -> CostRecord:
        """Sum across records, optionally filtered to one stage."""
        rows = [r for r in self.records if stage is None or r.stage == stage]
        return CostRecord(
            stage=stage or "total",
            prompt_tokens=sum(r.prompt_tokens for r in rows),
            completion_tokens=sum(r.completion_tokens for r in rows),
            latency_s=sum(r.latency_s for r in rows),
            retrieval_calls=sum(r.retrieval_calls for r in rows),
            llm_calls=sum(r.llm_calls for r in rows),
            detector_calls=sum(r.detector_calls for r in rows),
        )

    def by_stage(self) -> dict[str, CostRecord]:
        return {s: self.total(s) for s in sorted({r.stage for r in self.records})}

    def summary(self) -> dict[str, float]:
        t = self.total()
        return {
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.total_tokens,
            "latency_s": round(t.latency_s, 4),
            "retrieval_calls": t.retrieval_calls,
            "llm_calls": t.llm_calls,
            "detector_calls": t.detector_calls,
        }


def count_tokens(text: str, tokenizer: object | None = None) -> int:
    """Token count for cost accounting.

    With a HF tokenizer, exact. Without one, a ~4-chars-per-token estimate --
    fine for relative comparisons between arms, but never report estimated
    counts in the paper without saying so.
    """
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))  # type: ignore[attr-defined]
    return max(1, round(len(text) / 4)) if text else 0
