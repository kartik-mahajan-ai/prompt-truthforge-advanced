"""Core data types shared by every stage of the TruthForge loop.

Everything is a plain dataclass with `to_dict`, so a whole run serialises to
JSONL without a schema library. Keep it that way -- the run logs are the paper.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@dataclass
class Passage:
    """A retrieved (or given) piece of evidence."""

    doc_id: str
    text: str
    score: float = 0.0
    source: str = "unknown"  # "bm25" | "dense" | "rrf" | "given"
    rank: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Claims and verdicts
# --------------------------------------------------------------------------


@dataclass
class Claim:
    """One standalone, atomic assertion extracted from an answer."""

    text: str
    claim_id: str = field(default_factory=lambda: _new_id("clm"))
    # Character span in the answer this claim came from. Needed to splice
    # rewrites back in and to score against RAGTruth word-level spans.
    start: Optional[int] = None
    end: Optional[int] = None
    sent_index: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Label(str, enum.Enum):
    """Detector output for a single claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass
class Verdict:
    """A detector's judgement on one claim against one evidence set."""

    claim_id: str
    label: Label
    # P(supported). Calibration matters: the router thresholds on this.
    support_prob: float
    detector: str
    # Passage doc_ids the detector leaned on, when it can say.
    cited_doc_ids: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_flagged(self) -> bool:
        return self.label is Label.UNSUPPORTED

    @property
    def confidence(self) -> float:
        """Distance from the decision boundary, in [0, 0.5]."""
        return abs(self.support_prob - 0.5)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label.value
        return d


# --------------------------------------------------------------------------
# Routing -- the research object
# --------------------------------------------------------------------------


class Action(str, enum.Enum):
    ACCEPT = "accept"
    RERETRIEVE = "reretrieve"
    REWRITE = "rewrite"
    ABSTAIN = "abstain"


@dataclass
class RoutingDecision:
    """Why a claim went where it went.

    The `reason` field is load-bearing: the paper's error analysis is a
    groupby on it.
    """

    claim_id: str
    action: Action
    reason: str
    router: str  # "rule" | "learned" | "oracle" | "always_correct"
    # Router's own confidence, when it has one (learned router only).
    action_probs: dict[str, float] = field(default_factory=dict)
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


# --------------------------------------------------------------------------
# Correction outcomes -- the contribution
# --------------------------------------------------------------------------


class Outcome(str, enum.Enum):
    """Did the correction attempt help?"""

    BETTER = "better"
    SAME = "same"
    WORSE = "worse"


class WorseKind(str, enum.Enum):
    """Only set when Outcome.WORSE.

    This taxonomy is the novel artifact -- do not add categories without
    re-running inter-annotator agreement.
    """

    INFO_LOSS = "info_loss"      # shorter; dropped correct detail
    HEDGING = "hedging"          # a clear answer became vague mush
    NEW_ERROR = "new_error"      # the rewrite hallucinated something fresh

    @classmethod
    def values(cls) -> list[str]:
        return [k.value for k in cls]


# --------------------------------------------------------------------------
# Cost -- a first-class result, not a footnote
# --------------------------------------------------------------------------


@dataclass
class CostRecord:
    """Resource spend attributable to one stage or one attempt."""

    stage: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    retrieval_calls: int = 0
    llm_calls: int = 0
    detector_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class CorrectionAttempt:
    """One (claim, action) attempt and what it did to the answer.

    `outcome` / `worse_kind` are filled by `truthforge.eval.outcomes` -- either
    automatically (heuristic, for scale) or by a human annotator (gold, for the
    300-item artifact). `outcome_source` says which, and the two must never be
    silently mixed in a results table.
    """

    claim_id: str
    action: Action
    iteration: int
    before_text: str
    after_text: str
    before_support_prob: float
    after_support_prob: Optional[float] = None
    outcome: Optional[Outcome] = None
    worse_kind: Optional[WorseKind] = None
    outcome_source: Optional[str] = None  # "heuristic" | "human" | "oracle"
    annotator_id: Optional[str] = None
    notes: str = ""
    cost: Optional[CostRecord] = None
    # Set when the corrector's output was malformed and thrown away (e.g. it
    # echoed the whole answer instead of one sentence). A rejected attempt
    # still costs tokens, so it stays in the ledger and in the attempt list.
    rejection: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.before_text.strip() != self.after_text.strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["outcome"] = self.outcome.value if self.outcome else None
        d["worse_kind"] = self.worse_kind.value if self.worse_kind else None
        return d


# --------------------------------------------------------------------------
# End-to-end record
# --------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Everything one question produced. One of these per row of the run log."""

    question: str
    final_answer: str
    initial_answer: str
    turn_id: str = field(default_factory=lambda: _new_id("turn"))
    dataset: str = "unknown"
    example_id: Optional[str] = None
    passages: list[Passage] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    decisions: list[RoutingDecision] = field(default_factory=list)
    attempts: list[CorrectionAttempt] = field(default_factory=list)
    costs: list[CostRecord] = field(default_factory=list)
    abstained: bool = False
    iterations_used: int = 0
    wall_clock_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "dataset": self.dataset,
            "example_id": self.example_id,
            "question": self.question,
            "initial_answer": self.initial_answer,
            "final_answer": self.final_answer,
            "abstained": self.abstained,
            "iterations_used": self.iterations_used,
            "wall_clock_s": self.wall_clock_s,
            "created_at": self.created_at,
            "passages": [p.to_dict() for p in self.passages],
            "claims": [c.to_dict() for c in self.claims],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "decisions": [d.to_dict() for d in self.decisions],
            "attempts": [a.to_dict() for a in self.attempts],
            "costs": [c.to_dict() for c in self.costs],
            "meta": self.meta,
        }
