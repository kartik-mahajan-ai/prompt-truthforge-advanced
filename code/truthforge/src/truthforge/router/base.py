"""Router interface and per-claim bookkeeping.

The router is the research object of this project. Everything else in the
repository exists to feed it inputs and measure its outputs, so this module
stays deliberately small and its state deliberately explicit: if you cannot
reconstruct why a claim was routed somewhere from `ClaimState` plus the config,
the error analysis in the paper will not be reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Action, Passage, RoutingDecision, Verdict


@dataclass
class ClaimState:
    """Everything the router knows about one claim at decision time."""

    claim_id: str
    claim_text: str
    verdict: Verdict
    # Fraction of the claim's content tokens present in current evidence.
    coverage: float
    # [0, 1] disagreement among passages that both look relevant.
    contradiction: float
    iteration: int = 0
    # Actions already spent on this claim, in order.
    history: list[Action] = field(default_factory=list)
    # Did the most recent REWRITE leave the text identical? Decoding is
    # greedy, so re-running the corrector on unchanged input is deterministic
    # -- the retry is guaranteed to fail the same way. Only a re-retrieval
    # (which changes the evidence) can make a retry worthwhile.
    last_rewrite_failed: bool = False

    @property
    def attempts(self) -> int:
        """Correction attempts so far. ACCEPT and ABSTAIN are terminal, not
        attempts, so they do not count against the budget."""
        return sum(1 for a in self.history if a in (Action.REWRITE, Action.RERETRIEVE))

    @property
    def rewrites(self) -> int:
        return sum(1 for a in self.history if a is Action.REWRITE)

    @property
    def reretrievals(self) -> int:
        return sum(1 for a in self.history if a is Action.RERETRIEVE)

    def features(self) -> dict[str, float]:
        """Feature vector for the learned router. Keep the keys stable --
        trained coefficient files are matched to these names."""
        p = self.verdict.support_prob
        return {
            "support_prob": p,
            "detector_confidence": abs(p - 0.5),
            "coverage": self.coverage,
            "contradiction": self.contradiction,
            "claim_len_tokens": float(len(self.claim_text.split())),
            "has_number": float(any(ch.isdigit() for ch in self.claim_text)),
            "iteration": float(self.iteration),
            "attempts": float(self.attempts),
            "prior_rewrite": float(self.rewrites > 0),
            "prior_reretrieve": float(self.reretrievals > 0),
            "last_rewrite_failed": float(self.last_rewrite_failed),
        }


class Router(Protocol):
    name: str

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision: ...


def decision(
    state: ClaimState, action: Action, reason: str, router: str, **kw
) -> RoutingDecision:
    return RoutingDecision(
        claim_id=state.claim_id,
        action=action,
        reason=reason,
        router=router,
        iteration=state.iteration,
        **kw,
    )
