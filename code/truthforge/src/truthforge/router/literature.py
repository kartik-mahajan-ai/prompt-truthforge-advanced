"""Self-RAG and CRAG as routing policies, so the comparison is about routing.

Both papers ship a trained component we cannot run here: Self-RAG's critic is a
7B Llama-2 fine-tune with special reflection tokens, CRAG's retrieval evaluator
is a fine-tuned T5-large plus a live web-search fallback. Running either at
full size on an 8 GB laptop is not possible, and running a smaller stand-in and
calling it "Self-RAG" would be worse than not running it.

So these are **policy reimplementations, not model reimplementations**, and the
distinction is load-bearing. Each one reuses TruthForge's own detector and
corrector and changes only the decision rule. That is deliberate: if the
baselines used different models, a win could come from model capacity, and the
claim of this project is specifically about the decision. Holding the detector
and corrector fixed is the only way the comparison isolates it.

What that buys, and what it costs:

  buys  -- a like-for-like row. Differences in the ledger are differences in
           policy, full stop.
  costs -- these rows are NOT reproductions of the published numbers, and must
           never be presented as such. The paper reports them as
           "Self-RAG-style policy" and "CRAG-style policy" with this paragraph
           as the footnote.

What each policy actually does, and where the translation is lossy:

Self-RAG (Asai et al., ICLR 2024)
    Generates one candidate segment per retrieved passage plus a no-retrieval
    candidate, critiques each with IsREL / IsSUP / IsUSE, and emits the best by
    weighted score. Two properties survive translation exactly: it spends
    corrector calls on EVERY segment it is unsure about, proportional to k, and
    it never abstains -- the lowest-scoring candidate is still emitted. Those
    are the two things TruthForge claims to improve on, so they are the two
    things this router reproduces. `IsSUP` is three-way (full / partial / none)
    and maps onto the detector's support probability with two thresholds.

    Lossy: real Self-RAG's critic was trained to emit those tokens, so its
    partial/full boundary is learned, not thresholded. Stated in the paper.

CRAG (Yan et al., 2024)
    A retrieval evaluator grades the retrieved set Correct / Ambiguous /
    Incorrect, then: Correct -> refine and keep, Incorrect -> discard the
    retrieval and web-search instead, Ambiguous -> do both. The structural
    difference from TruthForge is the unit of decision: CRAG decides about the
    EVIDENCE, once per query, and TruthForge decides about each CLAIM. That is
    the comparison worth showing, and it survives translation intact.

    Lossy, and seriously: CRAG's Incorrect branch is a live web search. On
    RAGTruth the evidence is fixed, so there is nothing to search -- the branch
    degrades to RERETRIEVE where a retriever exists and ABSTAIN where it does
    not. CRAG's authors would call that a crippled condition and they would be
    right, so the fixed-evidence rows for CRAG carry an explicit footnote and
    the document-store rows (where RERETRIEVE is real) are the fair ones.
"""

from __future__ import annotations

from typing import Optional

from ..config import RouterConfig
from ..types import Action, Passage, RoutingDecision
from .base import ClaimState, decision


class SelfRAGRouter:
    """Critique-and-select: correct everything unsure, never abstain.

    Self-RAG's inference loop has no "leave it alone because fixing it is not
    worth it" branch and no "say I don't know" branch. Every segment below the
    full-support bar gets regenerated, and whatever scores best is emitted. In
    this action space that is REWRITE on anything short of fully supported, and
    ACCEPT -- never ABSTAIN -- when the budget runs out.

    `candidates_per_claim` records how many corrector calls real Self-RAG would
    have spent on this claim (one per retrieved passage). The loop only runs
    one, so the cost column reports both: what we measured and what the policy
    as published would have cost. Quietly reporting the one-call figure would
    make Self-RAG look cheaper than it is.
    """

    name = "self_rag"

    # Self-RAG's IsSUP is three-way. These are the two cut points.
    FULL_SUPPORT = 0.70
    NO_SUPPORT = 0.30

    def __init__(self, max_attempts: int = 2, candidates_per_claim: int = 5) -> None:
        self.max_attempts = max_attempts
        self.candidates_per_claim = candidates_per_claim
        # Corrector calls the published policy would have made, for the cost row.
        self.notional_calls = 0

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        p = state.verdict.support_prob

        # IsSUP = fully supported. Emit as generated.
        if not state.verdict.is_flagged and p >= self.FULL_SUPPORT:
            return decision(state, Action.ACCEPT, "is_sup_full", self.name)

        if state.attempts >= self.max_attempts:
            # No abstain token exists. The best candidate is emitted whatever
            # its score -- this is where Self-RAG's unsupported output comes
            # from, and the ledger should see it as a kept false claim.
            return decision(state, Action.ACCEPT, "budget_spent_emit_best", self.name)

        # IsSUP = partial or none -> regenerate from the evidence and re-critique.
        self.notional_calls += max(1, min(self.candidates_per_claim, len(passages) or 1))
        reason = "is_sup_none" if p <= self.NO_SUPPORT else "is_sup_partial"
        return decision(state, Action.REWRITE, reason, self.name)


class CRAGRouter:
    """Evidence-level triage: grade the retrieval, then act on the whole query.

    CRAG's decision is made once per query about the retrieved set, not per
    claim. Reproducing that faithfully means this router must ignore how
    confident the detector is about THIS claim and look at the evidence grade
    instead -- which is exactly the behaviour the comparison is meant to expose.
    A query whose evidence grades Correct gets every claim accepted, including
    the one claim that misstates it.

    The grade comes from coverage against the retrieved set, which is the same
    signal CRAG's evaluator was trained to produce, computed without a trained
    model. `web_search_available` stands for CRAG's Incorrect branch: true when
    a retriever can actually widen the search, false on RAGTruth where the
    evidence is frozen and the branch has nowhere to go.
    """

    name = "crag"

    def __init__(
        self,
        cfg: Optional[RouterConfig] = None,
        max_attempts: int = 2,
        upper: float = 0.60,
        lower: float = 0.25,
        web_search_available: bool = True,
    ) -> None:
        self.cfg = cfg or RouterConfig()
        self.max_attempts = max_attempts
        self.upper = upper
        self.lower = lower
        self.web_search_available = web_search_available
        # Counted so the paper can say how often the crippled branch fired.
        self.degraded_incorrect_branches = 0

    def _grade(self, state: ClaimState) -> str:
        if state.coverage >= self.upper:
            return "correct"
        if state.coverage <= self.lower:
            return "incorrect"
        return "ambiguous"

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        grade = self._grade(state)

        if state.attempts >= self.max_attempts:
            return decision(state, Action.ACCEPT, "budget_exhausted", self.name)

        # Correct: the evidence is right, so keep the generation. Note that
        # this fires whether or not the DETECTOR flagged the claim -- that is
        # the point of the baseline, and the source of its false-keeps.
        if grade == "correct":
            return decision(state, Action.ACCEPT, "evidence_correct_refine", self.name)

        # Incorrect: discard this retrieval and search elsewhere.
        if grade == "incorrect":
            if not self.web_search_available:
                self.degraded_incorrect_branches += 1
                return decision(
                    state, Action.ABSTAIN, "evidence_incorrect_no_web_search", self.name
                )
            if state.reretrievals > 0:
                return decision(
                    state, Action.ABSTAIN, "evidence_incorrect_after_search", self.name
                )
            return decision(state, Action.RERETRIEVE, "evidence_incorrect", self.name)

        # Ambiguous: CRAG runs both branches and combines. The combination step
        # is a regeneration over the merged evidence, so: re-retrieve first if
        # we have not, then rewrite against what came back.
        if self.web_search_available and state.reretrievals == 0:
            return decision(state, Action.RERETRIEVE, "evidence_ambiguous_both", self.name)
        return decision(state, Action.REWRITE, "evidence_ambiguous_combine", self.name)


def build_literature_router(name: str, cfg, max_attempts: int = 2, **kw):
    """Resolve a literature-baseline name, or return None if it is not one."""
    if name == "self_rag":
        return SelfRAGRouter(max_attempts=max_attempts, **kw)
    if name == "crag":
        return CRAGRouter(cfg.router, max_attempts=max_attempts, **kw)
    return None
