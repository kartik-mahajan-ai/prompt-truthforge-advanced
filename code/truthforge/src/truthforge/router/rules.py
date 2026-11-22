"""The rule router, plus the comparison arms it has to beat.

The ordering of the rules below is the algorithm. It is written as a flat,
commented cascade rather than a scoring function on purpose: every branch must
be nameable in the paper and countable in the error analysis.

    0. not flagged                    -> ACCEPT   (spend nothing)
    1. attempt budget exhausted       -> ABSTAIN  ("two attempts failed")
    1b. last rewrite changed nothing  -> ABSTAIN  (a retry is deterministic)
    2. sources contradict each other  -> ABSTAIN  (no rewrite can fix this)
    3. detector near the boundary     -> ACCEPT   (do not pay for a coin flip)
    4. evidence thin / missing        -> RERETRIEVE
    5. evidence present, low support  -> REWRITE
    6. anything left                  -> ACCEPT   (conservative default)

Rule 3 is the over-correction lever and the main ablation: widening
`uncertain_band` trades recall for a lower worse-rate. Rule 4 before rule 5 is
the other substantive claim -- that "go look again" beats "reword it" when the
right evidence was never retrieved.
"""

from __future__ import annotations

from typing import Optional

from ..config import RouterConfig
from ..types import Action, Passage, RoutingDecision
from .base import ClaimState, decision


class RuleRouter:
    """Threshold cascade over detector confidence, coverage and contradiction."""

    name = "rule"

    def __init__(self, cfg: Optional[RouterConfig] = None, max_attempts: int = 2) -> None:
        self.cfg = cfg or RouterConfig()
        self.max_attempts = max_attempts

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        c = self.cfg
        p = state.verdict.support_prob

        # 0. The detector is happy. Nothing to decide.
        if not state.verdict.is_flagged:
            return decision(
                state, Action.ACCEPT, "not_flagged", self.name
            )

        # 1. Budget. Two failed attempts means we stop claiming to know.
        if state.attempts >= self.max_attempts:
            return decision(
                state, Action.ABSTAIN, "attempt_budget_exhausted", self.name
            )

        # 1b. The last rewrite returned the claim unchanged. Decoding is
        #     greedy, so running the same corrector over the same evidence
        #     again is deterministic: the retry cannot produce anything new.
        #     Retrying would burn tokens to leave a claim we KNOW is wrong
        #     standing as fact. Abstain instead -- saying "I could not verify
        #     this" is the honest output when the corrector cannot fix it.
        if state.last_rewrite_failed and state.reretrievals == 0:
            return decision(
                state, Action.ABSTAIN, "corrector_could_not_fix_it", self.name
            )

        # 2. The passages disagree with each other. Rewriting would just pick a
        #    side with no basis for the choice.
        if state.contradiction >= c.contradiction_margin:
            return decision(
                state, Action.ABSTAIN, "contradictory_sources", self.name
            )

        # 3. The detector is barely on the unsupported side. Correcting here is
        #    where over-correction comes from: we would be rewriting a claim
        #    that is probably fine. Accept and record that we chose not to pay.
        if abs(p - 0.5) < c.uncertain_band:
            return decision(
                state, Action.ACCEPT, "detector_uncertain_not_worth_compute", self.name
            )

        # 4. The claim's content words are mostly absent from the evidence, so
        #    the retriever -- not the generator -- is the failure. Rewriting
        #    against absent evidence can only produce hedging or invention.
        if state.coverage < c.thin_evidence_coverage:
            if state.reretrievals > 0:
                # We already widened the search once and it did not help.
                return decision(
                    state, Action.ABSTAIN, "reretrieval_failed_evidence_still_thin", self.name
                )
            return decision(
                state, Action.RERETRIEVE, "thin_evidence_coverage", self.name
            )

        # 5. The evidence is there and the detector is confident the claim
        #    misstates it. This is the case rewriting is actually for.
        if p <= c.rewrite_support_prob:
            return decision(
                state, Action.REWRITE, "evidence_present_claim_misstates", self.name
            )

        # 6. Flagged, but not confidently, and evidence is adequate. Default to
        #    leaving it alone -- the whole point of the project is that this is
        #    a defensible choice.
        return decision(
            state, Action.ACCEPT, "flagged_but_below_action_threshold", self.name
        )


class AlwaysCorrectRouter:
    """The arm everyone else in the literature runs.

    Every flagged claim gets rewritten until the budget runs out. This is the
    baseline the over-correction numbers are measured against, so it must use
    the SAME detector and the SAME corrector as the rule router -- only the
    routing differs.
    """

    name = "always_correct"

    def __init__(self, max_attempts: int = 2) -> None:
        self.max_attempts = max_attempts

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        if not state.verdict.is_flagged:
            return decision(state, Action.ACCEPT, "not_flagged", self.name)
        if state.attempts >= self.max_attempts:
            return decision(state, Action.ACCEPT, "budget_exhausted_keep_last", self.name)
        return decision(state, Action.REWRITE, "always_correct", self.name)


class NoopRouter:
    """Vanilla RAG: detect, log, change nothing. Gives the detection numbers
    and the zero-correction cost floor in the same run."""

    name = "noop"

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        return decision(state, Action.ACCEPT, "noop", self.name)


class OracleRouter:
    """Upper bound: picks the action that a prior run showed to be best.

    Requires an auto-label map built by `scripts/build_router_labels.py`, which
    runs every action on every flagged claim and records which one produced a
    BETTER outcome (the Adaptive-RAG auto-labelling trick -- no human
    annotation needed). Falls back to the rule router for unseen claims, and
    counts those fallbacks so the oracle row can be honestly footnoted.
    """

    name = "oracle"

    def __init__(
        self,
        label_map: dict[str, str],
        fallback: Optional[RuleRouter] = None,
    ) -> None:
        self.label_map = label_map
        self.fallback = fallback or RuleRouter()
        # Counted over FLAGGED claims only. The auto-labeller labels flagged
        # claims and nothing else, so an unflagged claim with no label is not
        # a miss -- there was never anything for the oracle to decide. Counting
        # it would make a correctly working oracle look broken, since most
        # claims are unflagged.
        self.misses = 0
        self.flagged = 0

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        # Look up by claim TEXT, falling back to id. Claim ids are UUIDs minted
        # fresh in every process (`types._new_id`), and the labels are always
        # produced by a different process than the one running the oracle -- so
        # an id lookup misses every time, every decision falls through to the
        # rule router, and the "oracle upper bound" row is silently a copy of
        # the rule router. The caller passes one example's labels at a time,
        # so text is unique within the map.
        key = self.label_map.get(state.claim_text) or self.label_map.get(state.claim_id)
        flagged = state.verdict.is_flagged
        if flagged:
            self.flagged += 1
        if key is None:
            if flagged:
                self.misses += 1
            d = self.fallback.route(state, passages)
            d.router = self.name
            d.reason = f"oracle_miss:{d.reason}"
            return d
        return decision(state, Action(key), "oracle_label", self.name)


def build_router(
    cfg,
    label_map: Optional[dict[str, str]] = None,
    fixed_evidence: bool = False,
):
    """Resolve `cfg.router.name` to a router instance.

    `fixed_evidence` is true on datasets where the passages are given and
    cannot be widened (RAGTruth). CRAG's Incorrect branch is a live web
    search, so it has nowhere to go there and must degrade explicitly
    rather than silently behaving like a different policy.
    """
    max_attempts = cfg.loop.max_attempts_per_claim
    name = cfg.router.name
    if name == "rule":
        return RuleRouter(cfg.router, max_attempts=max_attempts)
    if name == "always_correct":
        return AlwaysCorrectRouter(max_attempts=max_attempts)
    if name == "noop":
        return NoopRouter()
    if name == "oracle":
        if label_map is None:
            raise ValueError("oracle router needs a label_map")
        return OracleRouter(label_map, RuleRouter(cfg.router, max_attempts=max_attempts))
    if name == "learned":
        from .learned import LearnedRouter

        return LearnedRouter.from_config(cfg)

    # Self-RAG and CRAG, reimplemented as policies over the same detector and
    # corrector so a difference in the ledger is a difference in routing. See
    # router/literature.py for what the translation does and does not preserve.
    from .literature import build_literature_router

    lit = build_literature_router(
        name,
        cfg,
        max_attempts=max_attempts,
        **({"web_search_available": not fixed_evidence} if name == "crag" else {}),
    )
    if lit is not None:
        return lit
    raise ValueError(f"unknown router: {name!r}")
