"""The Self-RAG and CRAG policies must differ from ours in the ways claimed.

These are not tests that the baselines are *good*. They are tests that each
baseline reproduces the specific behaviour the paper says distinguishes it,
because a baseline that quietly collapses into the rule router would make the
comparison table meaningless while still passing a smoke test.
"""

from __future__ import annotations

from truthforge.config import Config
from truthforge.router.base import ClaimState
from truthforge.router.literature import CRAGRouter, SelfRAGRouter
from truthforge.router.rules import RuleRouter, build_router
from truthforge.types import Action, Label, Passage, Verdict


def state(
    *,
    prob: float,
    flagged: bool,
    coverage: float = 0.8,
    contradiction: float = 0.0,
    history=(),
    text: str = "Paris is the capital of France.",
) -> ClaimState:
    return ClaimState(
        claim_id="c1",
        claim_text=text,
        verdict=Verdict(
            claim_id="c1",
            label=Label.UNSUPPORTED if flagged else Label.SUPPORTED,
            support_prob=prob,
            detector="fake",
        ),
        coverage=coverage,
        contradiction=contradiction,
        history=list(history),
    )


PASSAGES = [Passage(doc_id=f"p{i}", text="Paris is the capital of France.") for i in range(5)]


# -- Self-RAG ---------------------------------------------------------------


def test_self_rag_never_abstains():
    """Self-RAG has no abstain token. Even after the budget is spent it emits
    the best candidate -- that is where its unsupported output comes from, and
    the ledger has to see it as a kept false claim, not a withheld one."""
    r = SelfRAGRouter(max_attempts=2)
    actions = {
        r.route(
            state(prob=p, flagged=True, coverage=c, contradiction=x, history=h),
            PASSAGES,
        ).action
        for p in (0.01, 0.2, 0.45, 0.65)
        for c in (0.0, 0.3, 0.9)
        for x in (0.0, 0.9)
        for h in ((), (Action.REWRITE,), (Action.REWRITE, Action.REWRITE))
    }
    assert Action.ABSTAIN not in actions


def test_self_rag_rewrites_anything_short_of_full_support():
    r = SelfRAGRouter()
    assert r.route(state(prob=0.5, flagged=False), PASSAGES).action is Action.REWRITE
    assert r.route(state(prob=0.9, flagged=False), PASSAGES).action is Action.ACCEPT


def test_self_rag_charges_one_call_per_passage():
    """The loop runs one corrector call, but published Self-RAG runs one per
    retrieved passage. Reporting only the measured call would make the baseline
    look cheaper than the policy actually is."""
    r = SelfRAGRouter(candidates_per_claim=5)
    r.route(state(prob=0.2, flagged=True), PASSAGES)
    assert r.notional_calls == 5

    r2 = SelfRAGRouter(candidates_per_claim=5)
    r2.route(state(prob=0.2, flagged=True), PASSAGES[:2])
    assert r2.notional_calls == 2


def test_self_rag_separates_partial_from_no_support():
    r = SelfRAGRouter()
    assert r.route(state(prob=0.1, flagged=True), PASSAGES).reason == "is_sup_none"
    assert r.route(state(prob=0.5, flagged=True), PASSAGES).reason == "is_sup_partial"


# -- CRAG -------------------------------------------------------------------


def test_crag_keeps_a_flagged_claim_when_the_evidence_grades_correct():
    """The structural difference worth showing: CRAG decides about the
    evidence, we decide about the claim. Good evidence plus one claim that
    misstates it is a false-keep for CRAG and a REWRITE for us."""
    s = state(prob=0.05, flagged=True, coverage=0.9)
    crag = CRAGRouter(Config().router)
    rule = RuleRouter(Config().router)

    assert crag.route(s, PASSAGES).action is Action.ACCEPT
    assert rule.route(s, PASSAGES).action is Action.REWRITE


def test_crag_ignores_detector_confidence_entirely():
    crag = CRAGRouter(Config().router)
    for p in (0.01, 0.5, 0.99):
        d = crag.route(state(prob=p, flagged=True, coverage=0.9), PASSAGES)
        assert d.action is Action.ACCEPT


def test_crag_incorrect_branch_searches_when_it_can():
    crag = CRAGRouter(Config().router, web_search_available=True)
    d = crag.route(state(prob=0.2, flagged=True, coverage=0.05), PASSAGES)
    assert d.action is Action.RERETRIEVE


def test_crag_incorrect_branch_degrades_loudly_on_fixed_evidence():
    """On RAGTruth there is nothing to search. The branch must abstain AND say
    so in the reason, so the paper can footnote how often it fired instead of
    reporting a crippled row as a fair one."""
    crag = CRAGRouter(Config().router, web_search_available=False)
    d = crag.route(state(prob=0.2, flagged=True, coverage=0.05), PASSAGES)
    assert d.action is Action.ABSTAIN
    assert d.reason == "evidence_incorrect_no_web_search"
    assert crag.degraded_incorrect_branches == 1


def test_crag_ambiguous_runs_both_branches_in_order():
    crag = CRAGRouter(Config().router, web_search_available=True)
    first = crag.route(state(prob=0.2, flagged=True, coverage=0.4), PASSAGES)
    assert first.action is Action.RERETRIEVE

    after = crag.route(
        state(prob=0.2, flagged=True, coverage=0.4, history=(Action.RERETRIEVE,)),
        PASSAGES,
    )
    assert after.action is Action.REWRITE


# -- wiring -----------------------------------------------------------------


def test_build_router_resolves_both_baselines():
    for name in ("self_rag", "crag"):
        cfg = Config()
        cfg.router.name = name
        r = build_router(cfg)
        assert r.name == name


def test_fixed_evidence_flag_reaches_crag():
    cfg = Config()
    cfg.router.name = "crag"
    assert build_router(cfg, fixed_evidence=True).web_search_available is False
    assert build_router(cfg, fixed_evidence=False).web_search_available is True


def test_baselines_are_not_the_rule_router_in_disguise():
    """If a baseline agrees with us everywhere, the comparison table says
    nothing. Require a real disagreement rate on a spread of claim states."""
    cfg = Config()
    rule = RuleRouter(cfg.router)
    cases = [
        state(prob=p, flagged=f, coverage=c, contradiction=x)
        for p in (0.05, 0.25, 0.45, 0.55, 0.85)
        for f in (True, False)
        for c in (0.05, 0.4, 0.9)
        for x in (0.0, 0.5)
    ]
    for baseline in (SelfRAGRouter(), CRAGRouter(cfg.router)):
        disagree = sum(
            1
            for s in cases
            if baseline.route(s, PASSAGES).action is not rule.route(s, PASSAGES).action
        )
        assert disagree / len(cases) > 0.25, f"{baseline.name} tracks the rule router"


def test_oracle_finds_its_label_across_processes():
    """Claim ids are UUIDs minted fresh per process, and oracle labels always
    come from a different process than the run that uses them. Keyed by id,
    every lookup missed and the "oracle upper bound" silently became the rule
    router. Keyed by text within an example, it joins."""
    from truthforge.router.rules import OracleRouter

    text = "The Eiffel Tower is 500 metres tall."
    labels = {text: Action.REWRITE.value}          # written by process A

    s = state(prob=0.2, flagged=True, text=text)    # process B: a NEW claim_id
    s.claim_id = "clm_from_a_different_process"

    oracle = OracleRouter(labels)
    d = oracle.route(s, PASSAGES)
    assert d.action is Action.REWRITE
    assert d.reason == "oracle_label"
    assert oracle.misses == 0


def test_oracle_miss_is_counted_not_hidden():
    from truthforge.router.rules import OracleRouter

    oracle = OracleRouter({"some other claim": Action.ABSTAIN.value})
    d = oracle.route(state(prob=0.2, flagged=True), PASSAGES)
    assert d.reason.startswith("oracle_miss:")
    assert oracle.misses == 1
