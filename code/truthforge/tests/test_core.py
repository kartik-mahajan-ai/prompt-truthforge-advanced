"""Tests for the parts that would silently corrupt a result if they broke.

Nothing here needs a GPU, a model download or a network. If a test in this file
starts needing one, the dependency-free guarantee in pyproject.toml is gone and
the repo has stopped being rerunnable by a reviewer.
"""

from __future__ import annotations

import math

import pytest

from truthforge.claims.splitter import RuleSplitter
from truthforge.config import Config, RouterConfig
from truthforge.correct.rewrite import Rewriter, splice
from truthforge.cost import CostLedger
from truthforge.detect.baselines import (
    AlwaysUnsupportedDetector,
    LexicalOverlapDetector,
    coverage,
)
from truthforge.eval.metrics import auroc, detector_metrics, system_metrics
from truthforge.eval.outcomes import classify_attempt, cohens_kappa
from truthforge.loop import LoopComponents, TruthForgeLoop
from truthforge.models import EchoGenerator, ScriptedGenerator
from truthforge.retrieve.base import BM25Index, GivenRetriever, rrf_fuse
from truthforge.router.base import ClaimState
from truthforge.router.rules import AlwaysCorrectRouter, RuleRouter
from truthforge.types import (
    Action,
    Claim,
    CorrectionAttempt,
    Label,
    Outcome,
    Passage,
    Verdict,
    WorseKind,
)


# --------------------------------------------------------------------------
# splitter
# --------------------------------------------------------------------------


def test_splitter_offsets_round_trip():
    """Offsets must index the original answer exactly -- splicing depends on it."""
    answer = "Paris is the capital of France. It has 2.1 million residents. "
    claims = RuleSplitter().split(answer)
    assert len(claims) == 2
    for c in claims:
        assert answer[c.start : c.end] == c.meta["surface"]


def test_splitter_does_not_break_on_abbreviations():
    answer = "Dr. Smith joined in 1998. He led the team for approx. ten years."
    claims = RuleSplitter().split(answer)
    assert len(claims) == 2
    assert claims[0].meta["surface"].startswith("Dr. Smith")


def test_splitter_resolves_leading_pronoun():
    answer = "Marie Curie won two Nobel Prizes. She was born in Warsaw."
    claims = RuleSplitter().split(answer)
    assert claims[1].meta["decontextualised"]
    assert "Marie Curie" in claims[1].text


def test_splitter_drops_meta_commentary():
    answer = (
        "Based on the provided context, here is the answer. "
        "The treaty was signed in 1919."
    )
    claims = RuleSplitter().split(answer)
    assert len(claims) == 1
    assert "treaty" in claims[0].text


def test_splitter_handles_empty_input():
    assert RuleSplitter().split("") == []
    assert RuleSplitter().split("   ") == []


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_cost_ledger_accumulates_and_times():
    ledger = CostLedger()
    with ledger.track("detect") as rec:
        rec.detector_calls = 3
        rec.prompt_tokens = 100
    with ledger.track("rewrite") as rec:
        rec.llm_calls = 1
        rec.prompt_tokens = 50
        rec.completion_tokens = 20

    total = ledger.total()
    assert total.total_tokens == 170
    assert total.llm_calls == 1
    assert total.detector_calls == 3
    assert total.latency_s >= 0.0
    assert set(ledger.by_stage()) == {"detect", "rewrite"}


def test_cost_ledger_records_on_exception():
    """A stage that raises must still be billed, or the cost column lies."""
    ledger = CostLedger()
    with pytest.raises(ValueError):
        with ledger.track("generate") as rec:
            rec.llm_calls = 1
            raise ValueError("boom")
    assert ledger.total().llm_calls == 1


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def test_bm25_ranks_the_relevant_doc_first():
    idx = BM25Index(
        [
            "The Amazon River flows through Brazil and Peru.",
            "Python is a programming language used for data analysis.",
            "The Nile is a river in Africa.",
        ]
    )
    top = idx.retrieve("Amazon River Brazil", k=2)
    assert top[0].text.startswith("The Amazon River")
    assert top[0].rank == 0


def test_rrf_rewards_agreement_between_lists():
    a = [Passage("x", "x"), Passage("y", "y")]
    b = [Passage("y", "y"), Passage("z", "z")]
    fused = rrf_fuse([a, b], k=3)
    # y is 2nd and 1st; x is 1st and absent. y should win on combined rank.
    assert fused[0].doc_id == "y"
    assert {p.doc_id for p in fused} == {"x", "y", "z"}


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------


def test_lexical_detector_separates_supported_from_invented():
    passages = [
        Passage("p0", "The Eiffel Tower was completed in 1889 and is 330 metres tall.")
    ]
    det = LexicalOverlapDetector()
    supported = Claim(text="The Eiffel Tower was completed in 1889.")
    invented = Claim(text="The Eiffel Tower cost 45 million francs to construct.")
    p_sup = det.score_batch([supported], passages)[0]
    p_inv = det.score_batch([invented], passages)[0]
    assert p_sup > p_inv


def test_detector_check_rejects_score_count_mismatch():
    class Broken(AlwaysUnsupportedDetector):
        def score_batch(self, claims, passages):
            return [0.0]

    with pytest.raises(RuntimeError):
        Broken().check([Claim(text="a b c"), Claim(text="d e f")], [])


def test_coverage_is_zero_for_absent_content():
    passages = [Passage("p0", "Cats are mammals.")]
    assert coverage("Quantum chromodynamics describes gluons", passages) == 0.0
    assert coverage("Cats are mammals", passages) == 1.0


# --------------------------------------------------------------------------
# router -- one test per branch of the cascade
# --------------------------------------------------------------------------


def _state(p: float, cov: float = 1.0, contra: float = 0.0, history=None) -> ClaimState:
    return ClaimState(
        claim_id="c1",
        claim_text="The tower is 330 metres tall.",
        verdict=Verdict(
            claim_id="c1",
            label=Label.SUPPORTED if p >= 0.5 else Label.UNSUPPORTED,
            support_prob=p,
            detector="test",
        ),
        coverage=cov,
        contradiction=contra,
        history=list(history or []),
    )


@pytest.mark.parametrize(
    "state_kwargs, expected, reason_fragment",
    [
        ({"p": 0.9}, Action.ACCEPT, "not_flagged"),
        ({"p": 0.48}, Action.ACCEPT, "uncertain"),
        ({"p": 0.1, "cov": 0.05}, Action.RERETRIEVE, "thin_evidence"),
        ({"p": 0.1, "cov": 0.9}, Action.REWRITE, "misstates"),
        ({"p": 0.1, "contra": 0.9}, Action.ABSTAIN, "contradictory"),
        ({"p": 0.35, "cov": 0.9}, Action.ACCEPT, "below_action_threshold"),
    ],
)
def test_rule_router_branches(state_kwargs, expected, reason_fragment):
    d = RuleRouter(RouterConfig()).route(_state(**state_kwargs), [])
    assert d.action is expected
    assert reason_fragment in d.reason


def test_rule_router_abstains_after_budget():
    state = _state(0.1, history=[Action.REWRITE, Action.REWRITE])
    d = RuleRouter(RouterConfig(), max_attempts=2).route(state, [])
    assert d.action is Action.ABSTAIN
    assert d.reason == "attempt_budget_exhausted"


def test_accept_does_not_consume_the_attempt_budget():
    state = _state(0.1, history=[Action.ACCEPT, Action.ACCEPT, Action.ACCEPT])
    assert state.attempts == 0


def test_widening_uncertain_band_reduces_corrections():
    """The over-correction lever must actually move."""
    narrow = RuleRouter(RouterConfig(uncertain_band=0.01))
    wide = RuleRouter(RouterConfig(uncertain_band=0.30))
    states = [_state(p, cov=0.9) for p in (0.25, 0.30, 0.35, 0.45)]
    n_narrow = sum(1 for s in states if narrow.route(s, []).action is Action.REWRITE)
    n_wide = sum(1 for s in states if wide.route(s, []).action is Action.REWRITE)
    assert n_wide < n_narrow


def test_always_correct_router_rewrites_everything_flagged():
    r = AlwaysCorrectRouter()
    assert r.route(_state(0.1, cov=0.0), []).action is Action.REWRITE
    assert r.route(_state(0.49), []).action is Action.REWRITE
    assert r.route(_state(0.9), []).action is Action.ACCEPT


# --------------------------------------------------------------------------
# outcome taxonomy -- the contribution
# --------------------------------------------------------------------------


def _attempt(before: str, after: str, p0: float = 0.2, p1: float = 0.9):
    return CorrectionAttempt(
        claim_id="c1",
        action=Action.REWRITE,
        iteration=0,
        before_text=before,
        after_text=after,
        before_support_prob=p0,
        after_support_prob=p1,
    )


EVIDENCE = [
    Passage(
        "p0",
        "Neil Armstrong walked on the Moon on 20 July 1969 during the Apollo 11 "
        "mission. Water boils at 100 degrees Celsius at sea level.",
    )
]


def test_unchanged_text_is_same():
    a = classify_attempt(_attempt("Same sentence.", "Same sentence."), EVIDENCE)
    assert a.outcome is Outcome.SAME
    assert a.worse_kind is None
    assert a.outcome_source == "heuristic"


def test_info_loss_detected():
    a = classify_attempt(
        _attempt(
            "Neil Armstrong walked on the Moon on 20 July 1969 during Apollo 11.",
            "Neil Armstrong walked on the Moon.",
        ),
        EVIDENCE,
    )
    assert a.outcome is Outcome.WORSE
    assert a.worse_kind is WorseKind.INFO_LOSS


def test_hedging_detected():
    a = classify_attempt(
        _attempt(
            "Water boils at 100 degrees Celsius at sea level.",
            "Water may boil at approximately 100 degrees under typical conditions.",
        ),
        EVIDENCE,
    )
    assert a.outcome is Outcome.WORSE
    assert a.worse_kind is WorseKind.HEDGING


def test_new_error_detected():
    a = classify_attempt(
        _attempt(
            "Neil Armstrong walked on the Moon.",
            "Neil Armstrong walked on the Moon wearing a prototype titanium exosuit.",
        ),
        EVIDENCE,
    )
    assert a.outcome is Outcome.WORSE
    assert a.worse_kind is WorseKind.NEW_ERROR


def test_genuine_improvement_is_better():
    a = classify_attempt(
        _attempt(
            "Neil Armstrong walked on the Moon on 21 July 1969 during Apollo 11.",
            "Neil Armstrong walked on the Moon on 20 July 1969 during Apollo 11.",
        ),
        EVIDENCE,
    )
    assert a.outcome is Outcome.BETTER


def test_support_gain_by_deletion_is_not_counted_as_better():
    """The blind spot this project exists to measure.

    A rewrite that raises P(supported) purely by deleting the contested detail
    must NOT score as BETTER, however happy the detector is.
    """
    a = classify_attempt(
        _attempt(
            "Neil Armstrong walked on the Moon on 20 July 1969 during Apollo 11.",
            "Neil Armstrong walked on the Moon.",
            p0=0.2,
            p1=0.99,
        ),
        EVIDENCE,
    )
    assert a.outcome is Outcome.WORSE


def test_cohens_kappa_endpoints():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert math.isnan(cohens_kappa(["a"], ["a"]))
    partial = cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "a"])
    assert -1.0 <= partial <= 1.0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_detector_metrics_counts():
    preds = [Label.UNSUPPORTED, Label.SUPPORTED, Label.UNSUPPORTED, Label.SUPPORTED]
    gold = [True, False, False, True]
    m = detector_metrics(preds, gold)
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5


def test_auroc_perfect_and_degenerate():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert math.isnan(auroc([0.5, 0.5], [True, True]))


# --------------------------------------------------------------------------
# rewriting and splicing
# --------------------------------------------------------------------------


def test_splice_uses_offsets_and_preserves_surroundings():
    answer = "Alpha sentence. Beta sentence. Gamma sentence."
    claim = RuleSplitter().split(answer)[1]
    out = splice(answer, claim, "Replaced.")
    assert out == "Alpha sentence. Replaced. Gamma sentence."


def test_splice_returns_answer_unchanged_when_it_cannot_find_the_claim():
    answer = "Alpha sentence."
    orphan = Claim(text="Nowhere to be found.", start=None, end=None)
    assert splice(answer, orphan, "X") == answer


def test_rewriter_honours_the_abstain_sentinel():
    rw = Rewriter(ScriptedGenerator(["UNSUPPORTED"]), abstain_message="cannot verify")
    res = rw.rewrite(Claim(text="Something dubious."), "Something dubious.", EVIDENCE)
    assert res.abstained
    assert res.rewritten == "cannot verify"


def test_rewriter_strips_chatty_wrappers():
    rw = Rewriter(ScriptedGenerator(['"The tower is 330 metres tall."\n\nLet me know!']))
    res = rw.rewrite(Claim(text="The tower is 300 metres tall."), "x", EVIDENCE)
    assert res.rewritten == "The tower is 330 metres tall."


def test_echo_generator_returns_the_sentence_unchanged():
    """A stub must not manufacture improvements."""
    rw = Rewriter(EchoGenerator())
    claim = Claim(text="The tower is 330 metres tall.")
    res = rw.rewrite(claim, "The tower is 330 metres tall.", EVIDENCE)
    assert res.rewritten == claim.text
    assert not res.changed


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _loop(router, generator, cfg=None):
    cfg = cfg or Config()
    return TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=router,
            retriever=GivenRetriever(EVIDENCE),
            splitter=RuleSplitter(),
            rewriter=Rewriter(generator, abstain_message=cfg.loop.abstain_message),
            generate_fn=generator,
        ),
        cfg,
    )


def test_loop_runs_and_bills_every_stage():
    turn = _loop(RuleRouter(), EchoGenerator()).run(
        question="When did Armstrong walk on the Moon?",
        answer="Neil Armstrong walked on the Moon on 20 July 1969.",
        passages=EVIDENCE,
    )
    stages = {c.stage for c in turn.costs}
    assert {"split", "detect", "reverify"} <= stages
    assert turn.iterations_used >= 1
    assert all(c.latency_s >= 0 for c in turn.costs)


def test_noop_arm_spends_less_than_always_correct():
    """The cost column must be able to tell the arms apart."""
    from truthforge.router.rules import NoopRouter

    answer = (
        "Neil Armstrong walked on the Moon in 1969. "
        "The mission cost forty billion dollars and used a titanium exosuit."
    )
    kw = dict(question="q", answer=answer, passages=EVIDENCE)
    noop = _loop(NoopRouter(), EchoGenerator()).run(**kw)
    always = _loop(AlwaysCorrectRouter(), EchoGenerator()).run(**kw)

    noop_tokens = sum(c.total_tokens for c in noop.costs)
    always_tokens = sum(c.total_tokens for c in always.costs)
    assert always_tokens > noop_tokens
    assert sum(c.llm_calls for c in always.costs) > sum(c.llm_calls for c in noop.costs)


def test_reretrieve_is_downgraded_when_the_retriever_cannot_reretrieve():
    """RAGTruth fixes the passages. The loop must say so, not pretend."""
    cfg = Config()
    cfg.router.thin_evidence_coverage = 0.99  # force the thin-evidence branch
    turn = _loop(RuleRouter(cfg.router), EchoGenerator(), cfg).run(
        question="q",
        answer="Quantum chromodynamics describes the strong interaction via gluons.",
        passages=EVIDENCE,
    )
    reasons = [d.reason for d in turn.decisions]
    assert any(r.startswith("reretrieve_unavailable(") for r in reasons)
    assert all(d.action is not Action.RERETRIEVE for d in turn.decisions)


def test_loop_respects_max_iterations():
    cfg = Config()
    cfg.loop.max_iterations = 1
    turn = _loop(AlwaysCorrectRouter(), EchoGenerator(), cfg).run(
        question="q",
        answer="The mission cost forty billion dollars in unmarked bills.",
        passages=EVIDENCE,
    )
    assert turn.iterations_used == 1


def test_turn_result_serialises_to_json():
    import json

    turn = _loop(RuleRouter(), EchoGenerator()).run(
        question="q", answer="Water boils at 100 degrees Celsius.", passages=EVIDENCE
    )
    blob = json.dumps(turn.to_dict())
    assert json.loads(blob)["question"] == "q"


def test_system_metrics_on_an_empty_run():
    assert system_metrics([]) == {"n_turns": 0}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_config_round_trips(tmp_path):
    cfg = Config()
    cfg.router.uncertain_band = 0.25
    path = tmp_path / "config.json"
    cfg.save(path)
    assert Config.load(path).router.uncertain_band == 0.25


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_real_key": 1})


# --------------------------------------------------------------------------
# corrector output guards
#
# Regression tests for a real failure seen on the first run with a 1.5B
# corrector: asked to fix one sentence, it echoed the whole answer, which
# splice() then pasted over a single sentence's span -- duplicating text while
# the outcome classifier scored it SAME, because every duplicated word was
# genuinely supported by the evidence.
# --------------------------------------------------------------------------


ANSWER_3 = (
    "The Eiffel Tower was completed in 1889 for the Paris World's Fair. "
    "It stands 330 metres tall including antennas. "
    "It was designed by Gustave Eiffel and cost 45 million francs to build."
)
CLAIM_3 = "It was designed by Gustave Eiffel and cost 45 million francs to build."


def test_rewriter_rejects_a_whole_answer_echo():
    from truthforge.correct.rewrite import Rewriter

    rw = Rewriter(ScriptedGenerator([ANSWER_3]))
    res = rw.rewrite(Claim(text=CLAIM_3), ANSWER_3, EVIDENCE)
    assert res.rejected
    assert res.rewritten == CLAIM_3       # original kept
    assert not res.changed                # so the outcome is an honest SAME
    assert res.raw_output == ANSWER_3     # and the raw output is preserved


def test_rejected_rewrite_leaves_the_answer_untouched():
    """The actual damage was duplicated text in the final answer."""
    from truthforge.correct.rewrite import Rewriter, splice

    claims = RuleSplitter().split(ANSWER_3)
    target = claims[-1]
    rw = Rewriter(ScriptedGenerator([ANSWER_3]))
    res = rw.rewrite(target, ANSWER_3, EVIDENCE)
    spliced = splice(ANSWER_3, target, res.rewritten)
    assert spliced == ANSWER_3
    assert spliced.count("330 metres tall") == 1


def test_rewriter_accepts_a_clean_single_sentence():
    from truthforge.correct.rewrite import Rewriter

    good = "It was designed by Gustave Eiffel."
    rw = Rewriter(ScriptedGenerator([good]))
    res = rw.rewrite(Claim(text=CLAIM_3), ANSWER_3, EVIDENCE)
    assert not res.rejected
    assert res.rewritten == good
    assert res.changed


def test_rewriter_recovers_the_sentence_from_a_chatty_answer():
    """One good sentence plus commentary is recoverable, not a rejection."""
    from truthforge.correct.rewrite import Rewriter

    rw = Rewriter(ScriptedGenerator([
        "It was designed by Gustave Eiffel.\nI removed the unsupported cost."
    ]))
    res = rw.rewrite(Claim(text=CLAIM_3), ANSWER_3, EVIDENCE)
    assert not res.rejected
    assert res.rewritten == "It was designed by Gustave Eiffel."


def test_rewriter_rejects_an_overlong_output():
    from truthforge.correct.rewrite import Rewriter

    claim = "Mars has two moons."
    bloated = "Mars has two moons " + "and many other interesting properties " * 8
    rw = Rewriter(ScriptedGenerator([bloated]))
    res = rw.rewrite(Claim(text=claim), claim, EVIDENCE)
    assert res.rejected
    assert "too_long" in res.rejection


def test_rewriter_rejects_an_unrelated_output():
    from truthforge.correct.rewrite import Rewriter

    rw = Rewriter(ScriptedGenerator(["Bananas grow in tropical climates."]))
    res = rw.rewrite(Claim(text="Mars has two moons named Phobos and Deimos."),
                     "Mars has two moons named Phobos and Deimos.", EVIDENCE)
    assert res.rejected


def test_rejection_is_recorded_on_the_attempt_and_counted():
    """A rejected attempt still costs tokens, so it must stay visible."""
    from truthforge.correct.rewrite import Rewriter
    from truthforge.eval.metrics import correction_metrics
    from truthforge.router.rules import AlwaysCorrectRouter

    cfg = Config()
    cfg.loop.max_iterations = 1
    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=AlwaysCorrectRouter(),
            retriever=GivenRetriever(EVIDENCE),
            splitter=RuleSplitter(),
            rewriter=Rewriter(ScriptedGenerator([ANSWER_3] * 6)),
        ),
        cfg,
    )
    turn = loop.run(question="q", answer=ANSWER_3, passages=EVIDENCE)
    assert turn.final_answer == ANSWER_3  # nothing corrupted

    rejected = [a for a in turn.attempts if a.rejection]
    assert rejected, "the echo should have been rejected"

    from truthforge.eval.outcomes import classify_attempt

    for a in turn.attempts:
        classify_attempt(a, EVIDENCE)
    m = correction_metrics(turn.attempts)
    assert m["rejection_rate"] > 0
    assert m["n_rejected_outputs"] == len(rejected)


def test_turn_rehydration_preserves_every_reported_field(tmp_path):
    """Metrics computed from a reloaded run must match the live run.

    This caught a real bug: `rejection` was not restored by _turn_from_dict,
    so `rejection_rate` read 0 for every run loaded from disk while the live
    run reported it correctly -- the comparison table silently disagreed with
    the per-run metrics.json.
    """
    import json

    from truthforge.cli import _turn_from_dict
    from truthforge.correct.rewrite import Rewriter
    from truthforge.eval.metrics import system_metrics
    from truthforge.eval.outcomes import classify_attempt
    from truthforge.router.rules import AlwaysCorrectRouter

    cfg = Config()
    cfg.loop.max_iterations = 1
    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=AlwaysCorrectRouter(),
            retriever=GivenRetriever(EVIDENCE),
            splitter=RuleSplitter(),
            rewriter=Rewriter(ScriptedGenerator([ANSWER_3] * 6)),
        ),
        cfg,
    )
    turn = loop.run(question="q", answer=ANSWER_3, passages=EVIDENCE)
    for a in turn.attempts:
        classify_attempt(a, EVIDENCE)

    live = system_metrics([turn])
    reloaded = system_metrics([_turn_from_dict(json.loads(json.dumps(turn.to_dict())))])

    for key in (
        "faithfulness",
        "abstention_rate",
        "correction_n_attempts",
        "correction_worse_rate",
        "correction_futile_rate",
        "correction_rejection_rate",
        "correction_n_rejected_outputs",
        "tokens_per_turn",
    ):
        assert live.get(key) == reloaded.get(key), f"{key} did not survive the round trip"


def test_router_abstains_when_the_corrector_cannot_fix_it():
    """A rewrite that changed nothing will change nothing on retry: decoding is
    greedy, so the same corrector over the same evidence is deterministic.
    Retrying burns tokens and leaves a known-false claim standing as fact."""
    state = _state(0.05, cov=0.9)
    state.history = [Action.REWRITE]
    state.last_rewrite_failed = True
    d = RuleRouter(RouterConfig()).route(state, [])
    assert d.action is Action.ABSTAIN
    assert d.reason == "corrector_could_not_fix_it"


def test_a_failed_rewrite_still_retries_after_new_evidence():
    """Re-retrieval changes the evidence, so the retry is no longer futile."""
    state = _state(0.05, cov=0.9)
    state.history = [Action.REWRITE, Action.RERETRIEVE]
    state.last_rewrite_failed = True
    d = RuleRouter(RouterConfig(), max_attempts=4).route(state, [])
    assert d.action is Action.REWRITE


def test_loop_abstains_rather_than_asserting_an_unfixable_claim():
    """End to end: a corrector that cannot fix the claim must not leave the
    false statement in the final answer."""
    from truthforge.correct.rewrite import Rewriter

    answer = "Microsoft was founded by Bill Gates and Steve Jobs in 1975."
    evidence = [Passage("p0", "Microsoft was founded by Bill Gates and Paul Allen in 1975.")]

    class Stubborn:
        """A corrector that always returns the claim unchanged."""
        def __call__(self, prompt):
            return prompt.split("SENTENCE TO CORRECT:\n")[1].split("\n\nRules:")[0]

    cfg = Config()
    cfg.loop.max_iterations = 2
    cfg.loop.abstain_message = "COULD NOT VERIFY"
    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=RuleRouter(cfg.router, max_attempts=cfg.loop.max_attempts_per_claim),
            retriever=GivenRetriever(evidence),
            splitter=RuleSplitter(),
            rewriter=Rewriter(Stubborn()),
        ),
        cfg,
    )
    turn = loop.run(question="q", answer=answer, passages=evidence)

    reasons = [d.reason for d in turn.decisions]
    assert "corrector_could_not_fix_it" in reasons
    assert turn.abstained
    # Abstention uses the loop's message, not the rewriter's -- the rewriter's
    # only applies when the model itself emits the UNSUPPORTED sentinel.
    assert "COULD NOT VERIFY" in turn.final_answer
    assert "Steve Jobs" not in turn.final_answer
    # And it gave up after one wasted attempt, not two.
    assert len([a for a in turn.attempts if a.action is Action.REWRITE]) == 1


def test_faithfulness_excludes_abstained_claims():
    """A claim the system declined to assert is not in the final answer, so it
    must not count against faithfulness. Counting it would penalise the system
    for the exact behaviour the abstain action exists to produce."""
    from truthforge.types import TurnResult

    t = TurnResult(question="q", initial_answer="a", final_answer="a")
    t.verdicts = [
        Verdict(claim_id="c1", label=Label.SUPPORTED, support_prob=0.9, detector="d"),
        Verdict(claim_id="c2", label=Label.SUPPORTED, support_prob=0.9, detector="d"),
        Verdict(claim_id="c3", label=Label.UNSUPPORTED, support_prob=0.0, detector="d"),
    ]

    # metrics round to 4 dp, so compare with matching tolerance
    assert system_metrics([t])["faithfulness"] == pytest.approx(2 / 3, abs=1e-4)

    t.meta["abstained_claim_ids"] = ["c3"]
    t.abstained = True
    m = system_metrics([t])
    assert m["faithfulness"] == 1.0
    # and the abstention is still visible, so the pair cannot be gamed silently
    assert m["abstention_rate"] == 1.0


# --------------------------------------------------------------------------
# generation guards
#
# An answer that IS the evidence is trivially supported by the evidence, so
# every claim passes verification, the router finds nothing to do, and the
# pipeline reports a perfect result having answered nothing. Observed with
# Qwen2.5-1.5B over a multi-chunk document.
# --------------------------------------------------------------------------


def test_echoed_passages_are_stripped_from_the_answer():
    from truthforge.loop import _strip_echoed_passages

    ev = [Passage("p0", "Section 4: Employees are entitled to 25 days of paid annual leave.\n"
                        "Section 7: Termination requires 30 days written notice.")]
    echoed = ("Section 4: Employees are entitled to 25 days of paid annual leave.\n"
              "Employees get 25 days of leave each year.")
    out = _strip_echoed_passages(echoed, ev)
    assert out == "Employees get 25 days of leave each year."


def test_a_wholly_copied_answer_becomes_a_refusal():
    """If nothing survives stripping, the model composed nothing of its own."""
    from truthforge.loop import _strip_echoed_passages

    ev = [Passage("p0", "Section 7: Termination requires 30 days written notice.")]
    out = _strip_echoed_passages("Section 7: Termination requires 30 days written notice.", ev)
    assert out == "The documents do not contain this information."


def test_a_genuine_answer_is_left_alone():
    from truthforge.loop import _strip_echoed_passages

    ev = [Passage("p0", "Section 7: Termination requires 30 days written notice.")]
    real = "You must give 30 days of written notice to terminate."
    assert _strip_echoed_passages(real, ev) == real


def test_short_evidence_lines_do_not_suppress_the_answer():
    """Only substantial lines count as 'copied'; a shared short phrase must
    not delete a real answer."""
    from truthforge.loop import _strip_echoed_passages

    ev = [Passage("p0", "Yes.\nTermination requires notice.")]
    assert _strip_echoed_passages("Yes.", ev) == "Yes."


def test_refusals_are_not_routed_as_claims():
    """"Unable to answer based on the given passages" is a statement about the
    system's knowledge, not about the world. Scoring it for support is a
    category error: the detector marks it unsupported (p ~ 0.02-0.07 measured
    on RAGTruth), the router sends it to the corrector, and a 1.5B model is
    asked to make a refusal more faithful to passages it is refusing to use.
    Twelve of 81 flags on the RAGTruth QA slice were this."""
    from truthforge.claims.splitter import RuleSplitter, is_refusal

    for t in [
        "Unable to answer based on given passages.",
        "I am unable to answer the question based on the given passages.",
        "I cannot answer the question based on the given passages.",
        "Not enough information.",
        "No relevant information is available in the provided context.",
        "The answer is not provided.",
    ]:
        assert is_refusal(t), t

    answer = (
        "Unable to answer based on given passages.\n"
        "Paris is the capital of France."
    )
    claims = RuleSplitter().split(answer)
    assert [c.text for c in claims] == ["Paris is the capital of France."]


def test_the_refusal_filter_does_not_swallow_an_embedded_assertion():
    """The dangerous version of this filter matches anything CONTAINING a
    refusal phrase, which would exempt "The passages do not mention that the
    tower is 500m tall" from verification -- smuggling a factual claim out of
    the pipeline inside refusal-shaped wording. That is the opposite of what
    the filter is for, so the patterns are anchored to the whole sentence."""
    from truthforge.claims.splitter import is_refusal

    for t in [
        "The passages do not mention that the Eiffel Tower is 500 metres tall.",
        "I cannot answer the question because the tower was built in 1889.",
        "There is not enough information in the 1889 census to settle this.",
        "Paris is the capital of France.",
    ]:
        assert not is_refusal(t), t


def test_subject_prefix_is_never_a_sentence_initial_function_word():
    """English capitalises the first word of every sentence, so "capitalised
    run at the start" is not evidence of a name. The first version of
    `_leading_noun_phrase` treated it as one and produced claims like
    "(The) It stands 330 metres tall" -- 6.5% of claims on RAGTruth carried a
    prefix and two-thirds were garbage. The corrector then spent calls
    deleting the artifact and the detector scored the clean-up as damage; a
    human labelling 30 of those "damaging" rewrites judged 2 harmful."""
    from truthforge.claims.splitter import _leading_noun_phrase as L

    for s in [
        "The tower was completed in 1889.",
        "Based on the passages, it is safe.",
        "In general, the oven should be hot.",
        "If you want, add salt.",
        "Use a stiff brush.",
        "Passage 3 describes the method.",
        "Step 4 is to bake it.",
        "Seeking help is important.",
        "Rub the meat with salt.",
        "Dissolve the sugar in water.",
        "Remember to rest.",
        "Lightly oil the grill.",
        "Here's what to do.",
        "It stands 330 metres tall.",
    ]:
        assert L(s) is None, s


def test_subject_prefix_still_finds_real_names():
    from truthforge.claims.splitter import _leading_noun_phrase as L

    assert L("The Eiffel Tower was completed in 1889.") == "Eiffel Tower"
    assert L("Magellan sailed around the world.") == "Magellan"
    assert L("DNA carries genetic information.") == "DNA"
    assert L("Chard is a leafy vegetable.") == "Chard"
    assert L("New York City is large.") == "New York City"


def test_no_garbage_prefix_reaches_a_claim():
    from truthforge.claims.splitter import RuleSplitter

    claims = RuleSplitter().split(
        "The tower was completed in 1889. It stands 330 metres tall."
    )
    assert all(not c.text.startswith("(The)") for c in claims)
    assert [c.text for c in claims][1] == "It stands 330 metres tall."


def test_train_policy_actually_fits_against_the_installed_sklearn():
    """`train_policy` passed `multi_class="multinomial"`, which scikit-learn
    removed in 1.7. It raised before fitting a single row, so the learned
    router had never been trained on real labels -- and no test noticed,
    because none fitted a real model. This one does, for both the binary case
    (what the RAGTruth auto-labels produce) and the multi-class case."""
    import random

    pytest.importorskip("sklearn")
    from truthforge.router.learned import FEATURE_ORDER, train_policy

    rng = random.Random(0)

    def feats(cov):
        f = {k: rng.random() for k in FEATURE_ORDER}
        f["coverage"] = cov
        return f

    # binary: rewrite when coverage is high
    X = [feats(0.9) for _ in range(40)] + [feats(0.1) for _ in range(40)]
    y = ["rewrite"] * 40 + ["accept"] * 40
    pol = train_policy(X, y, seed=0)
    assert pol is not None

    # three classes
    X3 = X + [feats(0.5) for _ in range(40)]
    y3 = y + ["abstain"] * 40
    assert train_policy(X3, y3, seed=0) is not None
