"""Tests for the head-to-head evaluation framework.

The point of these metrics is that they cannot be gamed in either direction,
so most of what is tested here is the degenerate strategies failing.
"""

from __future__ import annotations

import pytest

from truthforge.eval.comparison import (
    Cell,
    Fate,
    claim_fates,
    compare_arms,
    pareto_front,
    paired_bootstrap,
    score_arm,
)
from truthforge.types import CorrectionAttempt, Action, Label, TurnResult, Verdict


def _turn(claims, dropped=(), rewritten=(), tokens=100):
    """claims: {claim_id: (first_supported, last_supported)}"""
    from truthforge.types import CostRecord

    t = TurnResult(question="q", initial_answer="a", final_answer="b")
    for cid, (first, last) in claims.items():
        for supported in (first, last):
            t.verdicts.append(
                Verdict(
                    claim_id=cid,
                    label=Label.SUPPORTED if supported else Label.UNSUPPORTED,
                    support_prob=0.9 if supported else 0.1,
                    detector="d",
                )
            )
    for cid in rewritten:
        t.attempts.append(
            CorrectionAttempt(
                claim_id=cid, action=Action.REWRITE, iteration=0,
                before_text="before", after_text="AFTER", before_support_prob=0.1,
            )
        )
    t.meta["abstained_claim_ids"] = list(dropped)
    t.costs.append(CostRecord(stage="detect", prompt_tokens=tokens))
    return t


# ---------------------------------------------------------------- ledger


def test_every_cell_is_reachable():
    t = _turn(
        {
            "a": (True, True),    # supported, untouched      -> true_keep
            "b": (False, False),  # unsupported, untouched    -> false_keep
            "c": (False, True),   # unsupported, rewritten ok -> true_fix
            "d": (True, False),   # supported, rewrite broke  -> false_fix
            "e": (False, False),  # unsupported, dropped      -> true_drop
            "f": (True, True),    # supported, dropped        -> false_drop
        },
        dropped=("e", "f"),
        rewritten=("c", "d"),
    )
    cells = {f.claim_id: f.cell for f in claim_fates(t)}
    assert cells == {
        "a": Cell.TRUE_KEEP, "b": Cell.FALSE_KEEP,
        "c": Cell.TRUE_FIX,  "d": Cell.FALSE_FIX,
        "e": Cell.TRUE_DROP, "f": Cell.FALSE_DROP,
    }


def test_a_rewrite_is_judged_on_where_it_landed():
    """Rewriting a supported claim is only over-correction if it BREAKS it."""
    ok = _turn({"a": (True, True)}, rewritten=("a",))
    assert claim_fates(ok)[0].cell is Cell.TRUE_KEEP

    broke = _turn({"a": (True, False)}, rewritten=("a",))
    assert claim_fates(broke)[0].cell is Cell.FALSE_FIX


def test_gold_labels_override_the_detector():
    t = _turn({"a": (False, False)})
    assert claim_fates(t)[0].was_supported is False
    assert claim_fates(t, gold={"a": True})[0].was_supported is True


# ------------------------------------------------- the two rates


def test_abstaining_on_everything_scores_zero():
    """The degenerate strategy that raw faithfulness would reward."""
    t = _turn({"a": (True, True), "b": (False, False)}, dropped=("a", "b"))
    s = score_arm("abstain_all", [t])
    assert s.error_removal == 1.0     # removed the bad claim
    assert s.retention == 0.0         # and the good one
    assert s.net_quality == 0.0       # so it scores nothing


def test_changing_nothing_scores_zero_when_errors_exist():
    t = _turn({"a": (True, True), "b": (False, False)})
    s = score_arm("noop", [t])
    assert s.retention == 1.0
    assert s.error_removal == 0.0
    assert s.net_quality == 0.0


def test_a_perfect_arm_scores_one():
    t = _turn({"a": (True, True), "b": (False, True)}, rewritten=("b",))
    s = score_arm("perfect", [t])
    assert s.net_quality == pytest.approx(1.0)


def test_net_quality_sits_between_the_two_rates():
    t = _turn(
        {"a": (True, True), "b": (True, True), "c": (False, True), "d": (False, False)},
        rewritten=("c",),
    )
    s = score_arm("mixed", [t])
    assert min(s.error_removal, s.retention) <= s.net_quality <= max(
        s.error_removal, s.retention
    )


def test_gold_source_is_recorded():
    t = _turn({"a": (True, True)})
    assert score_arm("x", [t]).gold_source == "detector_on_original"
    assert score_arm("x", [t], gold={"a": True}).gold_source == "annotated"


# ------------------------------------------------- significance


def test_bootstrap_detects_a_real_shift():
    a = [0.8] * 40
    b = [0.5] * 40
    r = paired_bootstrap(a, b, n_resamples=2000)
    assert r["mean_diff"] == pytest.approx(0.3)
    assert r["significant"]
    assert r["p_value"] < 0.05


def test_bootstrap_does_not_cry_wolf_on_noise():
    import random

    rng = random.Random(0)
    a = [rng.gauss(0.5, 0.1) for _ in range(60)]
    b = [rng.gauss(0.5, 0.1) for _ in range(60)]
    r = paired_bootstrap(a, b, n_resamples=2000)
    assert not r["significant"]


def test_bootstrap_requires_paired_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


# ------------------------------------------------- head to head


def _arm(name, quality, tokens, n=30):
    turns = [
        _turn({"a": (True, True), "b": (False, quality)}, rewritten=("b",), tokens=tokens)
        for _ in range(n)
    ]
    return score_arm(name, turns)


def test_equal_quality_but_cheaper_is_the_efficiency_claim():
    ours = _arm("rule", True, 100)
    base = _arm("always", True, 300)
    r = compare_arms(ours, base, n_resamples=2000)
    assert r["verdict"] == "equal quality, cheaper"
    assert "fewer tokens" in r["claim"]
    assert r["token_saving_frac"] == pytest.approx(2 / 3, abs=1e-3)


def test_losing_on_quality_is_reported_as_a_loss():
    ours = _arm("rule", False, 100)
    base = _arm("always", True, 300)
    r = compare_arms(ours, base, n_resamples=2000)
    assert r["verdict"] == "loses on quality"
    assert "No efficiency argument rescues this" in r["claim"]


def test_no_difference_refuses_to_claim_a_win():
    ours = _arm("rule", True, 200)
    base = _arm("always", True, 200)
    r = compare_arms(ours, base, n_resamples=2000)
    assert r["verdict"] == "no detectable difference"
    assert "Do not claim a win" in r["claim"]


def test_arms_must_cover_the_same_turns():
    with pytest.raises(ValueError, match="same turns"):
        compare_arms(_arm("a", True, 100, n=10), _arm("b", True, 100, n=11), 200)


def test_pareto_front_drops_dominated_arms():
    good = _arm("good", True, 100)
    costly = _arm("costly", True, 500)
    front = pareto_front([good, costly])
    assert front == ["good"]


def test_turn_round_trip_keeps_everything_the_reports_read():
    """A field missing from `_turn_from_dict` does not crash -- it reads zero.

    This has now bitten twice. `rejection` was dropped, so rejection_rate read
    0 for every run loaded from disk while the live run printed it correctly.
    Then `claims` was dropped, so `compare_arms --gold` joined 0 of 392 claims
    against RAGTruth's spans and silently fell back to scoring the detector
    against itself. Both looked like plausible numbers.
    """
    import json

    from truthforge.cli import _turn_from_dict
    from truthforge.types import (
        Action, Claim, CorrectionAttempt, CostRecord, Label, Passage,
        RoutingDecision, TurnResult, Verdict,
    )

    t = TurnResult(
        question="q", final_answer="a", initial_answer="a0", turn_id="t1",
        dataset="ragtruth", example_id="ex1", iterations_used=2, wall_clock_s=1.5,
        meta={"abstained_claim_ids": ["c1"]},
    )
    t.claims = [Claim(text="x", claim_id="c1", start=3, end=11, sent_index=0)]
    t.passages = [Passage(doc_id="p1", text="evidence text", source="given")]
    t.verdicts = [Verdict(claim_id="c1", label=Label.UNSUPPORTED,
                          support_prob=0.2, detector="nli")]
    t.decisions = [RoutingDecision(claim_id="c1", action=Action.REWRITE,
                                   reason="r", router="rule", iteration=0)]
    t.attempts = [CorrectionAttempt(claim_id="c1", action=Action.REWRITE, iteration=0,
                                    before_text="b", after_text="a",
                                    before_support_prob=0.2,
                                    rejection="output_too_long")]
    t.costs = [CostRecord(stage="detect", prompt_tokens=5, completion_tokens=1)]

    back = _turn_from_dict(json.loads(json.dumps(t.to_dict())))

    assert back.example_id == "ex1"
    assert back.meta["abstained_claim_ids"] == ["c1"]
    assert [(c.claim_id, c.start, c.end) for c in back.claims] == [("c1", 3, 11)]
    # Evidence: the annotation workbook shipped with an empty EVIDENCE column
    # because this was missing, which would have had a human judging rewrites
    # against nothing at all.
    assert [(p.doc_id, p.text) for p in back.passages] == [("p1", "evidence text")]
    assert [d.action for d in back.decisions] == [Action.REWRITE]
    assert back.attempts[0].rejection == "output_too_long"
    assert back.verdicts[0].label is Label.UNSUPPORTED
    assert back.costs[0].prompt_tokens == 5


def test_pooled_and_per_turn_quality_can_disagree_and_both_are_reported():
    """The headline number and the pre-registered test measure different things.

    `_rates` defaults error removal to 1.0 for a turn with no unsupported
    claims, so on a clean turn an arm that changes nothing scores a perfect
    1.0 and an arm that drops anything scores less. Most RAGTruth turns are
    clean, so averaging per-turn scores rewards inaction while the pooled
    ledger -- the number in the table -- rewards removing the errors that do
    exist. This is a property of the metric, not noise, and the report has to
    show both rather than pick whichever agrees with the conclusion.
    """
    from truthforge.eval.comparison import cluster_bootstrap_pooled

    # 9 clean turns, 1 turn holding every error.
    clean = {"true_keep": 10, "false_keep": 0, "true_fix": 0,
             "false_fix": 0, "true_drop": 0, "false_drop": 0}
    # "ours": drops a good claim on each clean turn, but fixes the real errors.
    ours = [dict(clean, true_keep=9, false_drop=1) for _ in range(9)]
    ours.append({"true_keep": 10, "false_keep": 0, "true_fix": 8,
                 "false_fix": 0, "true_drop": 0, "false_drop": 0})
    # "theirs": never changes anything.
    theirs = [dict(clean) for _ in range(9)]
    theirs.append({"true_keep": 10, "false_keep": 8, "true_fix": 0,
                   "false_fix": 0, "true_drop": 0, "false_drop": 0})

    r = cluster_bootstrap_pooled(ours, theirs, n_resamples=500)
    assert r["statistic"] == "pooled_net_quality"
    assert r["preregistered"] is False
    # Pooled: ours removes 8 of 8 errors, theirs none -> ours clearly ahead.
    assert r["mean_diff"] > 0.5


def test_compare_arms_reports_both_quality_tests():
    from truthforge.eval.comparison import ArmScore, compare_arms

    def arm(name, per_turn, cells, tokens):
        return ArmScore(
            arm=name, n_turns=len(per_turn), n_claims=10, cells=cells,
            error_removal=0.5, retention=0.9, net_quality=0.6,
            tokens_per_turn=tokens, latency_per_turn=1.0, gold_source="annotated",
            per_turn=per_turn, per_turn_tokens=[tokens] * len(per_turn),
            per_turn_cells=[dict(cells) for _ in per_turn],
        )

    cells = {"true_keep": 8, "false_keep": 1, "true_fix": 1,
             "false_fix": 0, "true_drop": 0, "false_drop": 0}
    a = arm("ours", [0.6] * 20, cells, 500.0)
    b = arm("theirs", [0.6] * 20, cells, 900.0)

    r = compare_arms(a, b, n_resamples=200)
    assert r["quality_delta"]["preregistered"] is True
    assert r["quality_delta"]["statistic"] == "per_turn_net_quality"
    assert r["quality_delta_pooled"]["preregistered"] is False
    assert r["quality_delta_pooled"]["statistic"] == "pooled_net_quality"


def test_gold_scoring_skips_claims_it_cannot_label_rather_than_guessing():
    """A claim with no character offsets cannot be joined to a span annotation.

    Falling back to the detector's verdict for just that claim would mix two
    gold sources under one `gold_source: "annotated"` stamp. On a 60-turn slice
    where the arms differ by eight claims, one silently detector-scored claim
    is enough to move a column, so it is excluded and counted instead.
    """
    from truthforge.eval.comparison import score_arm
    from truthforge.types import Label, TurnResult, Verdict

    t = TurnResult(question="q", final_answer="a", initial_answer="a",
                   turn_id="t1", dataset="ragtruth", example_id="e1")
    t.verdicts = [
        Verdict(claim_id="c1", label=Label.SUPPORTED, support_prob=0.9, detector="d"),
        Verdict(claim_id="c2", label=Label.UNSUPPORTED, support_prob=0.1, detector="d"),
        Verdict(claim_id="orphan", label=Label.UNSUPPORTED, support_prob=0.1, detector="d"),
    ]

    gold = {"c1": True, "c2": False}          # "orphan" has no label
    s = score_arm("arm", [t], gold)

    assert s.n_claims == 2
    assert s.n_unscorable == 1
    assert s.gold_source == "annotated"
    # The orphan must not have been counted as an unsupported kept claim.
    assert s.cells["false_keep"] == 1       # c2 only


def test_net_quality_is_flagged_where_it_ranks_arms_wrongly():
    """The headline metric fails in the near-zero-removal regime.

    net_quality is the harmonic mean of error removal and retention, chosen so
    that neither degenerate strategy wins. That holds at the ends and breaks in
    the middle: a harmonic mean is pinned near zero whenever either term is, so
    once error removal is tiny the gap between "exactly zero" and "slightly
    above zero" swamps anything retention can say.

    This is not hypothetical -- it is why the pooled statistic ranked the
    always-correct arm above vanilla at n=300 while the pre-registered test
    found it significantly worse. The metric is pre-registered and is NOT being
    changed; it is flagged, and a raw count is reported beside it.
    """
    from truthforge.eval.comparison import _rates, net_quality_is_unstable

    changed_nothing = {"true_keep": 1000, "false_keep": 122, "true_fix": 0,
                       "false_fix": 0, "true_drop": 0, "false_drop": 0}
    fixed_one_broke_many = {"true_keep": 500, "false_keep": 121, "true_fix": 1,
                            "false_fix": 500, "true_drop": 0, "false_drop": 0}

    assert _rates(fixed_one_broke_many)[2] > _rates(changed_nothing)[2], (
        "the pathology this guard exists for has gone away; revisit the guard"
    )
    assert net_quality_is_unstable(changed_nothing)
    assert net_quality_is_unstable(fixed_one_broke_many)


def test_net_claims_improved_stays_interpretable_in_that_regime():
    """The raw count has no such pathology: it is monotone in both directions
    and a reader can check it by counting."""
    from truthforge.eval.comparison import ArmScore

    def arm(cells):
        return ArmScore(arm="x", n_turns=1, n_claims=1, cells=cells,
                        error_removal=0.0, retention=0.0, net_quality=0.0,
                        tokens_per_turn=0.0, latency_per_turn=0.0,
                        gold_source="annotated")

    nothing = arm({"true_keep": 1000, "false_keep": 122, "true_fix": 0,
                   "false_fix": 0, "true_drop": 0, "false_drop": 0})
    damaging = arm({"true_keep": 500, "false_keep": 121, "true_fix": 1,
                    "false_fix": 500, "true_drop": 0, "false_drop": 0})
    helpful = arm({"true_keep": 1000, "false_keep": 22, "true_fix": 100,
                   "false_fix": 5, "true_drop": 0, "false_drop": 0})

    assert nothing.net_claims_improved == 0
    assert damaging.net_claims_improved == -499
    assert helpful.net_claims_improved == 95
    assert damaging.net_claims_improved < nothing.net_claims_improved < helpful.net_claims_improved


def test_a_win_on_net_quality_that_loses_on_raw_counts_is_flagged():
    """The two measures encode different exchange rates and can disagree.

    This happened at n=300: TruthForge beat always-correct on the
    pre-registered metric with p=0.0195 while being worse on raw claim counts
    (-203 against -68), because net_quality is a harmonic mean of RATES --
    removing 73 of 125 errors is 58% of a small pool, withholding 274 of 1 964
    correct claims is 14% of a large one. Reporting only the flattering one
    would be a choice, not a measurement, so the conflict is named in the
    claim text itself and cannot be quoted without it.
    """
    from truthforge.eval.comparison import ArmScore, compare_arms

    def arm(name, cells, per_turn, tokens):
        return ArmScore(
            arm=name, n_turns=len(per_turn), n_claims=sum(cells.values()),
            cells=cells, error_removal=0.0, retention=0.0, net_quality=0.0,
            tokens_per_turn=tokens, latency_per_turn=1.0,
            gold_source="annotated", per_turn=per_turn,
            per_turn_tokens=[tokens] * len(per_turn),
            per_turn_cells=[dict(cells) for _ in per_turn],
        )

    # ours: removes many errors, withholds far more correct claims
    ours = arm("ours",
               {"true_keep": 700, "false_keep": 52, "true_fix": 73,
                "false_fix": 2, "true_drop": 0, "false_drop": 274},
               [0.9] * 40, 7989.0)
    theirs = arm("theirs",
                 {"true_keep": 900, "false_keep": 121, "true_fix": 1,
                  "false_fix": 69, "true_drop": 0, "false_drop": 0},
                 [0.1] * 40, 9789.0)

    r = compare_arms(ours, theirs, n_resamples=300)
    assert r["verdict"] in ("dominates", "equal quality, cheaper",
                            "better quality, not cheaper")
    assert r["net_claims_ours"] == 73 - 276
    assert r["net_claims_baseline"] == 1 - 69
    assert r["net_claims_conflict"] is True
    assert "WORSE" in r["claim"], "the conflict must travel with the claim text"


def test_no_conflict_is_reported_when_the_measures_agree():
    from truthforge.eval.comparison import ArmScore, compare_arms

    def arm(name, cells, per_turn, tokens):
        return ArmScore(
            arm=name, n_turns=len(per_turn), n_claims=sum(cells.values()),
            cells=cells, error_removal=0.0, retention=0.0, net_quality=0.0,
            tokens_per_turn=tokens, latency_per_turn=1.0,
            gold_source="annotated", per_turn=per_turn,
            per_turn_tokens=[tokens] * len(per_turn),
            per_turn_cells=[dict(cells) for _ in per_turn],
        )

    ours = arm("ours", {"true_keep": 900, "false_keep": 20, "true_fix": 80,
                        "false_fix": 1, "true_drop": 0, "false_drop": 2},
               [0.9] * 40, 5000.0)
    theirs = arm("theirs", {"true_keep": 850, "false_keep": 90, "true_fix": 10,
                            "false_fix": 50, "true_drop": 0, "false_drop": 0},
                 [0.2] * 40, 9000.0)

    r = compare_arms(ours, theirs, n_resamples=300)
    assert r["net_claims_conflict"] is False
    assert "WORSE" not in r["claim"]
