"""Tests for the LLM outcome annotator.

No model is loaded: the judge takes a plain callable, so a scripted stand-in
exercises the parsing and the failure handling, which is where the quiet
mistakes live.
"""

from __future__ import annotations

from truthforge.eval.llm_judge import LLMJudge, label_batch
from truthforge.types import Action, CorrectionAttempt, Outcome, Passage, WorseKind

EV = [Passage("p0", "The Eiffel Tower was completed in 1889 and is 330 metres tall.")]


def judge_returning(*replies):
    seq = list(replies)
    return LLMJudge(lambda prompt: seq.pop(0) if seq else "", name="stub")


def test_each_label_is_parsed():
    cases = {
        "BETTER": (Outcome.BETTER, None),
        "SAME": (Outcome.SAME, None),
        "INFO_LOSS": (Outcome.WORSE, WorseKind.INFO_LOSS),
        "HEDGING": (Outcome.WORSE, WorseKind.HEDGING),
        "NEW_ERROR": (Outcome.WORSE, WorseKind.NEW_ERROR),
    }
    for reply, (outcome, kind) in cases.items():
        r = judge_returning(reply).judge("before text here", "after text here", EV)
        assert r.parsed
        assert r.outcome is outcome
        assert r.worse_kind is kind


def test_label_is_found_inside_a_chatty_reply():
    r = judge_returning("LABEL: NEW_ERROR -- it invented a figure").judge(
        "before text here", "after text here", EV
    )
    assert r.outcome is Outcome.WORSE
    assert r.worse_kind is WorseKind.NEW_ERROR


def test_info_loss_is_not_shadowed_by_a_shorter_label():
    """Longest-first matching: INFO_LOSS must not be read as something else."""
    r = judge_returning("INFO_LOSS").judge("a", "b", EV)
    assert r.worse_kind is WorseKind.INFO_LOSS


def test_unchanged_text_needs_no_model_call():
    called = []
    j = LLMJudge(lambda p: called.append(p) or "BETTER", name="stub")
    r = j.judge("identical sentence", "identical sentence", EV)
    assert r.outcome is Outcome.SAME
    assert not called, "no model call should be made for an unchanged attempt"


def test_an_unparseable_reply_is_left_unlabelled_not_recorded_as_same():
    """Recording a failed parse as SAME would inflate agreement with the
    heuristic, which also favours SAME."""
    r = judge_returning("I'm not sure about this one.").judge("before", "after", EV)
    assert not r.parsed
    assert r.outcome is None


def test_batch_reports_how_often_the_judge_failed():
    attempts = [
        CorrectionAttempt(claim_id=f"c{i}", action=Action.REWRITE, iteration=0,
                          before_text=f"before {i}", after_text=f"after {i}",
                          before_support_prob=0.1)
        for i in range(3)
    ]
    j = judge_returning("BETTER", "nonsense", "HEDGING")
    out = label_batch(j, attempts, {a.claim_id: EV for a in attempts})
    assert out["n_attempts"] == 3
    assert out["n_labelled"] == 2
    assert out["n_unparseable"] == 1
    assert out["label_kind"] == "automatic"
    assert all(v["annotator_id"] == "llm:stub" for v in out["labels"].values())


def test_label_attempt_stamps_the_source():
    a = CorrectionAttempt(claim_id="c1", action=Action.REWRITE, iteration=0,
                          before_text="before", after_text="after",
                          before_support_prob=0.1)
    judge_returning("INFO_LOSS").label_attempt(a, EV)
    assert a.outcome is Outcome.WORSE
    assert a.worse_kind is WorseKind.INFO_LOSS
    assert a.outcome_source == "llm:stub"
