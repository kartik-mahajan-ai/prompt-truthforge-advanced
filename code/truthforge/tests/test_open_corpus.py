"""The open-corpus construction must actually hold the answer out.

If a held-out passage leaks back into the corpus, an "unanswerable" question
is answerable and the refusal numbers mean nothing -- while still looking
entirely reasonable. RAGTruth reuses one source across several responses, so
the leak is the default behaviour, not an edge case.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from truthforge.types import Passage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "eval_open_corpus", ROOT / "scripts" / "eval_open_corpus.py"
)
eoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eoc)


class Ex:
    def __init__(self, eid, source_id, texts):
        self.example_id = eid
        self.question = f"q{eid}"
        self.passages = [
            Passage(doc_id=f"{source_id}:{i}", text=t) for i, t in enumerate(texts)
        ]


def test_held_out_passages_are_absent_from_the_corpus():
    examples = [Ex(str(i), f"s{i}", [f"text {i}a", f"text {i}b"]) for i in range(10)]
    texts, ids, questions = eoc.build_corpus(examples, 3, 3, seed=1)

    unanswerable = {q["example_id"] for q in questions if not q["answerable"]}
    assert len(unanswerable) == 3

    banned = {p.text for ex in examples if ex.example_id in unanswerable
              for p in ex.passages}
    assert not (banned & set(texts)), "a held-out passage leaked into the corpus"


def test_a_shared_source_does_not_leak_the_answer_back():
    """Two examples over the SAME source id. Holding out one must remove the
    text, not leave it reachable through its sibling -- keying the exclusion on
    example_id instead of source_id would pass a naive test and fail here."""
    shared = ["the secret is 42"]
    examples = [
        Ex("a", "s0", shared),          # held out
        Ex("b", "s0", shared),          # same source, not held out
        Ex("c", "s1", ["unrelated 1"]),
        Ex("d", "s2", ["unrelated 2"]),
    ]
    texts, _, questions = eoc.build_corpus(examples, 1, 1, seed=0)
    unanswerable = {q["example_id"] for q in questions if not q["answerable"]}

    if unanswerable & {"a", "b"}:
        assert "the secret is 42" not in texts


def test_answerable_questions_keep_their_evidence():
    examples = [Ex(str(i), f"s{i}", [f"only text {i}"]) for i in range(8)]
    texts, _, questions = eoc.build_corpus(examples, 4, 2, seed=3)
    by_id = {e.example_id: e for e in examples}
    for q in questions:
        if q["answerable"]:
            for p in by_id[q["example_id"]].passages:
                assert p.text in texts


def test_refuses_to_run_when_there_are_too_few_examples():
    with pytest.raises(SystemExit):
        eoc.build_corpus([Ex("0", "s0", ["t"])], 5, 5, seed=0)


def test_summarise_reports_the_two_refusal_rates_separately():
    """A single combined score would hide which way an arm fails, and failing
    by refusing everything is not the same as failing by answering everything."""
    rows = [
        {"answerable": True,  "refused": False, "reretrieves": 1, "rewrites": 0,
         "latency_s": 1.0, "reasons": ["a"]},
        {"answerable": True,  "refused": True,  "reretrieves": 0, "rewrites": 1,
         "latency_s": 1.0, "reasons": ["b"]},
        {"answerable": False, "refused": True,  "reretrieves": 2, "rewrites": 0,
         "latency_s": 1.0, "reasons": ["a"]},
        {"answerable": False, "refused": False, "reretrieves": 0, "rewrites": 0,
         "latency_s": 1.0, "reasons": []},
    ]
    s = eoc.summarise(rows)
    assert s["correct_refusal"] == 0.5
    assert s["over_refusal"] == 0.5
    assert s["turns_with_reretrieve"] == 2
    assert s["reretrieve_per_turn"] == 0.75
    assert s["routing_reasons"]["a"] == 2


def test_a_generator_refusal_counts_as_a_refusal():
    """The first version of score_turn counted only router-driven abstention
    and reported 0.0 correct refusals for every arm. The generator was in fact
    declining on 56% of unanswerable questions in its own words -- the system
    refused often and the instrument could not see it."""
    from truthforge.types import TurnResult

    t = TurnResult(
        question="how do I remove oil stains?", turn_id="t1",
        initial_answer="x", example_id="e1",
        final_answer=(
            "The passages do not contain information about oil stains on "
            "concrete driveways."
        ),
    )
    row = eoc.score_turn(t, answerable=False)
    assert row["refused"] is True
    assert row["text_refused"] is True
    # ...but the ROUTER did nothing, and the two must stay distinguishable:
    # "the model happened to say it did not know" is not the capability this
    # project claims.
    assert row["router_refused"] is False


def test_a_real_answer_is_not_scored_as_a_refusal():
    from truthforge.types import TurnResult

    t = TurnResult(
        question="how do I cook prime rib?", turn_id="t2",
        initial_answer="x", example_id="e2",
        final_answer=(
            "To cook a small prime rib roast, preheat the oven to 350 degrees. "
            "Season the prime rib with paprika and black pepper. Bake until "
            "the internal temperature reaches 130 degrees."
        ),
    )
    row = eoc.score_turn(t, answerable=True)
    assert row["refused"] is False


def test_summarise_separates_router_refusals_from_generator_refusals():
    rows = [
        {"answerable": False, "refused": True, "router_refused": False,
         "text_refused": True, "reretrieves": 0, "rewrites": 0,
         "latency_s": 1.0, "reasons": []},
        {"answerable": False, "refused": True, "router_refused": True,
         "text_refused": False, "reretrieves": 0, "rewrites": 0,
         "latency_s": 1.0, "reasons": []},
        {"answerable": True, "refused": False, "router_refused": False,
         "text_refused": False, "reretrieves": 0, "rewrites": 0,
         "latency_s": 1.0, "reasons": []},
    ]
    s = eoc.summarise(rows)
    assert s["correct_refusal"] == 1.0
    assert s["correct_refusal_by_router"] == 0.5
    assert s["over_refusal"] == 0.0


def test_a_gate_refusal_is_counted_as_a_refusal():
    """The answerability gate refuses BEFORE generating: no claims exist, and
    the answer is the system's own abstain message, which the refusal-phrase
    regex does not match. Inferring refusal from the text, the scorer counted
    0 of the gate's 29 refusals and reported that switching the gate on cut
    correct refusals from 0.56 to 0.04. The loop knows when it refused; the
    scorer has to ask it."""
    from truthforge.config import Config
    from truthforge.types import TurnResult

    t = TurnResult(
        question="How do I remove oil stains?", turn_id="t1",
        initial_answer="", example_id="e1",
        final_answer=Config().loop.abstain_message,
        abstained=True,
    )
    t.meta["refused_before_generating"] = True

    row = eoc.score_turn(t, answerable=False)
    assert row["refused"] is True
    assert row["router_refused"] is True, "the gate IS the router refusing"


def test_rescore_recognises_a_saved_gate_refusal(tmp_path):
    """Saved rows do not carry the loop's flag, so the offline rescorer has to
    recognise the system's own abstain message with no claims behind it."""
    import importlib.util
    import json

    from truthforge.config import Config

    spec = importlib.util.spec_from_file_location(
        "rescore", ROOT / "scripts" / "rescore_open_corpus.py")
    rescore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rescore)

    msg = Config().loop.abstain_message
    art = {
        "config": {},
        "arms": {"rule": {"summary": {"reretrieve_per_turn": 0.0}, "rows": [
            {"example_id": "1", "answerable": False, "refused": False,
             "router_refused": False, "n_claims": 0, "answer": msg},
            {"example_id": "2", "answerable": True, "refused": False,
             "router_refused": False, "n_claims": 3,
             "answer": "Paris is the capital of France."},
        ]}},
    }
    f = tmp_path / "oc.json"
    f.write_text(json.dumps(art), encoding="utf-8")
    sys_argv = sys.argv
    try:
        sys.argv = ["rescore", str(f)]
        rescore.main()
    finally:
        sys.argv = sys_argv
    out = json.loads(f.read_text(encoding="utf-8"))["arms"]["rule"]
    assert out["summary"]["correct_refusal"] == 1.0
    assert out["summary"]["correct_refusal_by_router"] == 1.0
    assert out["summary"]["over_refusal"] == 0.0
