"""The question-level gate: does the evidence address the question at all?

This exists because of a measured failure, not a hypothetical one. The
open-corpus evaluation held each unanswerable question's own sources out of
the corpus; 44% of those questions still received a confident answer, and no
arm's refusal rate differed from the base generator's. The cause is structural:
claim-level support verification cannot see topic drift. Evidence about X plus
a question about Y yields an answer about X in which every claim is genuinely
supported by the evidence.
"""

from __future__ import annotations

from truthforge.claims.splitter import RuleSplitter
from truthforge.config import Config
from truthforge.detect.baselines import LexicalOverlapDetector, question_coverage
from truthforge.loop import LoopComponents, TruthForgeLoop
from truthforge.retrieve.base import BM25Index
from truthforge.router.rules import RuleRouter
from truthforge.types import Passage

CORPUS = [
    "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
    "The Louvre museum in Paris holds the Mona Lisa.",
    "Paris is the capital city of France.",
]
IDS = [f"c{i}" for i in range(len(CORPUS))]


def _loop(gate: bool, gen=None):
    cfg = Config()
    cfg.router.answerability_gate = gate
    return TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=RuleRouter(cfg.router),
            retriever=BM25Index(CORPUS, IDS),
            splitter=RuleSplitter(),
            generate_fn=gen or (lambda p: "The Eiffel Tower stands 330 metres tall."),
        ),
        cfg,
    )


# -- the signal -------------------------------------------------------------


def test_question_coverage_separates_on_topic_from_off_topic():
    ps = [Passage(doc_id="1", text=CORPUS[0])]
    assert question_coverage("How tall is the Eiffel Tower?", ps) == 1.0
    assert question_coverage("How do I remove oil stains from concrete?", ps) == 0.0


def test_question_words_do_not_count_as_coverage():
    """Leaving "what"/"how"/"is" in makes every question look half-answered,
    because those words appear in almost any prose."""
    ps = [Passage(doc_id="1", text="What is this? How does it work? It is here.")]
    assert question_coverage("How tall is the Eiffel Tower?", ps) == 0.0


def test_a_contentless_question_does_not_block():
    ps = [Passage(doc_id="1", text=CORPUS[0])]
    assert question_coverage("What?", ps) == 1.0


# -- the gate ---------------------------------------------------------------


def test_the_gate_refuses_an_off_topic_question_before_generating():
    """Refusing after generating would still be correct, but the point is that
    the check is cheap and the generation is not."""
    called = []

    def gen(prompt):
        called.append(prompt)
        return "Use a degreaser and a stiff brush."

    t = _loop(gate=True, gen=gen).run(
        question="How do I remove oil stains from a concrete driveway?"
    )
    assert t.abstained is True
    assert t.meta["refused_before_generating"] is True
    assert t.meta["question_coverage"] == 0.0
    assert not called, "the generator ran even though the gate refused"


def test_the_gate_lets_an_on_topic_question_through():
    t = _loop(gate=True).run(question="How tall is the Eiffel Tower?")
    assert not t.meta.get("refused_before_generating")
    assert t.meta["question_coverage"] == 1.0
    assert "330" in t.final_answer


def test_the_gate_is_off_by_default():
    t = _loop(gate=False).run(question="How do I remove oil stains from concrete?")
    assert not t.meta.get("refused_before_generating")
    assert t.meta.get("question_coverage") is None


def test_the_gate_does_not_fire_when_evidence_was_handed_in():
    """On RAGTruth the passages are paired with the question by the dataset.
    Relevance was decided elsewhere, so the gate would be scoring the
    benchmark's construction rather than the system's judgement."""
    cfg = Config()
    cfg.router.answerability_gate = True
    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=RuleRouter(cfg.router),
            retriever=BM25Index(CORPUS, IDS),
            splitter=RuleSplitter(),
            generate_fn=lambda p: "Use a degreaser and a stiff brush.",
        ),
        cfg,
    )
    t = loop.run(
        question="How do I remove oil stains from a concrete driveway?",
        passages=[Passage(doc_id="x", text=CORPUS[0])],   # deliberately irrelevant
    )
    assert not t.meta.get("refused_before_generating")


def test_the_threshold_is_the_one_selected_on_the_development_set():
    """0.55 came from scripts/select_answerability_threshold.py run on the
    train split with a different seed, before any evaluation number was seen.
    If someone retunes it, this test should make them say so out loud."""
    assert Config().router.min_question_coverage == 0.55


def test_the_gate_widens_the_search_before_refusing():
    """Low question coverage has two causes with the same symptom: the corpus
    does not hold the answer, or the retriever did not find it. Only the first
    justifies a refusal, so the gate looks again before giving up. This is the
    one place RE-RETRIEVE can prevent a WRONG refusal rather than merely feed
    another rewrite."""
    big = CORPUS + [
        "Electric vehicle charging points are available in the staff car park.",
        "Charging is free for employees and costs visitors two pounds an hour.",
    ] + [f"Filler passage number {i} about unrelated matters." for i in range(12)]
    ids = [f"c{i}" for i in range(len(big))]

    calls = []

    class CountingRetriever(BM25Index):
        supports_reretrieval = True

        def retrieve(self, query, k=5):
            calls.append(k)
            return super().retrieve(query, k)

    cfg = Config()
    cfg.router.answerability_gate = True
    cfg.retrieval.top_k = 1          # a deliberately stingy first pass
    cfg.retrieval.reretrieve_top_k = 10
    # Set explicitly rather than relying on the shipped 0.55: this test is
    # about the MECHANISM (widen, then re-check), and tying it to the
    # production threshold would make it fail for an unrelated reason the day
    # that threshold moves. On this corpus top-1 gives coverage 0.6 and the
    # widened set gives 1.0, so 0.9 puts the boundary between them.
    cfg.router.min_question_coverage = 0.9

    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=RuleRouter(cfg.router),
            retriever=CountingRetriever(big, ids),
            splitter=RuleSplitter(),
            generate_fn=lambda p: "Charging is free for employees.",
        ),
        cfg,
    )
    t = loop.run(question="Is electric vehicle charging free for employees?")

    assert len(calls) == 2, "the gate did not widen the search"
    assert calls == [1, 10]
    assert t.meta.get("answerability_reretrieved") is True
    assert (
        t.meta["question_coverage"]
        > t.meta["question_coverage_before_reretrieval"]
    )
    assert not t.meta.get("refused_before_generating")


def test_a_genuinely_absent_topic_is_still_refused_after_widening():
    class Wide(BM25Index):
        supports_reretrieval = True

    cfg = Config()
    cfg.router.answerability_gate = True
    loop = TruthForgeLoop(
        LoopComponents(
            detector=LexicalOverlapDetector(),
            router=RuleRouter(cfg.router),
            retriever=Wide(CORPUS, IDS),
            splitter=RuleSplitter(),
            generate_fn=lambda p: "Use a degreaser.",
        ),
        cfg,
    )
    t = loop.run(question="How do I remove oil stains from a concrete driveway?")
    assert t.meta["refused_before_generating"] is True
    assert t.meta["answerability_reretrieved"] is True


# -- abstention confirmed by a second verifier ------------------------------


class _AlwaysAbstain:
    name = "always_abstain"

    def route(self, state, passages):
        from truthforge.router.base import decision
        from truthforge.types import Action

        if not state.verdict.is_flagged:
            return decision(state, Action.ACCEPT, "not_flagged", self.name)
        return decision(state, Action.ABSTAIN, "test_abstain", self.name)


class _FlagEverything:
    name = "flag_all"

    def check(self, claims, passages):
        from truthforge.types import Label, Verdict

        return [Verdict(claim_id=c.claim_id, label=Label.UNSUPPORTED, support_prob=0.1,
                        detector=self.name) for c in claims]


def _confirm_loop(confirm):
    cfg = Config()
    return TruthForgeLoop(
        LoopComponents(
            detector=_FlagEverything(), router=_AlwaysAbstain(),
            retriever=BM25Index(CORPUS, IDS), splitter=RuleSplitter(),
            confirm_fn=confirm,
        ),
        cfg,
    )


def test_an_unconfirmed_abstention_becomes_an_accept():
    """Every abstention inherits the precision of the flag behind it. When a
    second verifier does not back the flag, the claim is kept and the ledger
    records why, so the number of overruled abstentions is countable."""
    t = _confirm_loop(lambda claim, passages: False).run(
        question="How tall is the Eiffel Tower?",
        answer="The Eiffel Tower stands 330 metres tall.",
        passages=[Passage(doc_id="p", text=CORPUS[0])],
    )
    assert t.meta["abstained_claim_ids"] == []
    assert any(d.reason.startswith("abstain_not_confirmed") for d in t.decisions)
    assert any(c.stage == "confirm" for c in t.costs), "confirmation cost not recorded"


def test_a_confirmed_abstention_still_withholds():
    t = _confirm_loop(lambda claim, passages: True).run(
        question="How tall is the Eiffel Tower?",
        answer="The Eiffel Tower stands 330 metres tall.",
        passages=[Passage(doc_id="p", text=CORPUS[0])],
    )
    assert len(t.meta["abstained_claim_ids"]) == 1
    assert any(d.reason.endswith("+confirmed") for d in t.decisions)


def test_without_a_confirming_verifier_nothing_changes():
    t = _confirm_loop(None).run(
        question="How tall is the Eiffel Tower?",
        answer="The Eiffel Tower stands 330 metres tall.",
        passages=[Passage(doc_id="p", text=CORPUS[0])],
    )
    assert len(t.meta["abstained_claim_ids"]) == 1
    assert not any(c.stage == "confirm" for c in t.costs)
