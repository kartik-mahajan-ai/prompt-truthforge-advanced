"""Trivial detectors.

Gate 1 in DECISIONS.md: MiniCheck must beat ALL of these by a margin worth
reporting. If it does not, the claim splitter is broken and building the loop
is wasted work.

`LexicalOverlapDetector` is the one that matters. Majority-class and random
baselines are easy to beat; a bag-of-words overlap score is not, and a
detection paper that does not report it is not reporting anything.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

from ..types import Claim, Passage
from .base import Detector, evidence_text

_WORD = re.compile(r"[a-z0-9]+")

# Function words carry no evidential signal and swamp the overlap score.
_STOP = frozenset(
    """a an and are as at be been being but by for from has have had he her his
    i in is it its of on or she that the their them they this to was were what
    when where which who will with would you your not no nor do does did can
    could should may might must there here than then so if""".split()
)


def _tokens(text: str, drop_stop: bool = True) -> list[str]:
    toks = _WORD.findall(text.lower())
    return [t for t in toks if t not in _STOP] if drop_stop else toks


class AlwaysSupportedDetector(Detector):
    """Majority class on most datasets. Flags nothing, so the loop never fires:
    this is the floor that `over-correction rate` is trivially zero at."""

    name = "always_supported"

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        return [1.0] * len(claims)


class AlwaysUnsupportedDetector(Detector):
    """Flags everything. Equivalent to the `always-correct` arm's detector, and
    the upper bound on recall at the worst possible precision."""

    name = "always_unsupported"

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        return [0.0] * len(claims)


class RandomDetector(Detector):
    """Seeded coin flip. Sanity check that the eval harness is not scoring noise
    as signal."""

    name = "random"

    def __init__(self, flag_threshold: float = 0.5, seed: int = 20260920) -> None:
        super().__init__(flag_threshold)
        self.rng = random.Random(seed)

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        return [self.rng.random() for _ in claims]


class LexicalOverlapDetector(Detector):
    """IDF-weighted recall of claim tokens against the evidence.

    Intuition: an unsupported claim contains content words that appear nowhere
    in the retrieved passages. Rare words are weighted up, so matching "the" is
    worth nothing and matching "1997" is worth a lot.

    Score is squashed through a logistic so the router sees a graded
    probability rather than a raw ratio.
    """

    name = "lexical_overlap"

    def __init__(
        self,
        flag_threshold: float = 0.5,
        midpoint: float = 0.55,
        steepness: float = 8.0,
    ) -> None:
        super().__init__(flag_threshold)
        self.midpoint = midpoint
        self.steepness = steepness

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        docs = [_tokens(p.text) for p in passages] or [[]]
        n_docs = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        evidence_vocab = set().union(*(set(d) for d in docs)) if docs else set()

        def idf(tok: str) -> float:
            # Smoothed IDF; unseen tokens get the max weight.
            return math.log((n_docs + 1) / (df.get(tok, 0) + 0.5))

        scores = []
        for c in claims:
            toks = _tokens(c.text)
            if not toks:
                scores.append(1.0)  # nothing asserted -> nothing to contradict
                continue
            total = sum(idf(t) for t in toks)
            matched = sum(idf(t) for t in toks if t in evidence_vocab)
            recall = matched / total if total > 0 else 0.0
            scores.append(_logistic(recall, self.midpoint, self.steepness))
        return scores


def _logistic(x: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


BASELINES = {
    "always_supported": AlwaysSupportedDetector,
    "always_unsupported": AlwaysUnsupportedDetector,
    "random": RandomDetector,
    "lexical": LexicalOverlapDetector,
    "lexical_overlap": LexicalOverlapDetector,
}


def coverage(claim_text: str, passages: list[Passage]) -> float:
    """Fraction of a claim's content tokens present in the evidence.

    The router uses this to tell "we retrieved the wrong thing" (thin coverage
    -> re-retrieve) from "we retrieved the right thing and misread it"
    (good coverage, low support -> rewrite). Kept separate from the detector so
    the two signals stay independent.
    """
    toks = _tokens(claim_text)
    if not toks:
        return 1.0
    vocab = set()
    for p in passages:
        vocab.update(_tokens(p.text))
    return sum(1 for t in toks if t in vocab) / len(toks)


# Question words that carry no topical content. Keeping "what" or "how" in the
# coverage calculation makes every question look partly answered, because those
# words appear in almost any prose.
_WH_STOPWORDS = frozenset(
    "what when where which who whom whose why how does do did is are was were "
    "can could should would will shall may might must the a an of in on at to "
    "for from with about into over after before between and or but if then "
    "than that this these those there here it its they them their you your "
    "i me my we our us be been being have has had get got make made use used "
    "any some much many best good better way ways".split()
)


def question_coverage(question: str, passages: list[Passage]) -> float:
    """Fraction of a QUESTION's content words present in the evidence.

    This answers a different question from `coverage`, and the difference is
    the whole point. `coverage` asks "is this claim's content in the evidence?"
    -- it polices what the generator wrote. This asks "is the thing being asked
    about in the evidence at all?" -- it polices whether there is anything to
    write.

    Why a separate signal was needed. Claim-level support verification is
    structurally blind to topic drift: if retrieval returns passages about X
    and the question is about Y, the generator writes about X, and every claim
    it writes IS genuinely supported by the evidence. Faithfulness is perfect
    and the answer is useless. The open-corpus evaluation measured exactly
    this -- 44% of questions whose evidence had been held out of the corpus
    received a confident answer, and no arm's refusal rate differed from the
    base generator's.

    On that data this separates answerable from unanswerable at AUROC 0.830
    (answerable mean 0.775, unanswerable 0.427), which is why it is worth
    having despite being nothing but a word overlap. It is deliberately cheap:
    the finding is that the pipeline needs SOME question-level signal, not that
    this particular one is the best available.
    """
    toks = [t for t in _tokens(question) if t not in _WH_STOPWORDS]
    if not toks:
        # Nothing contentful to look for; do not block on it.
        return 1.0
    vocab = set()
    for p in passages:
        vocab.update(_tokens(p.text))
    return sum(1 for t in toks if t in vocab) / len(toks)


def contradiction_signal(claim_text: str, passages: list[Passage]) -> float:
    """Crude disagreement detector across passages, in [0, 1].

    High when the claim's content words are well covered by several passages
    individually but those passages disagree on the numbers/entities near them.
    This is a placeholder: it only checks whether passages with similar
    coverage carry different numeric literals. Replace with an NLI pass over
    passage pairs before anything goes in the paper -- it is flagged in
    DECISIONS.md as a known weak link.
    """
    toks = set(_tokens(claim_text))
    if not toks:
        return 0.0

    relevant: list[set[str]] = []
    for p in passages:
        ptoks = _tokens(p.text)
        if not ptoks:
            continue
        if len(toks & set(ptoks)) / len(toks) >= 0.4:
            relevant.append({t for t in ptoks if t.isdigit()})

    if len(relevant) < 2:
        return 0.0
    numeric_sets = [s for s in relevant if s]
    if len(numeric_sets) < 2:
        return 0.0
    union = set().union(*numeric_sets)
    inter = set.intersection(*numeric_sets)
    return 1.0 - (len(inter) / len(union)) if union else 0.0
