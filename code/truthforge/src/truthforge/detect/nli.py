"""NLI-backed detection and contradiction analysis.

This module closes weak link #2 in DECISIONS.md. The previous
`contradiction_signal` compared numeric literals across passages, which was a
placeholder and said so. This replaces it with a real entailment model.

Two distinct jobs, deliberately kept apart:

  1. ``NLIDetector`` -- claim vs. evidence. Is the claim entailed?
     Scored per passage and aggregated with a max, because a claim supported
     by any single passage is supported; requiring all passages to entail it
     would punish multi-topic evidence.

  2. ``ContradictionAnalyzer`` -- passage vs. passage, *given* a claim. Do the
     sources disagree with each other about this claim? This is what the
     ABSTAIN branch needs, and it is a fundamentally different question from
     "is the claim supported". A claim can be unsupported because the evidence
     is silent (→ re-retrieve) or because the evidence fights itself
     (→ abstain), and only the second one is unfixable by rewriting.

Default model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (~184M, ~400 MB
fp16). Fits alongside MiniCheck and a 3B corrector on 8 GB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..types import Claim, Passage
from .base import Detector

# MNLI label order varies by checkpoint. Resolved from the model config at load
# time rather than hard-coded -- getting this backwards silently inverts every
# verdict, and nothing downstream would notice.
_ENTAIL_NAMES = {"entailment", "entail", "label_0"}
_CONTRA_NAMES = {"contradiction", "contra", "label_2"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split a premise into sentences, honouring line breaks as well.

    Splitting on terminal punctuation alone fails on exactly the material
    uploaded documents are made of: headings, numbered sections and bullet
    items carry no full stop, so "Handbook\\n\\nSection 4: ..." stays glued
    together. The claim splitter had the same bug; here it meant a citation
    shown to a reader was a whole block when one line carried the fact.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _SENT_SPLIT.split(line):
            part = part.strip()
            if len(part) >= min_chars:
                out.append(part)
    return out


@dataclass
class NLIScores:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def supported(self) -> float:
        """P(supported) for the router.

        Neutral is *not* support. A claim the evidence is silent about scores
        low here, which is what routes it to re-retrieval rather than rewriting.
        """
        return self.entailment


class _NLIBackend:
    """Shared lazy-loaded cross-encoder. Importing torch is deferred."""

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._idx: dict[str, int] = {}

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        from .._hf import load_kwargs, resolve_device, with_dtype_fallback

        device = resolve_device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            with_dtype_fallback(
                AutoModelForSequenceClassification.from_pretrained,
                self.model_name,
                **load_kwargs(device, self.dtype),
            )
            .to(device)
            .eval()
        )
        self.device = device
        self._idx = self._resolve_label_indices()

    def _resolve_label_indices(self) -> dict[str, int]:
        """Map entailment/neutral/contradiction onto this checkpoint's logits."""
        id2label = getattr(self._model.config, "id2label", None) or {}
        idx: dict[str, int] = {}
        for i, name in id2label.items():
            low = str(name).strip().lower()
            if low in _ENTAIL_NAMES:
                idx["entailment"] = int(i)
            elif low in _CONTRA_NAMES:
                idx["contradiction"] = int(i)
            elif low.startswith("neutral"):
                idx["neutral"] = int(i)

        if len(idx) != 3:
            raise RuntimeError(
                f"could not resolve MNLI label order from id2label={id2label!r}. "
                "Refusing to guess: a wrong mapping silently inverts every "
                "verdict. Set it explicitly for this checkpoint."
            )
        return idx

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        """Score (premise, hypothesis) pairs."""
        self.ensure_loaded()
        import torch  # type: ignore

        out: list[NLIScores] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self._tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                return_tensors="pt",
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1)

            for row in probs:
                out.append(
                    NLIScores(
                        entailment=float(row[self._idx["entailment"]]),
                        neutral=float(row[self._idx["neutral"]]),
                        contradiction=float(row[self._idx["contradiction"]]),
                    )
                )
        return out


# Backends are expensive and stateless once loaded, so they are shared across
# the detector and the analyzer instead of loaded twice.
_BACKENDS: dict[tuple, _NLIBackend] = {}


def get_backend(
    model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cuda",
    dtype: str = "float16",
    batch_size: int = 16,
) -> _NLIBackend:
    key = (model, device, dtype, batch_size)
    if key not in _BACKENDS:
        _BACKENDS[key] = _NLIBackend(model, device, dtype, batch_size)
    return _BACKENDS[key]


class NLIDetector(Detector):
    """Claim-vs-evidence entailment, aggregated over passages with a max.

    Long passages are split into sentences and scored individually, then maxed.
    Feeding a 2000-token passage as one premise buries the one sentence that
    matters; this is the single biggest quality difference between a naive NLI
    detector and a useful one.
    """

    name = "nli"

    def __init__(
        self,
        flag_threshold: float = 0.5,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 16,
        sentence_level: bool = True,
        max_premises: int = 64,
        specificity_margin: float = 0.05,
    ) -> None:
        super().__init__(flag_threshold)
        self.backend = get_backend(model, device, dtype, batch_size)
        self.sentence_level = sentence_level
        self.max_premises = max_premises
        # How much entailment a citation may give up to be more specific.
        # Only affects which premise is SHOWN; scoring is untouched.
        self.specificity_margin = specificity_margin

    def _premises(self, passages: Sequence[Passage]) -> list[str]:
        """Build the premise set: whole passages AND their sentences.

        Sentence-level premises are sharper -- a 2000-token passage as one
        premise buries the sentence that matters -- but splitting destroys
        coreference. The probe set caught exactly this: given "The Eiffel
        Tower was completed in 1889. It stands 330 metres tall.", the claim
        "The Eiffel Tower stands 330 metres tall" is unentailed by either
        sentence alone, because the second one says only "It".

        Keeping the full passage in the premise set fixes that at the cost of
        one extra pair per passage, and the max-aggregation means the sharper
        premise still wins whenever it fires.
        """
        if not self.sentence_level:
            return [p.text for p in passages]

        premises: list[str] = []
        seen: set[str] = set()

        def add(text: str) -> None:
            t = text.strip()
            if t and t not in seen:
                seen.add(t)
                premises.append(t)

        for p in passages:
            add(p.text)
            for s in _sentences(p.text):
                add(s)

        return premises[: self.max_premises]

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        premises = self._premises(passages)
        if not premises:
            return [0.0] * len(claims)

        pairs = [(prem, c.text) for c in claims for prem in premises]
        scored = self.backend.score_pairs(pairs)

        n = len(premises)
        return [
            max(s.entailment for s in scored[i * n : (i + 1) * n])
            for i in range(len(claims))
        ]

    def detail(
        self, claims: Sequence[Claim], passages: Sequence[Passage]
    ) -> list[dict]:
        """Per-claim breakdown, for the UI and for error analysis.

        Returns the best-entailing and best-contradicting premise for each
        claim, which is what makes a verdict explainable to a reader.
        """
        premises = self._premises(passages)
        if not premises:
            return [{"entailment": 0.0, "contradiction": 0.0} for _ in claims]

        pairs = [(prem, c.text) for c in claims for prem in premises]
        scored = self.backend.score_pairs(pairs)

        n = len(premises)
        out = []
        for i in range(len(claims)):
            row = scored[i * n : (i + 1) * n]
            best_e = max(range(n), key=lambda j: row[j].entailment)
            best_c = max(range(n), key=lambda j: row[j].contradiction)

            # Cite the most SPECIFIC premise, not merely the highest-scoring
            # one. The premise set holds whole passages as well as their
            # sentences, and a whole passage usually edges out the single
            # sentence inside it -- so the citation shown to a reader would be
            # a wall of text when one line actually carries the fact. Among
            # premises that entail about as well as the best, take the
            # shortest.
            top = row[best_e].entailment
            close = [
                j for j in range(n)
                if row[j].entailment >= top - self.specificity_margin
            ]
            cite = min(close, key=lambda j: len(premises[j])) if close else best_e

            out.append(
                {
                    "entailment": row[best_e].entailment,
                    "contradiction": row[best_c].contradiction,
                    "neutral": row[best_e].neutral,
                    "best_entailing_premise": premises[cite],
                    "best_contradicting_premise": premises[best_c],
                }
            )
        return out


class ContradictionAnalyzer:
    """Do the sources disagree with each other about this claim?

    Method: find the passage sentences that are *relevant* to the claim (either
    entail or contradict it above a floor), then check whether relevant
    sentences disagree -- one entails while another contradicts.

    This is deliberately not "does any passage contradict the claim". A single
    contradicting passage means the claim is wrong and a rewrite can fix it.
    Disagreement *between sources* is the unfixable case, and the only one that
    justifies abstaining.
    """

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 16,
        relevance_floor: float = 0.35,
        max_premises: int = 48,
    ) -> None:
        self.backend = get_backend(model, device, dtype, batch_size)
        self.relevance_floor = relevance_floor
        self.max_premises = max_premises

    def score(self, claim_text: str, passages: Sequence[Passage]) -> float:
        return self.analyze(claim_text, passages)["contradiction"]

    def analyze(self, claim_text: str, passages: Sequence[Passage]) -> dict:
        premises: list[tuple[str, str]] = []  # (doc_id, sentence)
        for p in passages:
            for s in _sentences(p.text) or [p.text]:
                premises.append((p.doc_id, s))
        premises = premises[: self.max_premises]

        if len(premises) < 2:
            return {"contradiction": 0.0, "reason": "fewer than two premises"}

        scored = self.backend.score_pairs([(s, claim_text) for _, s in premises])

        relevant = [
            (doc_id, sent, sc)
            for (doc_id, sent), sc in zip(premises, scored)
            if max(sc.entailment, sc.contradiction) >= self.relevance_floor
        ]
        if len(relevant) < 2:
            return {"contradiction": 0.0, "reason": "fewer than two relevant premises"}

        pro = max(relevant, key=lambda r: r[2].entailment)
        con = max(relevant, key=lambda r: r[2].contradiction)

        # Both directions must be strongly represented for this to be a real
        # source conflict. The product means one weak side kills the signal.
        signal = pro[2].entailment * con[2].contradiction

        return {
            "contradiction": float(signal),
            "supporting_doc_id": pro[0],
            "supporting_sentence": pro[1],
            "supporting_entailment": pro[2].entailment,
            "conflicting_doc_id": con[0],
            "conflicting_sentence": con[1],
            "conflicting_contradiction": con[2].contradiction,
            "cross_document": pro[0] != con[0],
            "reason": "entailing and contradicting premises both present"
            if signal > 0
            else "no disagreement",
        }


def build_contradiction_fn(cfg, required: bool = False):
    """Return a `(claim_text, passages) -> float` source-conflict scorer.

    Falls back to the lexical placeholder in `detect.baselines` when the NLI
    model cannot load, and warns loudly -- silently degrading here would leave
    the ABSTAIN branch running on a signal that DECISIONS.md declares
    meaningless, while the run log claimed otherwise.

    Pass `required=True` to make a load failure fatal instead.
    """
    import warnings

    try:
        analyzer = ContradictionAnalyzer(
            model=getattr(cfg.detector, "nli_model",
                          "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
            device=cfg.detector.device,
            dtype=cfg.detector.dtype,
            batch_size=cfg.detector.batch_size,
        )
        analyzer.backend.ensure_loaded()
        return analyzer.score
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        from .baselines import contradiction_signal

        warnings.warn(
            f"NLI contradiction analyzer unavailable ({type(exc).__name__}: "
            f"{exc}); falling back to the lexical placeholder. The "
            "`contradictory_sources` routing reason is NOT evidence of "
            "anything in this run -- see weak link #2 in DECISIONS.md.",
            RuntimeWarning,
            stacklevel=2,
        )
        return contradiction_signal
