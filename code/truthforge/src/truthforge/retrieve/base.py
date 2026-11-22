"""Retrieval: BM25 + dense, fused with RRF.

BM25 is implemented here in pure Python -- no rank_bm25, no Java -- because it
keeps the whole pipeline runnable with zero heavy dependencies and because the
corpora in this project are small enough that an in-memory index is fine. Swap
it for a real index when the corpus stops fitting in RAM.

`GivenRetriever` is what RAGTruth uses: the grounding passages are fixed by the
dataset, so there is nothing to retrieve. That has a consequence the loop must
respect -- **the RERETRIEVE action is unavailable on RAGTruth**. Any arm
evaluated on RAGTruth therefore routes over three actions, not four, and that
must be stated next to every RAGTruth number. It is the main reason the QA
suite (Phase 2) is needed before the router's headline claim can be tested.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional, Protocol, Sequence

from ..types import Passage

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Retriever(Protocol):
    name: str
    supports_reretrieval: bool

    def retrieve(self, query: str, k: int = 5) -> list[Passage]: ...


class GivenRetriever:
    """Returns dataset-supplied passages. Retrieval is a no-op."""

    name = "given"
    supports_reretrieval = False

    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        return self.passages[:k] if k else self.passages


class BM25Index:
    """Okapi BM25 over an in-memory corpus."""

    name = "bm25"
    supports_reretrieval = True

    def __init__(
        self,
        docs: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs = list(docs)
        self.doc_ids = list(doc_ids) if doc_ids else [str(i) for i in range(len(docs))]
        if len(self.doc_ids) != len(self.docs):
            raise ValueError("doc_ids and docs must be the same length")
        self.k1, self.b = k1, b

        self.tokenized = [tokenize(d) for d in self.docs]
        self.lengths = [len(t) for t in self.tokenized]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.tokenized]

        df = Counter()
        for t in self.tokenized:
            df.update(set(t))
        n = len(self.docs)
        # Robertson-Sparck Jones IDF, floored at zero so very common terms
        # cannot contribute negative score.
        self.idf = {
            term: max(0.0, math.log(1 + (n - d + 0.5) / (d + 0.5)))
            for term, d in df.items()
        }

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = self.lengths[i] or 1
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            scores.append(s)

        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Passage(
                doc_id=self.doc_ids[i],
                text=self.docs[i],
                score=scores[i],
                source="bm25",
                rank=r,
            )
            for r, i in enumerate(order)
        ]


class DenseIndex:
    """Sentence-embedding retrieval. Loads torch lazily.

    `BAAI/bge-small-en-v1.5` is the default: ~130 MB, leaves the 8 GB card free
    for the generator and the judge.
    """

    name = "dense"
    supports_reretrieval = True

    def __init__(
        self,
        docs: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
        model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.docs = list(docs)
        self.doc_ids = list(doc_ids) if doc_ids else [str(i) for i in range(len(docs))]
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tok = None
        self._torch = None
        self._emb = None

    # bge models want this prefix on queries but not on documents.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def _ensure_built(self) -> None:
        """Encode the corpus using plain `transformers`.

        This was written against `sentence_transformers` and had never once
        run: the package is not installed, so the UI's dense toggle raised
        ModuleNotFoundError the moment anyone ticked it. The fix is not to
        install it. For a BERT-family bi-encoder, sentence_transformers does
        exactly what the six lines below do -- CLS pooling then L2 normalise --
        and `transformers` is already a hard dependency here. Adding a package
        to call a wrapper around a dependency we already have would cost disk
        for nothing.
        """
        if self._emb is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self._torch = torch
        self._emb = self._encode(self.docs)

    def _encode(self, texts: Sequence[str]):
        """CLS-pool and L2-normalise, in batches, under no_grad."""
        import numpy as np  # type: ignore

        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self._tok(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                # bge uses the CLS token, not mean pooling. Using the wrong one
                # silently degrades every score instead of failing.
                h = self._model(**enc).last_hidden_state[:, 0]
                h = torch.nn.functional.normalize(h, p=2, dim=1)
            out.append(h.float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        self._ensure_built()
        import numpy as np  # type: ignore

        q = self._encode([self.QUERY_PREFIX + query])
        sims = (self._emb @ q[0]).astype(float)
        order = np.argsort(-sims)[:k]
        return [
            Passage(
                doc_id=self.doc_ids[i],
                text=self.docs[i],
                score=float(sims[i]),
                source="dense",
                rank=r,
            )
            for r, i in enumerate(order)
        ]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Passage]], k: int = 5, rrf_k: int = 60
) -> list[Passage]:
    """Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (rrf_k + rank(d)), ranks 1-based. Scores
    from the individual retrievers are deliberately discarded -- that is the
    point of RRF, and it is why BM25 and dense scores never need calibrating
    against each other.
    """
    scores: dict[str, float] = {}
    best: dict[str, Passage] = {}
    for lst in ranked_lists:
        for rank, p in enumerate(lst, start=1):
            scores[p.doc_id] = scores.get(p.doc_id, 0.0) + 1.0 / (rrf_k + rank)
            best.setdefault(p.doc_id, p)

    order = sorted(scores, key=lambda d: -scores[d])[:k]
    return [
        Passage(
            doc_id=d,
            text=best[d].text,
            score=scores[d],
            source="rrf",
            rank=r,
            meta={"fused_from": best[d].source},
        )
        for r, d in enumerate(order)
    ]


class HybridRetriever:
    """BM25 + dense, fused with RRF. The retriever the paper reports."""

    name = "hybrid_rrf"
    supports_reretrieval = True

    def __init__(
        self,
        docs: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
        bm25_k: int = 20,
        dense_k: int = 20,
        rrf_k: int = 60,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        use_dense: bool = True,
    ) -> None:
        self.bm25 = BM25Index(docs, doc_ids)
        self.dense = (
            DenseIndex(docs, doc_ids, model=dense_model, device=device)
            if use_dense
            else None
        )
        self.bm25_k, self.dense_k, self.rrf_k = bm25_k, dense_k, rrf_k

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        lists = [self.bm25.retrieve(query, self.bm25_k)]
        if self.dense is not None:
            lists.append(self.dense.retrieve(query, self.dense_k))
        return rrf_fuse(lists, k=k, rrf_k=self.rrf_k)
