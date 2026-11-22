"""Dense retrieval: that it runs at all, and that the toggle is not inert.

Both of these were real bugs. `DenseIndex` was written against
`sentence_transformers`, which is not installed, so it had never executed; and
`DocumentStore.retriever` cached the index without recording which mode built
it, so a dense request after a BM25 request silently got BM25 back.

The model test is skipped when the weights are not present locally -- it must
not download 130 MB in CI -- but it is not deleted, because "dense retrieval
works" is a claim the paper makes.
"""

from __future__ import annotations

import os

import pytest

from truthforge.retrieve.base import BM25Index

CHUNKS = [
    "The office canteen serves hot meals between 12:00 and 14:00 daily.",
    "Expense claims over 500 must be countersigned by a department head.",
    "Parking permits are issued on a first-come, first-served basis.",
    "Fire drills are held quarterly and attendance is mandatory.",
    "Staff may take 25 days of paid annual leave each calendar year.",
]
IDS = [f"c{i}" for i in range(len(CHUNKS))]


def _weights_cached() -> bool:
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(cache, "models--BAAI--bge-small-en-v1.5"))


def test_retriever_cache_is_keyed_on_the_dense_flag():
    """The toggle must survive a prior call in the other mode."""
    from truthforge.docs.store import Chunk, DocumentStore

    store = DocumentStore()
    store.chunks = [
        Chunk(chunk_id=i, source="f.txt", page=1, text=t, start=0, end=len(t))
        for i, t in zip(IDS, CHUNKS)
    ]

    sparse = store.retriever(use_dense=False)
    assert isinstance(sparse, BM25Index)
    assert store.retriever(use_dense=False) is sparse       # cache still works

    class _Fake:
        pass

    # A dense request must not be answered with the cached sparse index.
    store._retriever, store._retriever_dense = _Fake(), True
    assert store.retriever(use_dense=True) is not sparse
    assert isinstance(store.retriever(use_dense=False), BM25Index)


def test_bm25_cannot_bridge_a_vocabulary_gap():
    """Establishes the premise of the dense test below: this query shares no
    content word with the chunk that answers it, so BM25 ranks it last."""
    ranked = [p.doc_id for p in BM25Index(CHUNKS, IDS).retrieve(
        "How much vacation time off do I get?", k=5)]
    assert ranked.index("c4") == len(CHUNKS) - 1


@pytest.mark.skipif(not _weights_cached(), reason="bge-small not cached locally")
def test_dense_bridges_the_gap_bm25_cannot():
    from truthforge.retrieve.base import DenseIndex, HybridRetriever

    q = "How much vacation time off do I get?"
    dense = [p.doc_id for p in DenseIndex(CHUNKS, IDS, device="cpu").retrieve(q, k=5)]
    hybrid = [p.doc_id for p in HybridRetriever(
        CHUNKS, IDS, device="cpu", use_dense=True).retrieve(q, k=5)]

    assert dense[0] == "c4"                     # dense finds it first
    assert hybrid.index("c4") < len(CHUNKS) - 1  # fusion carries it up too
