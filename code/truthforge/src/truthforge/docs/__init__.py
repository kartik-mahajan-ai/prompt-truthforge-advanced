"""Uploaded-document question answering.

This is the application layer, and it exists for a reason beyond being a demo:
it is the only configuration in which the router's RE-RETRIEVE action can
actually fire.

On RAGTruth the grounding passages are fixed by the dataset, so "go look again"
has nowhere to look and degrades into deleting the sentence (weak link #1).
Every RAGTruth number is therefore a three-action result. With a real document
store there is a corpus to search, so the fourth action becomes testable.

It also lands where the self-correction literature says correction actually
works. Kamoi et al. (TACL 2024) conclude that self-correction succeeds when the
correction step has access to information the generation step did not.
Rewriting against the same fixed passages is the regime that fails;
re-retrieving from a document store is the regime that does not.
"""

from .store import Chunk, DocumentStore, chunk_text, extract_text

__all__ = ["Chunk", "DocumentStore", "chunk_text", "extract_text"]
