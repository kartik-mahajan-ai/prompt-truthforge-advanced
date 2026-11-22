"""Text extraction, chunking, and an in-memory document store.

Design notes that matter for the research, not just the plumbing:

* **Chunks carry provenance.** Every chunk knows its file, its page (for PDFs)
  and its character offsets. Without that, an answer cannot cite anything and
  "the evidence supports this" is unverifiable by the person reading it.

* **Overlap is deliberate.** A fact split across a chunk boundary is invisible
  to retrieval, which looks exactly like a hallucination downstream: the claim
  is true, the evidence exists, and the system says it cannot find it. The
  overlap costs storage and prevents a whole class of false abstentions.

* **No new dependencies.** PDFs go through pypdfium2 and .docx through
  python-docx, both of which the project already has. Plain text needs
  neither, so the store works with zero extras installed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..types import Passage

SUPPORTED = {".txt", ".md", ".markdown", ".rst", ".csv", ".log", ".json", ".pdf", ".docx"}


class ExtractionError(RuntimeError):
    """Raised when a file cannot be turned into text."""


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_text(path: str | Path, data: Optional[bytes] = None) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a file. Page is 1 for non-paged formats.

    `data` lets the caller pass bytes it already has (an upload) without
    writing a temporary file.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(p, data)
    if suffix == ".docx":
        return _extract_docx(p, data)
    if suffix in SUPPORTED:
        raw = data if data is not None else p.read_bytes()
        return [(1, raw.decode("utf-8", errors="replace"))]

    raise ExtractionError(
        f"unsupported file type {suffix!r}. Supported: {', '.join(sorted(SUPPORTED))}"
    )


def _extract_pdf(p: Path, data: Optional[bytes]) -> list[tuple[int, str]]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pip install pypdfium2 to read PDFs") from exc

    pdf = pdfium.PdfDocument(data if data is not None else str(p))
    out = []
    for i in range(len(pdf)):
        text = pdf[i].get_textpage().get_text_range()
        if text and text.strip():
            out.append((i + 1, text))
    if not out:
        raise ExtractionError(
            f"{p.name}: no selectable text. Scanned PDFs need OCR, which this "
            "project does not do -- the pages are images, not text."
        )
    return out


def _extract_docx(p: Path, data: Optional[bytes]) -> list[tuple[int, str]]:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pip install python-docx to read .docx files") from exc

    import io

    d = docx.Document(io.BytesIO(data) if data is not None else str(p))
    parts = [para.text for para in d.paragraphs if para.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    if not parts:
        raise ExtractionError(f"{p.name}: no text found")
    return [(1, "\n".join(parts))]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    source: str          # file name
    page: int
    text: str
    start: int           # char offset within the page's text
    end: int

    @property
    def citation(self) -> str:
        return f"{self.source} p.{self.page}" if self.page > 1 else self.source

    def to_passage(self, rank: Optional[int] = None) -> Passage:
        return Passage(
            doc_id=self.chunk_id,
            text=self.text,
            source="document",
            rank=rank,
            meta={"citation": self.citation, "file": self.source, "page": self.page},
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "source": self.source, "page": self.page,
            "text": self.text, "citation": self.citation,
        }


_PARA = re.compile(r"\n\s*\n")


def chunk_text(
    pages: Iterable[tuple[int, str]],
    source: str,
    target_chars: int = 900,
    overlap_chars: int = 180,
) -> list[Chunk]:
    """Split pages into overlapping chunks that respect paragraph boundaries.

    Paragraph-aware rather than fixed-width: cutting mid-sentence produces
    chunks that retrieve badly and read worse when shown as a citation.
    """
    chunks: list[Chunk] = []

    for page, text in pages:
        if not text.strip():
            continue
        paras, cursor = [], 0
        for part in _PARA.split(text):
            idx = text.find(part, cursor)
            if idx < 0:
                idx = cursor
            paras.append((idx, part))
            cursor = idx + len(part)

        buf, buf_start = "", None
        for start, para in paras:
            para = para.strip()
            if not para:
                continue
            if buf_start is None:
                buf_start = start
            candidate = f"{buf}\n\n{para}" if buf else para

            if len(candidate) <= target_chars:
                buf = candidate
                continue

            if buf:
                chunks.append(_make(source, page, buf, buf_start))
                # Carry the tail forward so a fact spanning the boundary is
                # still retrievable from the next chunk.
                tail = buf[-overlap_chars:] if overlap_chars else ""
                buf = f"{tail}\n\n{para}" if tail else para
                buf_start = start
            else:
                # One paragraph longer than a whole chunk: hard-split it.
                for i in range(0, len(para), target_chars):
                    chunks.append(_make(source, page, para[i : i + target_chars], start + i))
                buf, buf_start = "", None

        if buf.strip():
            chunks.append(_make(source, page, buf, buf_start or 0))

    return chunks


def _make(source: str, page: int, text: str, start: int) -> Chunk:
    text = text.strip()
    digest = hashlib.sha1(f"{source}:{page}:{start}:{text[:80]}".encode()).hexdigest()[:12]
    return Chunk(
        chunk_id=f"{digest}", source=source, page=page, text=text,
        start=start, end=start + len(text),
    )


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


@dataclass
class DocumentStore:
    """Chunks from uploaded files, plus a retriever over them."""

    target_chars: int = 900
    overlap_chars: int = 180
    chunks: list[Chunk] = field(default_factory=list)
    files: dict[str, int] = field(default_factory=dict)   # name -> chunk count
    _retriever: object = None
    # Which mode the cached index was built in. Without this the cache
    # answers a dense request with the BM25 index it happens to hold, and
    # the UI toggle silently does nothing after the first query.
    _retriever_dense: bool = False

    def add(self, filename: str, data: bytes) -> dict:
        pages = extract_text(filename, data)
        new = chunk_text(pages, filename, self.target_chars, self.overlap_chars)
        if not new:
            raise ExtractionError(f"{filename}: produced no chunks")

        # Replacing a re-uploaded file rather than duplicating it: duplicate
        # chunks would let one document outvote the others in retrieval.
        if filename in self.files:
            self.chunks = [c for c in self.chunks if c.source != filename]

        self.chunks.extend(new)
        self.files[filename] = len(new)
        self._retriever = None     # index is stale
        return {
            "file": filename,
            "pages": len({p for p, _ in pages}),
            "chunks": len(new),
            "characters": sum(len(c.text) for c in new),
        }

    def remove(self, filename: str) -> bool:
        if filename not in self.files:
            return False
        self.chunks = [c for c in self.chunks if c.source != filename]
        del self.files[filename]
        self._retriever = None
        return True

    def clear(self) -> None:
        self.chunks.clear()
        self.files.clear()
        self._retriever = None

    def retriever(self, use_dense: bool = False, device: str = "cuda"):
        """Build (and cache) a retriever over the current chunks.

        The cache is keyed on the mode as well as the contents. Returning a
        cached BM25 index to a caller that asked for dense would make the
        toggle a no-op in every session after the first query -- and an A/B
        that silently never switches is worse than no A/B.
        """
        if self._retriever is not None and self._retriever_dense == use_dense:
            return self._retriever
        if not self.chunks:
            raise ValueError("no documents uploaded")

        from ..retrieve.base import BM25Index, HybridRetriever

        texts = [c.text for c in self.chunks]
        ids = [c.chunk_id for c in self.chunks]
        self._retriever = (
            HybridRetriever(texts, ids, device=device, use_dense=True)
            if use_dense
            else BM25Index(texts, ids)
        )
        self._retriever_dense = use_dense
        return self._retriever

    def by_id(self, chunk_id: str) -> Optional[Chunk]:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def stats(self) -> dict:
        return {
            "files": [
                {"name": n, "chunks": c} for n, c in sorted(self.files.items())
            ],
            "n_files": len(self.files),
            "n_chunks": len(self.chunks),
            "n_characters": sum(len(c.text) for c in self.chunks),
        }
