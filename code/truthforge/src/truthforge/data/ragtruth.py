"""RAGTruth loader.

Expects the layout of the official release (ParticleMedia/RAGTruth):

    data/raw/ragtruth/source_info.jsonl
    data/raw/ragtruth/response.jsonl

`source_info.jsonl` rows carry the task and the grounding context; the
`source_info` field is a dict for QA / Data2txt and a string for Summary, so it
is normalised to a passage list here.

`response.jsonl` rows carry the model output and word-level `labels`, each a
span `{start, end, text, label_type}` over the response string.

IMPORTANT (from the brief): one re-annotation study reports that the original
spans miss a substantial number of hallucinations. Treat span recall as a lower
bound and check `RAGTRUTH_SPAN_CAVEAT` before quoting detector recall numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from ..types import Passage

RAGTRUTH_SPAN_CAVEAT = (
    "Original RAGTruth spans under-annotate; a re-annotation pass found "
    "additional hallucinations the release marks as clean. Report detector "
    "recall against these spans as a LOWER BOUND, and never claim a "
    "false-positive is a false-positive without reading the passage."
)

# The four span types in the release, grouped the way the router cares about:
# a conflict has contradicting evidence in context (rewritable); baseless info
# has no evidence at all (re-retrieve or abstain).
CONFLICT_TYPES = {"Evident Conflict", "Subtle Conflict"}
BASELESS_TYPES = {"Evident Baseless Info", "Subtle Baseless Info"}


@dataclass
class RAGTruthExample:
    """One (context, response) pair with its gold hallucination spans."""

    example_id: str
    source_id: str
    task_type: str            # "QA" | "Data2txt" | "Summary"
    model: str                # which LLM produced the response
    split: str
    question: str
    response: str
    passages: list[Passage]
    spans: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hallucination(self) -> bool:
        return len(self.spans) > 0

    @property
    def hallucination_char_fraction(self) -> float:
        """RAGTruth's 'intensity' proxy: share of the response inside a span."""
        if not self.response:
            return 0.0
        covered = sum(max(0, s["end"] - s["start"]) for s in self.spans)
        return min(1.0, covered / len(self.response))

    def span_types(self) -> set[str]:
        return {s.get("label_type", "") for s in self.spans}

    def overlaps_span(self, start: int, end: int) -> bool:
        """Does a [start, end) character range touch any gold span?

        This is the bridge between our claim spans and RAGTruth's word spans,
        and it is deliberately permissive (any overlap counts). A stricter
        criterion belongs in the eval module, not the loader.
        """
        return any(start < s["end"] and end > s["start"] for s in self.spans)


def _normalise_passages(source_info: Any, source_id: str) -> list[Passage]:
    """Turn RAGTruth's polymorphic `source_info` into passages."""
    if isinstance(source_info, str):
        return [Passage(doc_id=f"{source_id}:0", text=source_info, source="given")]

    if isinstance(source_info, dict):
        # QA rows: {"question": ..., "passages": "..."} (passages may be one
        # concatenated string or a list). Data2txt rows are structured records.
        raw = source_info.get("passages", source_info.get("passage"))
        if isinstance(raw, list):
            return [
                Passage(doc_id=f"{source_id}:{i}", text=str(p), source="given", rank=i)
                for i, p in enumerate(raw)
            ]
        if isinstance(raw, str):
            # The release concatenates numbered passages with "passage N:".
            chunks = _split_numbered_passages(raw)
            return [
                Passage(doc_id=f"{source_id}:{i}", text=c, source="given", rank=i)
                for i, c in enumerate(chunks)
            ]
        # Data2txt: no free text, serialise the record itself.
        return [
            Passage(
                doc_id=f"{source_id}:0",
                text=json.dumps(source_info, ensure_ascii=False),
                source="given",
            )
        ]

    return [Passage(doc_id=f"{source_id}:0", text=str(source_info), source="given")]


def _split_numbered_passages(text: str) -> list[str]:
    """Split on 'passage N:' markers, falling back to the whole string."""
    import re

    parts = re.split(r"(?im)^\s*passage\s*\d+\s*:\s*", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks or [text.strip()]


def _extract_question(source_info: Any, prompt: str) -> str:
    if isinstance(source_info, dict):
        for key in ("question", "query", "instruction"):
            if isinstance(source_info.get(key), str):
                return source_info[key]
    return prompt


def load_ragtruth(
    root: str | Path = "data/raw/ragtruth",
    split: Optional[str] = None,
    task_types: Optional[set[str]] = None,
    models: Optional[set[str]] = None,
    limit: Optional[int] = None,
) -> list[RAGTruthExample]:
    """Load and join `source_info.jsonl` with `response.jsonl`.

    Args:
        root: directory holding the two jsonl files.
        split: "train" | "test" | None for both.
        task_types: e.g. {"QA"} to restrict to the question-answering subset.
        models: restrict to responses from specific generator models.
        limit: stop after N examples (post-filter).
    """
    root = Path(root)
    src_path, resp_path = root / "source_info.jsonl", root / "response.jsonl"
    for p in (src_path, resp_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python scripts/download_ragtruth.py` first."
            )

    sources = {row["source_id"]: row for row in _read_jsonl(src_path)}

    out: list[RAGTruthExample] = []
    for row in _read_jsonl(resp_path):
        if split and row.get("split") != split:
            continue
        if models and row.get("model") not in models:
            continue

        src = sources.get(row["source_id"])
        if src is None:
            continue
        if task_types and src.get("task_type") not in task_types:
            continue

        info = src.get("source_info")
        spans = [
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "text": s.get("text", ""),
                "label_type": s.get("label_type", ""),
                "implicit_true": s.get("implicit_true", False),
                "due_to_null": s.get("due_to_null", False),
            }
            for s in row.get("labels", []) or []
        ]

        out.append(
            RAGTruthExample(
                example_id=str(row.get("id", f"{row['source_id']}:{row.get('model')}")),
                source_id=str(row["source_id"]),
                task_type=str(src.get("task_type", "unknown")),
                model=str(row.get("model", "unknown")),
                split=str(row.get("split", "unknown")),
                question=_extract_question(info, str(src.get("prompt", ""))),
                response=str(row.get("response", "")),
                passages=_normalise_passages(info, str(row["source_id"])),
                spans=spans,
                meta={
                    "temperature": row.get("temperature"),
                    "quality": row.get("quality"),
                    "prompt": src.get("prompt"),
                },
            )
        )
        if limit is not None and len(out) >= limit:
            break

    return out


def load_sample(path: str | Path = "data/sample/ragtruth_tiny.jsonl") -> list[RAGTruthExample]:
    """Load the checked-in synthetic fixture.

    Hand-written, RAGTruth-shaped, NOT real data. It exists so the pipeline and
    the tests run before anyone downloads anything. Never report numbers from it.
    """
    rows = list(_read_jsonl(Path(path)))
    return [
        RAGTruthExample(
            example_id=r["id"],
            source_id=r["source_id"],
            task_type=r.get("task_type", "QA"),
            model=r.get("model", "synthetic"),
            split="sample",
            question=r["question"],
            response=r["response"],
            passages=[
                Passage(doc_id=f"{r['source_id']}:{i}", text=p, source="given", rank=i)
                for i, p in enumerate(r["passages"])
            ],
            spans=r.get("labels", []),
            meta={"synthetic": True},
        )
        for r in rows
    ]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
