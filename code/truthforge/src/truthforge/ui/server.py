"""HTTP server for the TruthForge UI.

Stdlib `ThreadingHTTPServer` with a small JSON API. Design notes that matter:

* **One inference lock.** Two concurrent requests hitting the same CUDA model
  is a good way to produce garbage or an OOM on an 8 GB card. All model work
  serialises through `_INFER_LOCK`. The UI is a single-operator research tool;
  throughput is not the goal.

* **Models are cached, never reloaded per request.** Loading MiniCheck takes
  ~20 s; doing that per click would make the latency panel meaningless.

* **Localhost only.** The server loads local models and reads and writes files
  under the project directory.

* **Errors are returned, not swallowed.** A failed run sends the traceback to
  the browser, because the most common failure here (OOM, missing weights,
  wrong dtype) is one the operator has to see to fix.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from ..config import Config
from ..types import Passage

STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_INFER_LOCK = threading.Lock()

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------


@dataclass
class Registry:
    """Lazily-built, cached components shared across requests."""

    device: str = "cuda"
    _detectors: dict[str, Any] = None  # type: ignore[assignment]
    _generators: dict[str, Any] = None  # type: ignore[assignment]
    _analyzer: Any = None
    _docs: Any = None
    _errors: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._detectors = {}
        self._generators = {}
        self._errors = {}

    def detector(self, name: str, threshold: float = 0.5):
        key = f"{name}@{threshold}"
        if key in self._detectors:
            return self._detectors[key]

        cfg = Config()
        cfg.detector.name = name
        cfg.detector.device = self.device
        cfg.detector.flag_threshold = threshold

        from ..detect.minicheck import build_detector

        det = build_detector(cfg)
        # Force the load now so the first real request is not charged for it
        # and so a load failure surfaces here rather than mid-run.
        if hasattr(det, "_ensure_loaded"):
            det._ensure_loaded()
        elif hasattr(det, "backend") and hasattr(det.backend, "ensure_loaded"):
            det.backend.ensure_loaded()

        self._detectors[key] = det
        return det

    def generator(self, name: str):
        if name in self._generators:
            return self._generators[name]
        from ..config import GeneratorConfig
        from ..models import build_generator

        gen = build_generator(name, GeneratorConfig(device=self.device))
        if hasattr(gen, "_ensure_loaded"):
            gen._ensure_loaded()
        self._generators[name] = gen
        return gen

    def documents(self):
        if self._docs is None:
            from ..docs import DocumentStore

            self._docs = DocumentStore()
        return self._docs

    def contradiction_analyzer(self):
        if self._analyzer is None:
            from ..detect.nli import ContradictionAnalyzer

            self._analyzer = ContradictionAnalyzer(device=self.device)
        return self._analyzer

    def status(self) -> dict:
        from .._hf import cuda_memory_summary

        loaded_det = {}
        for key, det in self._detectors.items():
            loaded_det[key] = det.info() if hasattr(det, "info") else {"name": det.name}

        try:
            import torch  # type: ignore
            import transformers  # type: ignore

            versions = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
            torch_ok = True
        except ImportError as exc:
            versions = {"error": str(exc)}
            torch_ok = False

        return {
            "device": self.device,
            "torch_available": torch_ok,
            "versions": versions,
            "cuda": cuda_memory_summary(),
            "detectors_loaded": loaded_det,
            "generators_loaded": sorted(self._generators),
            "load_errors": dict(self._errors),
            # Deliberately short. The trivial baselines (always_supported,
            # always_unsupported, random) exist for Gate 1 and belong in
            # scripts/gate1_detector_vs_baseline.py, not in a dropdown someone
            # might pick by accident. `ensemble` is untested end to end and
            # `learned` errors until a policy is trained, so neither is
            # offered here yet. build_detector still accepts all of them.
            # NLI first: it is the only fact-checker that is usable
            # untuned. MiniCheck ranks better but its raw scores all sit above
            # 0.5, so uncalibrated it marks everything "supported" and the
            # demo looks broken. Once a scaler is fitted on RAGTruth,
            # MiniCheck becomes the sensible default.
            "available_detectors": ["nli", "minicheck", "lexical"],
            "available_routers": ["rule", "always_correct", "noop"],
            "available_generators": ["local", "echo"],
        }

    def note_error(self, key: str, exc: Exception) -> None:
        self._errors[key] = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# API handlers
# --------------------------------------------------------------------------


class Api:
    def __init__(self, registry: Registry) -> None:
        self.reg = registry

    # -- status --------------------------------------------------------

    def status(self, query: dict, body: Optional[dict]) -> dict:
        return self.reg.status()

    # -- examples ------------------------------------------------------

    def examples(self, query: dict, body: Optional[dict]) -> dict:
        source = (query.get("source", ["sample"]))[0]
        limit = int((query.get("limit", ["50"]))[0])

        from ..data.ragtruth import load_ragtruth, load_sample

        if source == "ragtruth":
            root = PROJECT_ROOT / "data" / "raw" / "ragtruth"
            if not (root / "response.jsonl").exists():
                return {
                    "source": "ragtruth",
                    "available": False,
                    "reason": "RAGTruth not downloaded. Run "
                    "`python scripts/download_ragtruth.py`.",
                    "examples": [],
                }
            examples = load_ragtruth(root=root, split="test", limit=limit)
        else:
            path = PROJECT_ROOT / "data" / "sample" / "ragtruth_tiny.jsonl"
            if not path.exists():
                return {
                    "source": "sample",
                    "available": False,
                    "reason": "Fixture missing. Run `python scripts/make_sample.py`.",
                    "examples": [],
                }
            examples = load_sample(path)

        return {
            "source": source,
            "available": True,
            "synthetic": source == "sample",
            "examples": [
                {
                    "id": ex.example_id,
                    "question": ex.question,
                    "response": ex.response,
                    "passages": [p.text for p in ex.passages],
                    "task_type": ex.task_type,
                    "model": ex.model,
                    "n_gold_spans": len(ex.spans),
                    "gold_spans": ex.spans,
                }
                for ex in examples[:limit]
            ],
        }

    # -- the pipeline --------------------------------------------------

    def run(self, query: dict, body: Optional[dict]) -> dict:
        if not body:
            raise ValueError("POST /api/run needs a JSON body")

        question = (body.get("question") or "").strip()
        answer = (body.get("answer") or "").strip() or None
        passage_texts = [p for p in (body.get("passages") or []) if p and p.strip()]
        if not passage_texts:
            raise ValueError("at least one evidence passage is required")
        if not answer and not question:
            raise ValueError("provide a question (to generate) or an answer (to check)")

        detector_name = body.get("detector", "lexical")
        router_name = body.get("router", "rule")
        generator_name = body.get("generator", "echo")

        cfg = Config()
        cfg.detector.name = detector_name
        cfg.detector.device = self.reg.device
        cfg.router.name = router_name
        cfg.loop.max_iterations = int(body.get("max_iterations", 2))
        cfg.loop.max_attempts_per_claim = int(body.get("max_attempts", 2))
        style = body.get("abstain_style")
        if style in ("message", "remove"):
            cfg.loop.abstain_style = style

        for key in (
            "accept_support_prob",
            "rewrite_support_prob",
            "thin_evidence_coverage",
            "uncertain_band",
            "contradiction_margin",
        ):
            if key in body and body[key] is not None:
                setattr(cfg.router, key, float(body[key]))

        passages = [
            Passage(doc_id=f"p{i}", text=t, source="given", rank=i)
            for i, t in enumerate(passage_texts)
        ]

        from ..claims.splitter import RuleSplitter
        from ..correct.rewrite import Rewriter
        from ..eval.outcomes import classify_attempt
        from ..loop import LoopComponents, TruthForgeLoop
        from ..retrieve.base import GivenRetriever
        from ..router.rules import build_router

        with _INFER_LOCK:
            detector = self.reg.detector(detector_name, cfg.detector.flag_threshold)
            generate_fn = self.reg.generator(generator_name)
            rewriter = Rewriter(generate_fn, abstain_message=cfg.loop.abstain_message)

            contradiction_fn = None
            if detector_name in ("minicheck", "nli", "ensemble"):
                try:
                    contradiction_fn = self.reg.contradiction_analyzer().score
                except Exception as exc:  # noqa: BLE001
                    self.reg.note_error("contradiction_analyzer", exc)

            loop = TruthForgeLoop(
                LoopComponents(
                    detector=detector,
                    router=build_router(cfg),
                    retriever=GivenRetriever(passages),
                    splitter=RuleSplitter(),
                    rewriter=rewriter,
                    generate_fn=generate_fn,
                    contradiction_fn=contradiction_fn,
                ),
                cfg,
            )
            turn = loop.run(
                question=question or "(no question given)",
                answer=answer,
                passages=passages,
                example_id=body.get("example_id"),
                dataset="ui",
            )
            for attempt in turn.attempts:
                classify_attempt(attempt, passages)

            explain = self._explain(turn, passages, detector_name)

        payload = turn.to_dict()
        payload["explain"] = explain
        payload["config"] = {
            "detector": detector_name,
            "router": router_name,
            "generator": generator_name,
            "router_thresholds": {
                "accept_support_prob": cfg.router.accept_support_prob,
                "rewrite_support_prob": cfg.router.rewrite_support_prob,
                "thin_evidence_coverage": cfg.router.thin_evidence_coverage,
                "uncertain_band": cfg.router.uncertain_band,
                "contradiction_margin": cfg.router.contradiction_margin,
            },
            "max_iterations": cfg.loop.max_iterations,
        }
        from ..eval.metrics import system_metrics

        payload["metrics"] = system_metrics([turn])
        payload["cost_by_stage"] = _cost_by_stage(turn)
        payload["detector_info"] = (
            detector.info() if hasattr(detector, "info") else {"name": detector.name}
        )
        payload["warnings"] = _run_warnings(detector, detector_name, turn)
        return payload

    def _explain(self, turn, passages, detector_name: str) -> dict:
        """Per-claim supporting evidence, for the evidence panel.

        Uses the NLI analyzer when it is already loaded or cheap to load;
        falls back to lexical coverage so the panel is never empty.
        """
        from ..detect.baselines import coverage

        out: dict[str, dict] = {}
        for claim in turn.claims:
            out[claim.claim_id] = {"coverage": round(coverage(claim.text, passages), 3)}

        if detector_name in ("nli", "ensemble", "minicheck"):
            try:
                from ..detect.nli import ContradictionAnalyzer, NLIDetector

                nli = NLIDetector(device=self.reg.device)
                details = nli.detail(turn.claims, passages)
                analyzer = ContradictionAnalyzer(device=self.reg.device)
                for claim, d in zip(turn.claims, details):
                    conflict = analyzer.analyze(claim.text, passages)
                    out[claim.claim_id].update(
                        {
                            "entailment": round(d["entailment"], 4),
                            "contradiction": round(d["contradiction"], 4),
                            "best_evidence": d.get("best_entailing_premise"),
                            "conflicting_evidence": d.get("best_contradicting_premise"),
                            "source_conflict": round(conflict["contradiction"], 4),
                            "cross_document": conflict.get("cross_document", False),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                self.reg.note_error("explain_nli", exc)
        return out

    # -- documents -------------------------------------------------------

    def docs_list(self, query: dict, body):
        return self.reg.documents().stats()

    def docs_upload(self, query: dict, body):
        """Accept one file as base64.

        Base64 in a JSON body rather than multipart: the stdlib server would
        otherwise need a multipart parser, and the browser produces this
        happily with FileReader.
        """
        import base64
        import binascii

        if not body or "filename" not in body or "content_b64" not in body:
            raise ValueError("need filename and content_b64")

        name = str(body["filename"]).strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"unacceptable filename: {name!r}")

        try:
            data = base64.b64decode(body["content_b64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"content_b64 is not valid base64: {exc}") from exc

        limit = 20 * 1024 * 1024
        if len(data) > limit:
            raise ValueError(
                f"{name} is {len(data) / 1e6:.1f} MB; the limit is {limit / 1e6:.0f} MB"
            )

        from ..docs.store import ExtractionError

        try:
            added = self.reg.documents().add(name, data)
        except ExtractionError as exc:
            raise ValueError(str(exc)) from exc

        return {"added": added, **self.reg.documents().stats()}

    def docs_remove(self, query: dict, body):
        store = self.reg.documents()
        if body and body.get("all"):
            store.clear()
        elif body and body.get("filename"):
            if not store.remove(str(body["filename"])):
                raise ValueError(f"no such file: {body['filename']}")
        else:
            raise ValueError("need filename, or all=true")
        return store.stats()

    # -- ask ---------------------------------------------------------------

    def ask(self, query: dict, body):
        """Answer from the uploaded documents, or decline to.

        The declining is the point. A question the documents do not cover must
        come back as "not enough evidence", not as a confident answer
        assembled from the model's own memory.
        """
        if not body or not (body.get("question") or "").strip():
            raise ValueError("question is required")

        store = self.reg.documents()
        if not store.chunks:
            raise ValueError("upload at least one document first")

        question = body["question"].strip()

        cfg = Config()
        cfg.detector.name = body.get("detector", "nli")
        cfg.detector.device = self.reg.device
        cfg.router.name = body.get("router", "rule")
        cfg.loop.max_iterations = int(body.get("max_iterations", 1))
        cfg.retrieval.top_k = int(body.get("top_k", 5))
        # Deleting an unsupported sentence is right here: the user asked a
        # question, and a sentence the documents do not support should not
        # appear in the answer at all.
        cfg.loop.abstain_style = "remove"

        # Turn on the question-level gate. This IS the "say so when the answer
        # is not in the documents" behaviour, and claim-level checking alone
        # does not provide it: if retrieval returns chunks about something
        # else, the generator writes about that something else and every
        # sentence it writes is genuinely supported by the chunks. The
        # open-corpus evaluation measured 44% of questions with no answer in
        # the corpus receiving a confident answer. Here the system did the
        # retrieving, so the gate is meaningful and is switched on.
        cfg.router.answerability_gate = True
        if body.get("min_question_coverage") is not None:
            cfg.router.min_question_coverage = float(body["min_question_coverage"])

        from ..claims.splitter import RuleSplitter
        from ..correct.rewrite import Rewriter
        from ..eval.metrics import system_metrics
        from ..eval.outcomes import classify_attempt
        from ..loop import LoopComponents, TruthForgeLoop
        from ..router.rules import build_router

        with _INFER_LOCK:
            retriever = store.retriever(
                use_dense=bool(body.get("dense", False)), device=self.reg.device
            )
            detector = self.reg.detector(cfg.detector.name, cfg.detector.flag_threshold)
            generate_fn = self.reg.generator(body.get("generator", "local"))

            contradiction_fn = None
            if cfg.detector.name in ("minicheck", "nli", "ensemble"):
                try:
                    contradiction_fn = self.reg.contradiction_analyzer().score
                except Exception as exc:  # noqa: BLE001
                    self.reg.note_error("contradiction_analyzer", exc)

            loop = TruthForgeLoop(
                LoopComponents(
                    detector=detector,
                    router=build_router(cfg),
                    retriever=retriever,
                    splitter=RuleSplitter(),
                    rewriter=Rewriter(
                        generate_fn, abstain_message=cfg.loop.abstain_message
                    ),
                    generate_fn=generate_fn,
                    contradiction_fn=contradiction_fn,
                ),
                cfg,
            )
            # answer=None -> the loop generates from the retrieved chunks.
            turn = loop.run(question=question, dataset="documents")
            for a in turn.attempts:
                classify_attempt(a, turn.passages)

        if turn.meta.get("refused_before_generating"):
            # Nothing was generated, so there is nothing to explain per-claim.
            # Say which check refused and how far off the question was, rather
            # than returning a bare "no".
            qc = turn.meta.get("question_coverage")
            return {
                "question": question,
                "answer": None,
                "insufficient_evidence": True,
                # Two different refusals reach the same panel and the reader
                # should be able to tell them apart: this one never wrote an
                # answer, the other one wrote and then deleted every sentence.
                "refused_by": "question_not_covered_by_documents",
                "message": (
                    "The uploaded documents do not appear to cover this "
                    f"question ({qc:.0%} of its key terms appear in the best "
                    f"matching text, below the {cfg.router.min_question_coverage:.0%} "
                    "needed), so no answer was drafted."
                ),
                "question_coverage": qc,
                "min_question_coverage": cfg.router.min_question_coverage,
                "draft": None,
                "n_claims": 0,
                "n_dropped": 0,
                "sources": [
                    {
                        "doc_id": p.doc_id,
                        "citation": (store.by_id(p.doc_id).citation
                                     if store.by_id(p.doc_id) else p.doc_id),
                        "text": p.text[:400],
                        "score": round(p.score, 3),
                    }
                    for p in turn.passages[:3]
                ],
                "support": {},
                "claims": [],
                "verdicts": [],
                "decisions": [],
                "cost_by_stage": _cost_by_stage(turn),
                "metrics": system_metrics([turn]),
                "warnings": [],
                "seconds": round(turn.wall_clock_s, 2),
            }

        answer = turn.final_answer.strip()
        abstained = set(turn.meta.get("abstained_claim_ids", []))
        kept = [c for c in turn.claims if c.claim_id not in abstained]
        insufficient = not answer or not kept

        # Which source sentence backs each surviving sentence of the answer.
        # Without this the user is asked to take "verified" on trust; with it
        # every sentence is auditable against a named chunk.
        support = self._per_claim_evidence(turn, store, cfg.detector.name)

        cites = []
        for p in turn.passages:
            ch = store.by_id(p.doc_id)
            if ch:
                cites.append({
                    "citation": ch.citation,
                    "chunk_id": ch.chunk_id,
                    "text": ch.text,
                    "score": round(p.score, 3),
                })

        d = turn.to_dict()
        return {
            "question": question,
            "answer": None if insufficient else answer,
            "insufficient_evidence": insufficient,
            "message": (
                "Not enough evidence in the uploaded documents to answer this."
                if insufficient
                else None
            ),
            "draft": turn.initial_answer,
            "n_claims": len(turn.claims),
            "n_dropped": len(abstained),
            "sources": cites,
            "support": support,
            "claims": d["claims"],
            "verdicts": d["verdicts"],
            "decisions": d["decisions"],
            "cost_by_stage": _cost_by_stage(turn),
            "metrics": system_metrics([turn]),
            "warnings": _run_warnings(detector, cfg.detector.name, turn),
            "seconds": round(turn.wall_clock_s, 2),
        }

    def _per_claim_evidence(self, turn, store, detector_name: str) -> dict:
        """claim_id -> the sentence that supports it, and where it came from.

        Uses the NLI detector's own best-entailing premise, which it computes
        anyway while scoring, so this costs one extra pass rather than new
        machinery. Falls back to lexical overlap when NLI is unavailable, so
        the panel is never empty.
        """
        from ..detect.baselines import coverage

        out: dict = {}
        kept = [c for c in turn.claims
                if c.claim_id not in set(turn.meta.get("abstained_claim_ids", []))]
        if not kept:
            return out

        details = None
        if detector_name in ("nli", "ensemble", "minicheck"):
            try:
                from ..detect.nli import NLIDetector

                details = NLIDetector(device=self.reg.device).detail(kept, turn.passages)
            except Exception as exc:  # noqa: BLE001
                self.reg.note_error("per_claim_evidence", exc)

        for i, claim in enumerate(kept):
            entry = {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "coverage": round(coverage(claim.text, turn.passages), 3),
            }
            if details:
                d = details[i]
                sentence = d.get("best_entailing_premise")
                entry.update({
                    "entailment": round(d["entailment"], 4),
                    "supporting_sentence": sentence,
                    "citation": _citation_for(sentence, turn.passages, store),
                })
            out[claim.claim_id] = entry
        return out

    # -- runs ----------------------------------------------------------

    def runs(self, query: dict, body: Optional[dict]) -> dict:
        runs_dir = PROJECT_ROOT / "runs"
        out = []
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir()):
                turns_file = d / "turns.jsonl"
                if not (d.is_dir() and turns_file.exists()):
                    continue
                metrics = {}
                mpath = d / "metrics.json"
                if mpath.exists():
                    try:
                        metrics = json.loads(mpath.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        metrics = {"error": "metrics.json is not valid JSON"}
                cfg = {}
                cpath = d / "config.json"
                if cpath.exists():
                    try:
                        cfg = json.loads(cpath.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                out.append(
                    {
                        "name": d.name,
                        "n_turns": metrics.get("n_turns", 0),
                        "metrics": metrics,
                        "detector": cfg.get("detector", {}).get("name"),
                        "router": cfg.get("router", {}).get("name"),
                        "dataset": cfg.get("dataset"),
                        "mtime": turns_file.stat().st_mtime,
                    }
                )
        return {"runs": out, "runs_dir": str(runs_dir)}

    def compare(self, query: dict, body: Optional[dict]) -> dict:
        names = [n for n in (query.get("runs", [""])[0]).split(",") if n]
        if not names:
            return {"rows": [], "reason": "no runs selected"}

        from ..cli import _turn_from_dict
        from ..eval.metrics import results_row, system_metrics

        from ..eval.comparison import compare_arms, pareto_front, score_arm

        rows, detail, scores = [], {}, {}
        for name in names:
            path = PROJECT_ROOT / "runs" / name / "turns.jsonl"
            if not path.exists():
                continue
            turns = [
                _turn_from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows.append(results_row(name, turns))
            detail[name] = system_metrics(turns)
            # Score against RAGTruth's annotated spans when the run is on
            # RAGTruth. Without this the UI shows the detector grading the
            # work its own flags caused, which is the circular version the
            # CLI already refuses to present without a warning.
            scores[name] = score_arm(name, turns, _gold_for(turns))

        # The head-to-head is what actually licenses a claim, so compute it
        # here rather than leaving it to a terminal the operator may not open.
        ledger = {k: v.to_dict() for k, v in scores.items()}
        verdict = None
        # Match by suffix: runs are named rule / rt_rule / rt_rule_dense and
        # a literal "rule" lookup silently produced no verdict at all on the
        # RAGTruth runs.
        def _pick(suffix):
            exact = [k for k in scores if k == suffix]
            return (exact or [k for k in scores if k.endswith("_" + suffix)] or [None])[0]

        ours = _pick("rule")
        base = _pick("always") or _pick("always_correct")
        if ours and base:
            try:
                # Fewer resamples than the CLI: this runs on every click.
                verdict = compare_arms(scores[ours], scores[base], n_resamples=2000)
            except ValueError as exc:
                verdict = {"error": str(exc)}

        return {
            "rows": rows,
            "detail": detail,
            "ledger": ledger,
            "pareto_front": pareto_front(list(scores.values())) if scores else [],
            "verdict": verdict,
            "gold_source": next(
                (v.gold_source for v in scores.values()), "detector_on_original"
            ),
        }

    # -- annotation ----------------------------------------------------

    def annotation_batch(self, query: dict, body: Optional[dict]) -> dict:
        run = (query.get("run", [""]))[0]
        if not run:
            return {"items": [], "reason": "no run selected"}

        path = PROJECT_ROOT / "runs" / run / "turns.jsonl"
        if not path.exists():
            return {"items": [], "reason": f"no turns.jsonl in runs/{run}"}

        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            turn = json.loads(line)
            evidence = "\n\n".join(p["text"] for p in turn.get("passages", []))
            for a in turn.get("attempts", []):
                items.append(
                    {
                        "attempt_key": f"{a['claim_id']}:{a['iteration']}:{a['action']}",
                        "claim_id": a["claim_id"],
                        "action": a["action"],
                        "iteration": a["iteration"],
                        "before_text": a["before_text"],
                        "after_text": a["after_text"],
                        "evidence": evidence,
                        # The heuristic label is deliberately NOT sent. The
                        # annotator's judgement is what validates it, so it
                        # must not be anchored to it.
                    }
                )

        existing = self._load_labels(run)
        return {
            "run": run,
            "items": items,
            "labelled": existing,
            "n_total": len(items),
            "n_labelled": len(existing),
        }

    def annotate(self, query: dict, body: Optional[dict]) -> dict:
        if not body:
            raise ValueError("POST /api/annotate needs a JSON body")

        run = body.get("run")
        annotator = (body.get("annotator_id") or "").strip()
        if not run or not annotator:
            raise ValueError("run and annotator_id are required")

        outcome = (body.get("outcome") or "").strip().lower()
        worse_kind = (body.get("worse_kind") or "").strip().lower()

        from ..types import Outcome, WorseKind

        valid = {o.value for o in Outcome}
        if outcome not in valid:
            raise ValueError(f"outcome must be one of {sorted(valid)}")
        if outcome == "worse" and worse_kind not in set(WorseKind.values()):
            raise ValueError(f"a WORSE label needs a worse_kind in {WorseKind.values()}")
        if outcome != "worse" and worse_kind:
            raise ValueError("worse_kind must be empty unless the outcome is WORSE")

        path = self._labels_path(run, annotator)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = {}
        if path.exists():
            import csv

            with path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    rows[row["attempt_key"]] = row

        rows[body["attempt_key"]] = {
            "attempt_key": body["attempt_key"],
            "claim_id": body.get("claim_id", ""),
            "action": body.get("action", ""),
            "iteration": body.get("iteration", ""),
            "before_text": body.get("before_text", ""),
            "after_text": body.get("after_text", ""),
            "evidence": (body.get("evidence") or "")[:3000],
            "outcome": outcome,
            "worse_kind": worse_kind,
            "annotator_id": annotator,
            "notes": body.get("notes", ""),
        }

        import csv

        from ..eval.outcomes import ANNOTATION_COLUMNS

        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ANNOTATION_COLUMNS)
            w.writeheader()
            for r in rows.values():
                w.writerow({k: r.get(k, "") for k in ANNOTATION_COLUMNS})

        return {"saved": True, "path": str(path), "n_labelled": len(rows)}

    def agreement(self, query: dict, body: Optional[dict]) -> dict:
        run = (query.get("run", [""]))[0]
        a_id = (query.get("a", ["A"]))[0]
        b_id = (query.get("b", ["B"]))[0]

        from ..eval.outcomes import agreement_report, load_human_labels

        pa, pb = self._labels_path(run, a_id), self._labels_path(run, b_id)
        if not pa.exists() or not pb.exists():
            return {
                "available": False,
                "reason": f"need both {pa.name} and {pb.name} in annotation/",
            }

        heuristic = {}
        turns_path = PROJECT_ROOT / "runs" / run / "turns.jsonl"
        if turns_path.exists():
            for line in turns_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                for at in json.loads(line).get("attempts", []):
                    if at.get("outcome"):
                        key = f"{at['claim_id']}:{at['iteration']}:{at['action']}"
                        heuristic[key] = at["outcome"]

        report = agreement_report(
            load_human_labels(pa), load_human_labels(pb), heuristic or None
        )
        return {"available": True, "report": report}

    def _labels_path(self, run: str, annotator: str) -> Path:
        safe = "".join(c for c in annotator if c.isalnum() or c in "-_")[:32] or "X"
        safe_run = "".join(c for c in run if c.isalnum() or c in "-_.")[:64]
        return PROJECT_ROOT / "annotation" / f"labels_{safe_run}_{safe}.csv"

    def _load_labels(self, run: str) -> dict:
        import csv

        out: dict[str, dict] = {}
        adir = PROJECT_ROOT / "annotation"
        if not adir.exists():
            return out
        for path in adir.glob(f"labels_{run}_*.csv"):
            annotator = path.stem.rsplit("_", 1)[-1]
            with path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("outcome"):
                        out.setdefault(row["attempt_key"], {})[annotator] = {
                            "outcome": row["outcome"],
                            "worse_kind": row.get("worse_kind", ""),
                        }
        return out


#: Detectors whose raw scores are known to sit almost entirely above the
#: decision boundary. Running these uncalibrated makes the router unreachable.
_NEEDS_CALIBRATION = {"minicheck", "nli", "ensemble"}


def _run_warnings(detector, detector_name: str, turn) -> list[dict]:
    """Conditions that make a run's numbers untrustworthy.

    These are surfaced in the UI rather than logged, because the failures they
    describe are invisible in the output: an uncalibrated judge still produces
    a confident-looking faithfulness score, it is just one the router never
    got to act on.
    """
    out: list[dict] = []

    calibrated = "calibrated" in getattr(detector, "name", "")
    if detector_name in _NEEDS_CALIBRATION and not calibrated:
        # Two different defects, two different consequences. Saying
        # "unsupported claims read as supported" about a detector that just
        # got them right would be simply false, and the operator would stop
        # trusting these banners.
        probs = [v.support_prob for v in turn.verdicts]
        one_sided = bool(probs) and min(probs) > 0.45

        if one_sided:
            out.append({
                "level": "warn",
                "title": "Fact-checker is not calibrated — it scored nothing below 0.45",
                "body": (
                    f"Every claim came back above the decision boundary, so "
                    f"{detector_name} flagged nothing and the router had no "
                    "work to do. Clearly false claims can score 0.504 here. "
                    f"Fit a scaler: python scripts/calibrate_detector.py "
                    f"--detector {detector_name} (needs RAGTruth; a scaler "
                    "fitted on the demo fixture is deliberately refused)."
                ),
            })
        else:
            out.append({
                "level": "info",
                "title": "Fact-checker is uncalibrated, but it separated this example",
                "body": (
                    f"{detector_name} scores here are near 0 or near 1 with "
                    "little in between, so the router's “skip when unsure” "
                    "dial has almost nothing to act on. Labels on this example "
                    "look right; the threshold is still unvalidated. Calibrate "
                    "before reporting any number."
                ),
            })

    if not turn.meta.get("contradiction_signal_is_real", False):
        out.append({
            "level": "info",
            "title": "Source-conflict signal is the lexical placeholder",
            "body": (
                "The 'contradictory_sources' routing reason is not evidence of "
                "anything in this run. Pick a neural detector to load the NLI "
                "analyzer."
            ),
        })

    rejected = [a for a in turn.attempts if a.rejection]
    if rejected:
        out.append({
            "level": "warn",
            "title": f"{len(rejected)} of {len(turn.attempts)} corrector outputs discarded",
            "body": (
                "The corrector did not return a single corrected sentence "
                f"({', '.join(sorted({str(a.rejection).split('(')[0] for a in rejected}))}). "
                "Those attempts cost tokens and changed nothing, which inflates "
                "the futile rate for reasons that belong to the corrector."
            ),
        })

    return out


def _citation_for(sentence, passages, store) -> str | None:
    """Which uploaded chunk a supporting sentence came from."""
    if not sentence:
        return None
    for p in passages:
        if sentence.strip() and sentence.strip() in p.text:
            chunk = store.by_id(p.doc_id)
            if chunk:
                return chunk.citation
            return p.meta.get("citation")
    return None


def _gold_for(turns) -> Optional[dict]:
    """claim_id -> was_supported from RAGTruth spans, or None if unavailable.

    Returns None rather than a partial map: a map that covers some claims and
    not others makes the ledger mix two gold sources under one label, and the
    difference between arms here is a handful of claims.
    """
    rt = [t for t in turns if t.dataset == "ragtruth" and t.example_id]
    if not rt:
        return None
    try:
        from ..data.ragtruth import load_ragtruth

        exs = {e.example_id: e for e in load_ragtruth(root=str(PROJECT_ROOT / "data/raw/ragtruth"))}
    except Exception:  # noqa: BLE001
        return None

    gold = {}
    for t in rt:
        ex = exs.get(t.example_id)
        if ex is None:
            continue
        for c in t.claims:
            if c.start is not None and c.end is not None:
                gold[c.claim_id] = not ex.overlaps_span(c.start, c.end)

    want = {v.claim_id for t in turns for v in t.verdicts}
    return gold if want and len(want & set(gold)) / len(want) >= 0.9 else None


def _cost_by_stage(turn) -> list[dict]:
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(
        lambda: {"tokens": 0, "latency_s": 0.0, "calls": 0}
    )
    for c in turn.costs:
        a = agg[c.stage]
        a["tokens"] += c.total_tokens
        a["latency_s"] += c.latency_s
        a["calls"] += c.llm_calls + c.detector_calls + c.retrieval_calls
    return [
        {"stage": k, "tokens": v["tokens"], "latency_s": round(v["latency_s"], 4),
         "calls": v["calls"]}
        for k, v in agg.items()
    ]


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------


def _make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    routes: dict[tuple[str, str], Callable] = {
        ("GET", "/api/status"): api.status,
        ("GET", "/api/examples"): api.examples,
        ("GET", "/api/runs"): api.runs,
        ("GET", "/api/compare"): api.compare,
        ("GET", "/api/annotate/batch"): api.annotation_batch,
        ("GET", "/api/annotate/agreement"): api.agreement,
        ("POST", "/api/run"): api.run,
        ("GET", "/api/docs"): api.docs_list,
        ("POST", "/api/docs/upload"): api.docs_upload,
        ("POST", "/api/docs/remove"): api.docs_remove,
        ("POST", "/api/ask"): api.ask,
        ("POST", "/api/annotate"): api.annotate,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "TruthForge"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # The default logs every static asset; only note API calls.
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

        # -- responses ---------------------------------------------------

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, code: int = 200) -> None:
            self._send(
                code,
                json.dumps(payload, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        # -- dispatch -----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._dispatch("GET", parsed.path, parse_qs(parsed.query), None)
            else:
                self._static(parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                self._json({"error": f"invalid JSON body: {exc}"}, 400)
                return
            self._dispatch("POST", parsed.path, parse_qs(parsed.query), body)

        def _dispatch(self, method: str, path: str, query: dict, body: Optional[dict]) -> None:
            handler = routes.get((method, path.rstrip("/") or "/"))
            if handler is None:
                self._json({"error": f"no route for {method} {path}"}, 404)
                return
            try:
                self._json(handler(query, body))
            except ValueError as exc:
                self._json({"error": str(exc), "kind": "bad_request"}, 400)
            except Exception as exc:  # noqa: BLE001
                # The operator needs the traceback: OOM, missing weights and
                # dtype mismatches are all diagnosable only from it.
                self._json(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "kind": "server_error",
                    },
                    500,
                )

        def _static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            try:
                target.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._json({"error": "forbidden"}, 403)
                return
            if not target.is_file():
                self._json({"error": f"not found: {rel}"}, 404)
                return
            self._send(
                200, target.read_bytes(), _MIME.get(target.suffix, "application/octet-stream")
            )

    return Handler


class _Server(ThreadingHTTPServer):
    """Quiet about clients that hang up mid-response.

    A browser navigating away, or a probe with a short timeout, aborts the
    socket while we are still writing. The stdlib prints a full traceback for
    that, which reads like a crash in the operator's terminal when it is just
    the other end going away. Real errors still print.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):  # noqa: D102
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def create_server(host: str = "127.0.0.1", port: int = 8420, device: str = "cuda"):
    registry = Registry(device=device)
    handler = _make_handler(Api(registry))
    httpd = _Server((host, port), handler)
    httpd.registry = registry  # type: ignore[attr-defined]
    return httpd


def serve(
    host: str = "127.0.0.1",
    port: int = 8420,
    open_browser: bool = False,
    preload: Optional[list[str]] = None,
    device: str = "cuda",
) -> int:
    httpd = create_server(host, port, device)
    url = f"http://{host}:{port}/"

    if preload:
        print(f"preloading: {', '.join(preload)}")
        reg = httpd.registry  # type: ignore[attr-defined]
        for name in preload:
            t0 = time.perf_counter()
            try:
                # "local"/"echo" name a generator; everything else a detector.
                if name in ("local", "echo"):
                    reg.generator(name)
                else:
                    reg.detector(name)
                print(f"  {name}: ready in {time.perf_counter() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001
                reg.note_error(name, exc)
                print(f"  {name}: FAILED -- {type(exc).__name__}: {exc}")

        from .._hf import cuda_memory_summary

        mem = cuda_memory_summary()
        if mem.get("available"):
            print(f"  VRAM: {mem['used_mb']} MB used of {mem['total_mb']} MB")

    print(f"\nTruthForge UI on {url}")
    print("  Ctrl+C to stop\n")

    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0
