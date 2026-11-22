"""MiniCheck detector -- the primary judge.

MiniCheck (Tang et al., EMNLP 2024) reaches GPT-4-level fact-checking accuracy
at a fraction of the cost; the 770M Flan-T5-Large variant is what makes the
cost story in this project credible, and it fits in ~1.6 GB of VRAM at fp16.

**On the input template.** MiniCheck's Flan-T5 checkpoint expects a specific
prompt format, and an inexact one produces scores that look plausible and mean
nothing -- the worst possible failure, because nothing downstream notices.
Rather than hard-code a guess, this module *measures*: `verify_template` runs
every candidate template over a small hand-built probe set of obviously
supported and obviously unsupported claims and keeps the one that separates
them best (by AUROC). The chosen template, its probe AUROC and its accuracy are
recorded on the detector and written into every verdict's metadata.

A template that cannot clear `min_probe_auroc` is rejected and the detector
refuses to run, because a judge that cannot tell "Paris is the capital of
France" from "Paris is the capital of Brazil" given the evidence is not a judge.

If the official `minicheck` package is installed it is preferred outright: it
owns the true format and the document chunking, so there is nothing to verify.

Trap #2 from the brief is satisfied structurally: the judge is Flan-T5, the
corrector is a decoder-only chat model. Do not switch the judge to the
generator's family to save VRAM.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

from ..types import Claim, Passage
from .base import Detector

# Candidate input templates, most likely first. `verify_template` picks
# empirically; this list is a search space, not an assertion.
TEMPLATE_CANDIDATES: tuple[str, ...] = (
    "predict: {doc}\n\n{claim}",
    "predict: premise: {doc} hypothesis: {claim}",
    "premise: {doc} hypothesis: {claim}",
    "{doc}\n\nQuestion: Is the following claim supported by the passage above? "
    "Claim: {claim}\nAnswer:",
    "Document: {doc}\nClaim: {claim}\nIs the claim supported by the document?",
)

_SUPPORTED_TOKENS = ("1", "yes", "true", "supported", "entailment")
_UNSUPPORTED_TOKENS = ("0", "no", "false", "unsupported", "contradiction")

# Deliberately easy. This probe does not measure quality -- it catches a
# broken template, a wrong label polarity, or a model that did not load.
# Every item is unambiguous to a competent reader of the evidence.
PROBE_SET: tuple[tuple[str, str, bool], ...] = (
    (
        "Paris is the capital and most populous city of France.",
        "Paris is the capital of France.",
        True,
    ),
    (
        "Paris is the capital and most populous city of France.",
        "Paris is the capital of Brazil.",
        False,
    ),
    (
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
        "The Eiffel Tower was completed in 1889.",
        True,
    ),
    (
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
        "The Eiffel Tower was completed in 1923.",
        False,
    ),
    (
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Water boils at 100 degrees Celsius at sea level.",
        True,
    ),
    (
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Water boils at 50 degrees Celsius at sea level.",
        False,
    ),
    (
        "Mars has two known moons, Phobos and Deimos.",
        "Mars has two moons.",
        True,
    ),
    (
        "Mars has two known moons, Phobos and Deimos.",
        "Mars has seventeen moons.",
        False,
    ),
    (
        "Jane Austen published Pride and Prejudice in 1813.",
        "Pride and Prejudice was published in 1813.",
        True,
    ),
    (
        "Jane Austen published Pride and Prejudice in 1813.",
        "Pride and Prejudice was written by Charles Dickens.",
        False,
    ),
)


class TemplateVerificationError(RuntimeError):
    """Raised when no candidate template separates the probe set."""


class MiniCheckDetector(Detector):
    """Wraps MiniCheck behind the project's `Detector` interface."""

    name = "minicheck"

    def __init__(
        self,
        model: str = "lytang/MiniCheck-Flan-T5-Large",
        flag_threshold: float = 0.5,
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 8,
        max_input_tokens: int = 2048,
        backend: str = "auto",  # "auto" | "package" | "transformers"
        template: Optional[str] = None,
        min_probe_auroc: float = 0.85,
        chunk_passages: bool = True,
    ) -> None:
        super().__init__(flag_threshold)
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.backend = backend
        self.chunk_passages = chunk_passages
        self.min_probe_auroc = min_probe_auroc

        self.template = template
        self.template_report: dict[str, Any] = {}
        self._scorer: Any = None
        self._model: Any = None
        self._tokenizer: Any = None

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._scorer is not None or self._model is not None:
            return

        if self.backend in ("auto", "package"):
            try:
                from minicheck.minicheck import MiniCheck  # type: ignore

                self._scorer = MiniCheck(
                    model_name=_package_alias(self.model_name),
                    enable_prefix_caching=False,
                )
                self.backend = "package"
                self.template_report = {
                    "source": "official_package",
                    "verified": True,
                }
                return
            except Exception as exc:  # noqa: BLE001
                if self.backend == "package":
                    raise
                warnings.warn(
                    f"minicheck package unavailable ({type(exc).__name__}: {exc}); "
                    "loading the checkpoint directly and verifying the input "
                    "template against the probe set.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._load_transformers()
        self.backend = "transformers"

        if self.template is None:
            self.template, self.template_report = self.verify_template()
        else:
            self.template_report = {
                "source": "caller_supplied",
                "template": self.template,
                **self._probe(self.template),
            }

    def _load_transformers(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        from .._hf import load_kwargs, resolve_device, with_dtype_fallback

        device = resolve_device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            with_dtype_fallback(
                AutoModelForSeq2SeqLM.from_pretrained,
                self.model_name,
                **load_kwargs(device, self.dtype),
            )
            .to(device)
            .eval()
        )
        self.device = device

    # -- template verification --------------------------------------------

    def verify_template(self) -> tuple[str, dict[str, Any]]:
        """Pick the template that best separates the probe set.

        Returns (template, report). Raises `TemplateVerificationError` when no
        candidate clears `min_probe_auroc` -- a silent fallback here would put
        meaningless numbers in the paper.
        """
        results = []
        for cand in TEMPLATE_CANDIDATES:
            try:
                results.append({"template": cand, **self._probe(cand)})
            except Exception as exc:  # noqa: BLE001
                results.append({"template": cand, "auroc": 0.0, "error": str(exc)})

        # Rank by AUROC (can the template separate at all?) then by the
        # margin between the two classes. NOT by accuracy at a fixed 0.5
        # threshold: MiniCheck's raw scores are compressed well above 0.5, so
        # accuracy@0.5 measures calibration, not template fit, and using it
        # here would reject a perfectly good template.
        results.sort(
            key=lambda r: (r.get("auroc", 0.0), r.get("margin", 0.0)), reverse=True
        )
        best = results[0]

        if best.get("auroc", 0.0) < self.min_probe_auroc:
            raise TemplateVerificationError(
                f"no MiniCheck input template separated the probe set "
                f"(best AUROC {best.get('auroc', 0.0):.3f} < "
                f"{self.min_probe_auroc}). The checkpoint may have changed, or "
                f"the label tokens may differ. Install the official `minicheck` "
                f"package, or pass an explicit template= after checking the "
                f"authors' repo. Candidates tried: "
                + "; ".join(f"{r['template'][:40]!r}->{r.get('auroc', 0):.2f}" for r in results)
            )

        return best["template"], {
            "source": "empirical_probe",
            "verified": True,
            "chosen": best["template"],
            "probe_auroc": best["auroc"],
            "probe_accuracy": best["accuracy"],
            "probe_n": len(PROBE_SET),
            "probe_margin": best.get("margin", 0.0),
            "needs_calibration": best.get("needs_calibration", False),
            "all_candidates": [
                {
                    "template": r["template"],
                    "auroc": r.get("auroc", 0.0),
                    "margin": round(r.get("margin", 0.0), 4),
                }
                for r in results
            ],
        }

    def _probe(self, template: str) -> dict[str, Any]:
        """Score the probe set with one template."""
        from ..eval.metrics import auroc

        docs = [d for d, _, _ in PROBE_SET]
        claims = [c for _, c, _ in PROBE_SET]
        gold = [supported for _, _, supported in PROBE_SET]

        probs = self._raw_scores(docs, claims, template)
        # auroc() wants the positive class ranked high; here positive =
        # supported, so pass the support probability directly.
        score = auroc(probs, gold)
        acc = sum(1 for p, g in zip(probs, gold) if (p >= 0.5) == g) / len(gold)
        mean_sup = sum(p for p, g in zip(probs, gold) if g) / max(1, sum(gold))
        mean_uns = sum(p for p, g in zip(probs, gold) if not g) / max(
            1, len(gold) - sum(gold)
        )
        return {
            "auroc": 0.0 if score != score else float(score),  # nan -> 0
            "accuracy": float(acc),
            "margin": float(mean_sup - mean_uns),
            "mean_prob_supported": mean_sup,
            "mean_prob_unsupported": mean_uns,
            # A template can rank perfectly while every score sits above 0.5.
            # The router thresholds on the probability, so say so loudly.
            "needs_calibration": bool(mean_uns > 0.5 or mean_sup < 0.5),
        }

    # -- scoring -----------------------------------------------------------

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        self._ensure_loaded()
        if self.backend == "package":
            return self._score_package([c.text for c in claims], passages)
        return self._score_transformers([c.text for c in claims], passages)

    def _score_package(self, claims: Sequence[str], passages: Sequence[Passage]) -> list[float]:
        doc = "\n\n".join(p.text for p in passages)
        _labels, probs, _chunks, _info = self._scorer.score(
            docs=[doc] * len(claims), claims=list(claims)
        )
        return [float(p) for p in probs]

    def _score_transformers(
        self, claims: Sequence[str], passages: Sequence[Passage]
    ) -> list[float]:
        """Score each claim against each passage and take the max.

        A claim supported by any one passage is supported. Concatenating all
        passages into a single premise instead would bury the supporting
        sentence in noise -- the "middle curse" the RAG hallucination survey
        names as a generation deficiency, and it applies to judges too.
        """
        chunks = (
            [p.text for p in passages]
            if self.chunk_passages
            else ["\n\n".join(p.text for p in passages)]
        )
        if not chunks:
            return [0.0] * len(claims)

        docs = [chunk for _ in claims for chunk in chunks]
        reps = [c for c in claims for _ in chunks]
        flat = self._raw_scores(docs, reps, self.template)

        n = len(chunks)
        return [max(flat[i * n : (i + 1) * n]) for i in range(len(claims))]

    def _raw_scores(
        self, docs: Sequence[str], claims: Sequence[str], template: str
    ) -> list[float]:
        """P(supported) for aligned (doc, claim) pairs under one template."""
        import torch  # type: ignore

        prompts = [template.format(doc=d, claim=c) for d, c in zip(docs, claims)]
        yes_ids = self._label_token_ids(_SUPPORTED_TOKENS)
        no_ids = self._label_token_ids(_UNSUPPORTED_TOKENS)

        out: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)

            with torch.no_grad():
                dec = torch.zeros((len(batch), 1), dtype=torch.long, device=self.device)
                logits = self._model(**enc, decoder_input_ids=dec).logits[:, 0, :]
                probs = torch.softmax(logits.float(), dim=-1)
                yes = probs[:, yes_ids].sum(dim=-1)
                no = probs[:, no_ids].sum(dim=-1)
                # Renormalise over the label tokens so the result is a
                # probability over {supported, unsupported} rather than over
                # the whole vocabulary.
                out.extend((yes / (yes + no).clamp(min=1e-6)).tolist())

        return out

    def _label_token_ids(self, words: tuple[str, ...]) -> list[int]:
        ids = set()
        for w in words:
            for variant in (w, w.capitalize(), w.upper(), f" {w}"):
                toks = self._tokenizer.encode(variant, add_special_tokens=False)
                if toks:
                    ids.add(toks[0])
        return sorted(ids)

    # -- introspection -----------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "template": self.template,
            "template_report": self.template_report,
            "loaded": self._model is not None or self._scorer is not None,
        }


def _package_alias(model_name: str) -> str:
    """The official package takes short aliases, not HF repo ids."""
    return {
        "lytang/MiniCheck-Flan-T5-Large": "flan-t5-large",
        "lytang/MiniCheck-RoBERTa-Large": "roberta-large",
        "lytang/MiniCheck-DeBERTa-v3-Large": "deberta-v3-large",
        "bespokelabs/Bespoke-MiniCheck-7B": "Bespoke-MiniCheck-7B",
    }.get(model_name, model_name)


def build_detector(cfg, calibrate: bool = True, **kwargs) -> Detector:
    """Resolve `cfg.detector.name` to a detector, calibrated where possible.

    Baselines need no GPU and no download, which is what keeps the Gate 1
    script and the test suite runnable on any machine.

    Calibration is applied automatically when a scaler exists for this
    detector, because the alternative is worse: MiniCheck's raw scores all sit
    above 0.5, so an uncalibrated run never reaches the router's REWRITE
    branch and degenerates into `noop` while every metric still prints a
    plausible number.
    """
    from .baselines import BASELINES

    name: str = cfg.detector.name
    d = cfg.detector

    if name in BASELINES:
        base: Detector = BASELINES[name](flag_threshold=d.flag_threshold)
    elif name == "minicheck":
        base = MiniCheckDetector(
            model=d.model,
            flag_threshold=d.flag_threshold,
            device=d.device,
            dtype=d.dtype,
            batch_size=d.batch_size,
            **kwargs,
        )
    elif name == "nli":
        from .nli import NLIDetector

        base = NLIDetector(
            flag_threshold=d.flag_threshold,
            model=getattr(d, "nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
            device=d.device,
            dtype=d.dtype,
            batch_size=d.batch_size,
        )
    elif name == "ensemble":
        from .ensemble import build_ensemble

        base = build_ensemble(cfg)
    else:
        raise ValueError(f"unknown detector: {name!r}")

    return _maybe_calibrate(base, cfg) if calibrate else base


def _maybe_calibrate(detector: Detector, cfg) -> Detector:
    """Wrap in a fitted scaler when one is available and applicable."""
    import json
    import warnings
    from pathlib import Path

    from .calibration import CalibratedDetector, TemperatureScaler

    configured = getattr(cfg.detector, "calibration_path", None)
    if configured == "":
        return detector  # explicitly opted out

    path = Path(configured) if configured else Path(
        f"runs/calibration_{cfg.detector.name}.json"
    )
    if not path.exists():
        if configured:
            raise FileNotFoundError(f"calibration_path {path} does not exist")
        return detector

    payload = json.loads(path.read_text(encoding="utf-8"))

    # A scaler fitted on the synthetic fixture is a smoke-test artifact. Using
    # it on real data would be a silent methodological error, so refuse.
    is_sample_run = str(getattr(cfg, "dataset", "")).endswith("sample")
    if payload.get("synthetic") and not is_sample_run:
        warnings.warn(
            f"{path} was fitted on the synthetic fixture; refusing to apply it "
            f"to dataset={cfg.dataset!r}. Re-fit on RAGTruth: "
            "python scripts/calibrate_detector.py --detector "
            f"{cfg.detector.name}",
            RuntimeWarning,
            stacklevel=3,
        )
        return detector

    scaler = TemperatureScaler(
        **{k: v for k, v in payload.items() if k in TemperatureScaler.__annotations__}
    )
    wrapped = CalibratedDetector(detector, scaler)

    # The flag threshold is part of the fitted artifact, not a constant. A
    # calibrated detector used at an assumed 0.5 flags nothing on a corpus
    # that is mostly supported.
    chosen = payload.get("threshold")
    if isinstance(chosen, dict) and "threshold" in chosen:
        wrapped.flag_threshold = float(chosen["threshold"])
        wrapped.threshold_source = chosen
    return wrapped
