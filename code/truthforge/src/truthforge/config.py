"""Run configuration.

Defaults are tuned for a single 8 GB consumer GPU (RTX 4060 Laptop), which is
the hardware this project was scaffolded on. Every threshold that the router
reads lives here so that a config diff fully explains a results diff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class RetrievalConfig:
    top_k: int = 5
    bm25_k: int = 20
    dense_k: int = 20
    rrf_k: int = 60  # RRF smoothing constant; 60 is the Cormack et al. default
    dense_model: str = "BAAI/bge-small-en-v1.5"  # ~130 MB, fits anywhere
    # On re-retrieval, widen instead of repeating the same query.
    reretrieve_top_k: int = 10


@dataclass
class GeneratorConfig:
    """The answering model and corrector. NOT the same family as the judge.

    Sizing on 8 GB, measured on this project's card (RTX 4060 Laptop):

        MiniCheck-Flan-T5-Large   ~1.6 GB
        DeBERTa-v3-base MNLI      ~0.4 GB
        -> ~2.0 GB is already gone before the corrector loads.

    A 3B corrector at fp16 is ~6.2 GB and does NOT co-reside with the judges;
    it OOMs partway through a run, which is the worst time to find out. 1.5B
    (~3.1 GB) fits with headroom and is the default for that reason.

    A weaker corrector inflates the worse-rate, which flatters this project's
    story -- see weak link #5 in DECISIONS.md. The mitigation is that the
    `always_correct` arm uses the SAME corrector, so the comparison stays
    internally valid. To run 3B or 7B, load the judge and the corrector in
    separate passes and footnote the latency column.
    """

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "float16"
    load_in_4bit: bool = False
    # One corrected sentence. 256 let a rambling model spend the whole
    # budget echoing the answer back -- the slowest path, and the one the
    # output guard discards anyway. Measured rewrites run 12-25 tokens;
    # 64 leaves generous headroom and roughly halves generation time on a
    # downclocked GPU.
    max_new_tokens: int = 64
    temperature: float = 0.0
    device: str = "cuda"


@dataclass
class DetectorConfig:
    """The judge. Different family from the generator by construction."""

    # "minicheck" | "nli" | "ensemble" | "lexical" | "always_unsupported" | "random"
    name: str = "minicheck"
    model: str = "lytang/MiniCheck-Flan-T5-Large"  # 770M, ~1.5 GB fp16
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # 184M
    dtype: str = "float16"
    device: str = "cuda"
    batch_size: int = 8
    # Below this P(supported), a claim is flagged.
    flag_threshold: float = 0.5
    # Logit-space weights when name == "ensemble".
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {"minicheck": 1.0, "nli": 0.7, "lexical": 0.3}
    )
    # Fitted temperature scaler. MiniCheck's raw scores all sit above 0.5, so
    # without this the router never reaches its REWRITE branch and the system
    # silently degenerates to `noop`. None = auto-discover
    # runs/calibration_<name>.json; "" = explicitly run uncalibrated.
    calibration_path: Optional[str] = None


@dataclass
class RouterConfig:
    """Thresholds for the rule router. These ARE the ablation table."""

    name: str = "rule"  # "rule" | "learned" | "always_correct" | "oracle" | "noop"

    # Detector is confident the claim is fine -> accept, spend nothing.
    accept_support_prob: float = 0.70
    # Detector is confident it is wrong AND evidence looks relevant -> rewrite.
    rewrite_support_prob: float = 0.30
    # Evidence coverage below this means "we never retrieved the right thing",
    # so rewriting cannot help -> re-retrieve instead.
    thin_evidence_coverage: float = 0.25
    # Detector sits near the boundary -> accept rather than burn compute on a
    # coin flip. Raising this is the cheapest over-correction lever.
    uncertain_band: float = 0.10  # |p - 0.5| < band
    # Two passages that both score high but disagree -> contradiction -> abstain.
    contradiction_margin: float = 0.15
    # Refuse the whole turn when the question's own content words are largely
    # absent from the retrieved evidence. This is a TURN-level gate, applied in
    # the loop rather than in `route()`, because it is a judgement about the
    # question and the evidence rather than about any one claim.
    #
    # It exists because claim-level support verification is structurally blind
    # to topic drift: evidence about X plus a question about Y yields an answer
    # about X in which every claim is genuinely supported. The open-corpus
    # evaluation measured 44% of unanswerable questions receiving a confident
    # answer, with no arm differing from the base generator.
    #
    # 0.55 was selected on a DEVELOPMENT set (train split, seed 777, 80
    # questions) maximising correct_refusal - over_refusal, before any
    # evaluation number was looked at. See
    # scripts/select_answerability_threshold.py and runs/answerability_threshold.json.
    # At this point the dev set gives correct 0.825 / over 0.350; a deployment
    # that weighs a needless refusal more heavily should read a lower tau off
    # the printed curve rather than re-tuning this one.
    min_question_coverage: float = 0.55
    # Off by default: the gate is only meaningful when THIS system did the
    # retrieving. See `TruthForgeLoop.run`.
    answerability_gate: bool = False

    # Trained policy for name == "learned".
    policy_path: str = "runs/router_policy.json"


@dataclass
class LoopConfig:
    max_iterations: int = 2
    # Abstain if a claim is still flagged after this many correction attempts.
    max_attempts_per_claim: int = 2
    abstain_message: str = (
        "I could not verify this from the retrieved sources."
    )
    # What an abstained claim leaves behind.
    #   "message" -- swap in abstain_message. Transparent: the reader can see
    #                something was withheld. Clunky to read.
    #   "remove"  -- delete the sentence. Produces a clean, fully grounded
    #                answer, but the reader is not told anything was dropped.
    #
    # NOTE for the paper: "remove" deletes text, which is what the info_loss
    # category measures. Dropping a FABRICATED claim is not info loss -- no
    # evidence-supported content is lost, and the outcome classifier scores it
    # on exactly that basis. But a system that defaults to deleting whatever it
    # cannot fix could flatter its own faithfulness score, so report which
    # style produced any number you quote.
    abstain_style: str = "message"


@dataclass
class Config:
    seed: int = 20260920
    dataset: str = "ragtruth"
    split: str = "test"
    limit: int | None = None
    out_dir: str = "runs"
    run_name: str = "dev"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        nested = {
            "retrieval": RetrievalConfig,
            "generator": GeneratorConfig,
            "detector": DetectorConfig,
            "router": RouterConfig,
            "loop": LoopConfig,
        }
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        for key, val in d.items():
            if key not in known:
                raise ValueError(f"unknown config key: {key!r}")
            kwargs[key] = nested[key](**val) if key in nested else val
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pip install pyyaml to load .yaml configs") from exc
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))
