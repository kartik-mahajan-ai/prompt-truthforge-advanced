"""Ensemble detector.

MiniCheck and NLI fail differently. MiniCheck was trained on document-grounded
fact verification and handles long, multi-sentence evidence; the NLI model is
sharper on single-sentence entailment but is thrown by claims that need two
passages combined. Lexical overlap catches neither, but it is the only signal
that reliably spots a claim whose content words appear *nowhere* in the
evidence -- the case where both neural judges can still hallucinate support.

Combination is a weighted mean in **logit** space: log-odds pooling, the
product-of-experts rule. That is the principled combination when the members
are calibrated and their errors are roughly independent, and it deliberately
lets a confident member move the result.

The catch, and it is a real one: log-odds pooling *amplifies* differences
between saturated scores rather than damping them. logit(0.999) = 6.9 against
logit(0.99) = 4.6 is a gap of 2.3, where in probability space the two are
0.009 apart and interchangeable. Both neural judges here saturate hard, so
**an uncalibrated member can swamp the pool**. Two guards:

  1. Calibrate members individually before ensembling (`CalibratedDetector`).
  2. `clip_logit` bounds each member's contribution. At the default 4.0 no
     single member can push past the equivalent of p=0.982 on its own, so one
     overconfident judge cannot overrule two moderate ones.

Neither guard is optional dressing -- without them the "ensemble" is whichever
member happens to be most saturated.

The weights are a config knob and belong in the ablation table. They are NOT
tuned on the test split.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..types import Claim, Label, Passage, Verdict
from .base import Detector

_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class EnsembleDetector(Detector):
    """Weighted logit-space combination of several detectors.

    `agreement` in the verdict metadata is the useful by-product: when the
    members disagree, the claim is genuinely borderline, and that is a better
    uncertainty signal for the router than any single model's distance from
    0.5. It is exposed so the router can use it, and so error analysis can ask
    whether disagreement predicts over-correction.
    """

    name = "ensemble"

    def __init__(
        self,
        members: Sequence[tuple[str, Detector, float]],
        flag_threshold: float = 0.5,
        min_weight_sum: float = 1e-3,
        clip_logit: float = 4.0,
    ) -> None:
        super().__init__(flag_threshold)
        if not members:
            raise ValueError("ensemble needs at least one member")
        total = sum(w for _, _, w in members)
        if total < min_weight_sum:
            raise ValueError(f"ensemble weights sum to {total}, which is ~zero")
        if clip_logit <= 0:
            raise ValueError("clip_logit must be positive")
        self.members = list(members)
        self.clip_logit = clip_logit
        self._last_member_scores: dict[str, list[float]] = {}

    def score_batch(self, claims: list[Claim], passages: list[Passage]) -> list[float]:
        per_member: dict[str, list[float]] = {}
        for name, det, _w in self.members:
            per_member[name] = det.score_batch(claims, passages)
        self._last_member_scores = per_member

        total_w = sum(w for _, _, w in self.members)
        c = self.clip_logit
        out = []
        for i in range(len(claims)):
            # Clip each member's log-odds before pooling. Without this a
            # single saturated judge (logit ~ 7) overrules every moderate one.
            z = sum(
                w * max(-c, min(c, _logit(per_member[name][i])))
                for name, _d, w in self.members
            ) / total_w
            out.append(_sigmoid(z))
        return out

    def check(self, claims: list[Claim], passages: list[Passage]) -> list[Verdict]:
        verdicts = super().check(claims, passages)
        scores = self._last_member_scores
        for i, v in enumerate(verdicts):
            member_probs = {name: scores[name][i] for name, _d, _w in self.members}
            v.meta["members"] = {k: round(p, 4) for k, p in member_probs.items()}
            v.meta["agreement"] = _agreement(list(member_probs.values()))
        return verdicts

    def info(self) -> dict:
        return {
            "name": self.name,
            "members": [
                {
                    "name": n,
                    "weight": w,
                    **(d.info() if hasattr(d, "info") else {"detector": d.name}),
                }
                for n, d, w in self.members
            ],
        }


def _agreement(probs: Sequence[float]) -> float:
    """1.0 when every member lands on the same side of 0.5, scaled by margin.

    Returns 0.0 on a split decision. This is the signal the router should
    prefer over a single model's confidence when deciding whether a correction
    is worth attempting.
    """
    if len(probs) < 2:
        return 1.0
    sides = [p >= 0.5 for p in probs]
    if len(set(sides)) > 1:
        return 0.0
    return round(min(abs(p - 0.5) for p in probs) * 2, 4)


DEFAULT_WEIGHTS = {"minicheck": 1.0, "nli": 0.7, "lexical": 0.3}


def build_ensemble(cfg, weights: Optional[dict[str, float]] = None) -> EnsembleDetector:
    """Assemble the default ensemble from a Config.

    Members that fail to load (no GPU, missing weights) are dropped with a
    warning rather than failing the run, but an ensemble that silently
    degenerates to lexical-only would be a disaster to discover in a results
    table -- so the surviving membership is recorded in `info()` and printed by
    the CLI.
    """
    import warnings

    from .baselines import LexicalOverlapDetector

    w = {**DEFAULT_WEIGHTS, **(weights or getattr(cfg.detector, "ensemble_weights", None) or {})}
    d = cfg.detector
    members: list[tuple[str, Detector, float]] = []

    if w.get("minicheck", 0) > 0:
        try:
            from .minicheck import MiniCheckDetector

            det = MiniCheckDetector(
                model=d.model, device=d.device, dtype=d.dtype, batch_size=d.batch_size
            )
            det._ensure_loaded()
            members.append(("minicheck", det, w["minicheck"]))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ensemble: dropping minicheck ({exc})", RuntimeWarning, stacklevel=2)

    if w.get("nli", 0) > 0:
        try:
            from .nli import NLIDetector

            det = NLIDetector(
                model=getattr(d, "nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
                device=d.device,
                dtype=d.dtype,
                batch_size=d.batch_size,
            )
            det.backend.ensure_loaded()
            members.append(("nli", det, w["nli"]))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ensemble: dropping nli ({exc})", RuntimeWarning, stacklevel=2)

    if w.get("lexical", 0) > 0:
        members.append(("lexical", LexicalOverlapDetector(), w["lexical"]))

    if len(members) == 1 and members[0][0] == "lexical":
        warnings.warn(
            "ensemble degenerated to lexical-only: both neural judges failed to "
            "load. Results from this run are NOT an ensemble result.",
            RuntimeWarning,
            stacklevel=2,
        )

    return EnsembleDetector(members, flag_threshold=d.flag_threshold)
