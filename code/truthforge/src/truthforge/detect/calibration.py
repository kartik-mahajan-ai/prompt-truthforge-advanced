"""Probability calibration for detectors.

The router thresholds on `support_prob`: accept above 0.70, rewrite below 0.30,
do nothing inside a band around 0.5. Those numbers only mean something if the
detector's output is a calibrated probability. Raw softmax scores from a
seq2seq head are usually badly overconfident, which silently collapses the
router -- if nearly every score is < 0.05 or > 0.95, the uncertain band is
empty and the rule router degenerates into `always_correct`.

So calibration is not polish here. It is what makes the router's thresholds
correspond to anything, and an uncalibrated run should be reported as such.

`TemperatureScaler` fits a single scalar on a held-out split by minimising NLL
with bisection on the gradient -- no torch, no sklearn, so calibration runs
anywhere. `reliability` reports ECE and the per-bin table that goes in the
appendix.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_EPS = 1e-6


def _clip(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class TemperatureScaler:
    """p_calibrated = sigmoid(logit(p_raw) / T + b).

    T > 1 softens an overconfident detector; T < 1 sharpens an underconfident
    one. The bias `b` corrects a detector that is systematically too eager to
    call things supported, which matters more than temperature when the class
    balance of the fitting split differs from deployment.
    """

    temperature: float = 1.0
    bias: float = 0.0
    fitted_on: int = 0
    nll_before: float = float("nan")
    nll_after: float = float("nan")
    ece_before: float = float("nan")
    ece_after: float = float("nan")

    def __call__(self, p: float) -> float:
        return _sigmoid(_logit(p) / self.temperature + self.bias)

    def transform(self, probs: Sequence[float]) -> list[float]:
        return [self(p) for p in probs]

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        probs: Sequence[float],
        labels: Sequence[bool],
        fit_bias: bool = True,
        iters: int = 60,
    ) -> "TemperatureScaler":
        """Fit on (raw P(supported), gold supported) pairs.

        Coordinate descent: golden-section on T, then on b, alternating. Both
        objectives are smooth and unimodal in one variable, so this is stable
        and needs no gradients.
        """
        if len(probs) != len(labels):
            raise ValueError("probs and labels must be the same length")
        if len(probs) < 20:
            raise ValueError(
                f"refusing to calibrate on {len(probs)} points -- a temperature "
                "fitted on a handful of examples is noise. Use >= 20, ideally "
                "a few hundred."
            )
        if len(set(labels)) < 2:
            raise ValueError("calibration split contains only one class")

        z = [_logit(p) for p in probs]
        y = [1.0 if lab else 0.0 for lab in labels]

        def nll(T: float, b: float) -> float:
            total = 0.0
            for zi, yi in zip(z, y):
                p = _clip(_sigmoid(zi / T + b))
                total -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
            return total / len(z)

        T, b = 1.0, 0.0
        for _ in range(4):
            T = _golden_min(lambda t: nll(t, b), 0.05, 20.0, iters)
            if fit_bias:
                b = _golden_min(lambda bb: nll(T, bb), -5.0, 5.0, iters)

        scaler = cls(temperature=T, bias=b, fitted_on=len(z))
        scaler.nll_before = nll(1.0, 0.0)
        scaler.nll_after = nll(T, b)
        scaler.ece_before = reliability(probs, labels)["ece"]
        scaler.ece_after = reliability(scaler.transform(probs), labels)["ece"]
        return scaler

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "bias": self.bias,
            "fitted_on": self.fitted_on,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


def _golden_min(f, lo: float, hi: float, iters: int) -> float:
    """Golden-section search for the minimum of a unimodal f on [lo, hi]."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def reliability(
    probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> dict:
    """Expected Calibration Error plus the per-bin table.

    ECE is the average gap between confidence and accuracy, weighted by bin
    population. A detector with ECE ~0.3 is not a probability source and its
    router thresholds are arbitrary.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must be the same length")
    if not probs:
        return {"ece": float("nan"), "bins": [], "n": 0}

    bins: list[dict] = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "conf": 0.0, "acc": 0.0}
        for i in range(n_bins)
    ]
    for p, lab in zip(probs, labels):
        i = min(n_bins - 1, int(_clip(p) * n_bins))
        bins[i]["n"] += 1
        bins[i]["conf"] += p
        bins[i]["acc"] += 1.0 if lab else 0.0

    n = len(probs)
    ece = 0.0
    for b in bins:
        if b["n"]:
            b["conf"] /= b["n"]
            b["acc"] /= b["n"]
            ece += (b["n"] / n) * abs(b["acc"] - b["conf"])
        else:
            b["conf"] = b["acc"] = 0.0

    return {
        "ece": ece,
        "bins": bins,
        "n": n,
        "base_rate": sum(1 for lab in labels if lab) / n,
        "mean_confidence": sum(probs) / n,
    }


def spread(probs: Sequence[float], band: float = 0.10) -> dict:
    """How much of the score mass lands where the router can act on it.

    A detector that puts 95% of its mass outside [0.1, 0.9] gives the router
    nothing to be uncertain about, and the `uncertain_band` ablation will look
    flat for a reason that has nothing to do with the research question. Check
    this before concluding the band does not matter.
    """
    if not probs:
        return {"n": 0}
    n = len(probs)
    return {
        "n": n,
        "frac_saturated": sum(1 for p in probs if p < 0.05 or p > 0.95) / n,
        "frac_in_uncertain_band": sum(1 for p in probs if abs(p - 0.5) < band) / n,
        "frac_actionable_rewrite": sum(1 for p in probs if p <= 0.30) / n,
        "min": min(probs),
        "max": max(probs),
        "mean": sum(probs) / n,
    }


class CalibratedDetector:
    """Wraps any detector and applies a fitted scaler to its scores.

    Composition rather than a mixin so that an uncalibrated run and a
    calibrated run differ by exactly one object in the config, and the run log
    records which one was used.
    """

    def __init__(self, detector, scaler: TemperatureScaler) -> None:
        self.detector = detector
        self.scaler = scaler
        self.name = f"{detector.name}+calibrated"
        self.flag_threshold = detector.flag_threshold

    def score_batch(self, claims, passages) -> list[float]:
        return self.scaler.transform(self.detector.score_batch(claims, passages))

    def check(self, claims, passages):
        verdicts = self.detector.check(claims, passages)
        from ..types import Label

        for v in verdicts:
            raw = v.support_prob
            v.support_prob = self.scaler(raw)
            v.label = (
                Label.SUPPORTED if v.support_prob >= self.flag_threshold
                else Label.UNSUPPORTED
            )
            v.detector = self.name
            v.meta["raw_support_prob"] = raw
            v.meta["calibration"] = {
                "temperature": self.scaler.temperature,
                "bias": self.scaler.bias,
            }
        return verdicts

    def info(self) -> dict:
        base = self.detector.info() if hasattr(self.detector, "info") else {}
        return {**base, "calibration": self.scaler.to_dict()}


# --------------------------------------------------------------------------
# Operating-point selection
# --------------------------------------------------------------------------


def select_threshold(
    probs: Sequence[float],
    labels: Sequence[bool],
    objective: str = "balanced_accuracy",
    min_precision: float = 0.0,
) -> dict:
    """Choose the flag threshold from data instead of assuming 0.5.

    This exists because assuming it was wrong, and silently so. MiniCheck's
    scores on RAGTruth QA occupy roughly [0.50, 0.99] -- the corpus is ~90%
    supported at claim level, and a well-calibrated detector on such a corpus
    SHOULD be confident most of the time. At a 0.5 threshold it therefore
    flags nothing at all, scores 0.5 balanced accuracy, and the whole
    correction loop becomes inert while every metric still prints.

    Note that calibration does not fix this and is not meant to: temperature
    scaling makes the probabilities honest, and an honest probability on a 90%
    corpus is usually high. Discrimination (AUROC) and operating point are
    separate concerns, and only the second one is a threshold.

    MUST be fitted on a split that is not the test set.

    Args:
        probs: P(supported) from the detector.
        labels: True where the claim is genuinely UNSUPPORTED.
        objective: "balanced_accuracy" | "f1" | "youden".
        min_precision: ignore thresholds below this precision; use when false
            positives are expensive, since every flag costs a correction.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must be the same length")
    if not probs:
        raise ValueError("no scores to select a threshold from")
    if len(set(labels)) < 2:
        raise ValueError("threshold selection needs both classes present")

    candidates = sorted({round(p, 4) for p in probs})
    best = {"threshold": 0.5, "score": -1.0, "objective": objective}

    for t in candidates:
        tp = sum(1 for p, y in zip(probs, labels) if p < t and y)
        fp = sum(1 for p, y in zip(probs, labels) if p < t and not y)
        fn = sum(1 for p, y in zip(probs, labels) if p >= t and y)
        tn = sum(1 for p, y in zip(probs, labels) if p >= t and not y)

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if precision < min_precision:
            continue

        if objective == "f1":
            score = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
        elif objective == "youden":
            score = tpr + tnr - 1
        else:
            score = (tpr + tnr) / 2

        if score > best["score"]:
            best = {
                "threshold": float(t),
                "score": float(score),
                "objective": objective,
                "precision": round(precision, 4),
                "recall": round(tpr, 4),
                "specificity": round(tnr, 4),
                "flag_rate": round((tp + fp) / len(probs), 4),
                "n_fitted_on": len(probs),
                "base_rate_unsupported": round(sum(labels) / len(labels), 4),
            }

    if best["score"] < 0:
        raise ValueError(
            f"no threshold reached min_precision={min_precision}; the detector "
            "cannot hit that precision on this data at any operating point"
        )
    return best
