"""Learned router.

Training labels come free from the Adaptive-RAG auto-labelling trick: run every
action on every flagged claim, see which one produced a BETTER outcome, use
that as the label. No human annotation. `scripts/build_router_labels.py`
produces the training file; this module consumes it.

Inference is a pure-Python multinomial logistic regression reading coefficients
from JSON, so a trained router runs with no sklearn, no torch and no GPU --
which keeps the latency column honest. Training uses sklearn if present.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..types import Action, Passage, RoutingDecision
from .base import ClaimState, decision
from .rules import RuleRouter

# Order is part of the saved model; changing it invalidates coefficient files.
FEATURE_ORDER: tuple[str, ...] = (
    "support_prob",
    "detector_confidence",
    "coverage",
    "contradiction",
    "claim_len_tokens",
    "has_number",
    "iteration",
    "attempts",
    "prior_rewrite",
    "prior_reretrieve",
    "last_rewrite_failed",
)

ACTION_ORDER: tuple[Action, ...] = (
    Action.ACCEPT,
    Action.RERETRIEVE,
    Action.REWRITE,
    Action.ABSTAIN,
)


@dataclass
class LinearPolicy:
    """Coefficients for one multinomial logistic head."""

    weights: list[list[float]]   # [n_actions][n_features]
    bias: list[float]            # [n_actions]
    feature_order: tuple[str, ...] = FEATURE_ORDER
    action_order: tuple[str, ...] = tuple(a.value for a in ACTION_ORDER)
    # Feature standardisation, saved with the model so inference matches
    # training exactly.
    mean: Optional[list[float]] = None
    scale: Optional[list[float]] = None

    def predict_proba(self, feats: dict[str, float]) -> dict[str, float]:
        x = [float(feats[k]) for k in self.feature_order]
        if self.mean and self.scale:
            x = [
                (xi - m) / (s if s else 1.0)
                for xi, m, s in zip(x, self.mean, self.scale)
            ]
        logits = [
            b + sum(w * xi for w, xi in zip(row, x))
            for row, b in zip(self.weights, self.bias)
        ]
        top = max(logits)
        exps = [math.exp(z - top) for z in logits]
        total = sum(exps)
        return {a: e / total for a, e in zip(self.action_order, exps)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "bias": self.bias,
                    "feature_order": list(self.feature_order),
                    "action_order": list(self.action_order),
                    "mean": self.mean,
                    "scale": self.scale,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=d["weights"],
            bias=d["bias"],
            feature_order=tuple(d["feature_order"]),
            action_order=tuple(d["action_order"]),
            mean=d.get("mean"),
            scale=d.get("scale"),
        )


class LearnedRouter:
    """Policy-backed router with the same hard constraints as the rule router.

    The learned policy chooses among actions, but it does NOT get to override
    the attempt budget: an exhausted claim abstains regardless of what the
    model wants. Letting a learned policy loop forever is how these systems
    blow up their cost column.
    """

    name = "learned"

    def __init__(
        self,
        policy: LinearPolicy,
        max_attempts: int = 2,
        fallback: Optional[RuleRouter] = None,
    ) -> None:
        self.policy = policy
        self.max_attempts = max_attempts
        self.fallback = fallback or RuleRouter(max_attempts=max_attempts)

    @classmethod
    def from_config(cls, cfg) -> "LearnedRouter":
        path = getattr(cfg.router, "policy_path", None) or "runs/router_policy.json"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no trained policy at {path}. Run "
                "`python scripts/build_router_labels.py` then "
                "`python scripts/train_router.py`."
            )
        return cls(
            LinearPolicy.load(path), max_attempts=cfg.loop.max_attempts_per_claim
        )

    def route(self, state: ClaimState, passages: list[Passage]) -> RoutingDecision:
        if not state.verdict.is_flagged:
            return decision(state, Action.ACCEPT, "not_flagged", self.name)
        if state.attempts >= self.max_attempts:
            return decision(
                state, Action.ABSTAIN, "attempt_budget_exhausted", self.name
            )

        probs = self.policy.predict_proba(state.features())
        best = max(probs, key=probs.__getitem__)
        return decision(
            state,
            Action(best),
            f"policy_argmax_p={probs[best]:.3f}",
            self.name,
            action_probs=probs,
        )


def train_policy(
    features: Sequence[dict[str, float]],
    labels: Sequence[str],
    C: float = 1.0,
    seed: int = 20260920,
) -> LinearPolicy:
    """Fit the policy on auto-labelled (features, best_action) pairs.

    Deliberately a linear model: the router's decision rule has to be
    inspectable, and with ten features and a few thousand claims anything
    bigger would memorise the dev set.
    """
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install scikit-learn numpy to train the router") from exc

    X = np.array([[f[k] for k in FEATURE_ORDER] for f in features], dtype=float)
    y = np.array(list(labels))

    scaler = StandardScaler().fit(X)
    # No `multi_class` argument: scikit-learn removed it (1.7), and the
    # multinomial behaviour it selected is now the default for more than two
    # classes. Passing it raised TypeError before a single row was fitted, so
    # the learned router had never once been trained on real labels.
    #
    # No class reweighting either, deliberately. The labels are ~94% ACCEPT,
    # and a router that therefore learns "almost never act" is reporting what
    # the auto-labelling found, not failing. Rebalancing after seeing that
    # distribution would be tuning the model toward the answer we wanted.
    clf = LogisticRegression(
        C=C, max_iter=2000, random_state=seed
    ).fit(scaler.transform(X), y)

    # Re-index sklearn's class order onto the canonical ACTION_ORDER so saved
    # policies are comparable across training runs with different label mixes.
    classes = list(clf.classes_)
    weights, bias = [], []
    for action in ACTION_ORDER:
        if action.value in classes:
            i = classes.index(action.value)
            if len(classes) == 2:
                # sklearn gives one row for binary problems.
                sign = 1.0 if i == 1 else -1.0
                weights.append((sign * clf.coef_[0]).tolist())
                bias.append(float(sign * clf.intercept_[0]))
            else:
                weights.append(clf.coef_[i].tolist())
                bias.append(float(clf.intercept_[i]))
        else:
            # Action never selected in training: make it unreachable rather
            # than letting an untrained head fire at random.
            weights.append([0.0] * len(FEATURE_ORDER))
            bias.append(-1e3)

    return LinearPolicy(
        weights=weights,
        bias=bias,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
    )
