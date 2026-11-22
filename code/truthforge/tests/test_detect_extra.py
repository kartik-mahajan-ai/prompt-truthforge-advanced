"""Tests for calibration, the ensemble, and the model-layer compatibility shims.

Everything here runs without torch. The neural detectors are exercised by
`scripts/verify_models.py`, which needs a GPU; these tests cover the logic
around them, which is where the silent-wrongness bugs live.
"""

from __future__ import annotations

import math
import random

import pytest

from truthforge.config import Config
from truthforge.detect.baselines import AlwaysSupportedDetector, LexicalOverlapDetector
from truthforge.detect.calibration import (
    CalibratedDetector,
    TemperatureScaler,
    reliability,
    spread,
)
from truthforge.detect.ensemble import EnsembleDetector, _agreement
from truthforge.detect.minicheck import PROBE_SET, TEMPLATE_CANDIDATES
from truthforge.types import Claim, Label, Passage

EVIDENCE = [Passage("p0", "The Eiffel Tower was completed in 1889 and is 330 metres tall.")]


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def _overconfident_sample(n: int = 400, seed: int = 7):
    """Scores that rank well but are squashed above 0.5 -- MiniCheck's actual
    failure mode, reproduced without needing the model."""
    rng = random.Random(seed)
    probs, labels = [], []
    for _ in range(n):
        supported = rng.random() < 0.6
        base = 0.80 if supported else 0.55
        probs.append(min(0.999, max(0.001, rng.gauss(base, 0.06))))
        labels.append(supported)
    return probs, labels


def test_scaler_improves_nll_and_ece():
    probs, labels = _overconfident_sample()
    s = TemperatureScaler.fit(probs, labels)
    assert s.nll_after <= s.nll_before
    assert s.ece_after <= s.ece_before


def test_scaler_fixes_the_threshold_problem():
    """The point of calibration here: move scores across the 0.5 boundary so
    the router's REWRITE branch becomes reachable at all."""
    probs, labels = _overconfident_sample()
    before = spread(probs)
    assert before["frac_actionable_rewrite"] == 0.0  # nothing below 0.30

    s = TemperatureScaler.fit(probs, labels)
    after = spread(s.transform(probs))
    assert after["frac_actionable_rewrite"] > 0.0

    acc_before = sum(1 for p, y in zip(probs, labels) if (p >= 0.5) == y) / len(probs)
    cal = s.transform(probs)
    acc_after = sum(1 for p, y in zip(cal, labels) if (p >= 0.5) == y) / len(cal)
    assert acc_after > acc_before


def test_scaler_is_monotonic():
    """Calibration must never reorder claims -- it changes thresholds, not
    the detector's judgement."""
    s = TemperatureScaler(temperature=2.0, bias=-0.5)
    xs = [0.01, 0.2, 0.4, 0.5, 0.6, 0.85, 0.99]
    ys = [s(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))


def test_scaler_refuses_tiny_or_single_class_samples():
    with pytest.raises(ValueError, match="refusing to calibrate"):
        TemperatureScaler.fit([0.5] * 5, [True] * 3 + [False] * 2)
    with pytest.raises(ValueError, match="only one class"):
        TemperatureScaler.fit([0.6] * 30, [True] * 30)


def test_scaler_round_trips(tmp_path):
    probs, labels = _overconfident_sample(120)
    s = TemperatureScaler.fit(probs, labels)
    path = tmp_path / "cal.json"
    s.save(path)
    loaded = TemperatureScaler.load(path)
    assert math.isclose(loaded(0.7), s(0.7), rel_tol=1e-9)


def test_reliability_perfect_and_worst():
    perfect = reliability([0.0, 0.0, 1.0, 1.0], [False, False, True, True])
    assert perfect["ece"] < 1e-6
    worst = reliability([1.0, 1.0], [False, False])
    assert worst["ece"] > 0.99


def test_calibrated_detector_relabels_and_keeps_the_raw_score():
    # A bias big enough to move a saturated score across the boundary: p=1.0
    # clips to logit ~13.8, so -5 would not be nearly enough.
    scaler = TemperatureScaler(temperature=1.0, bias=-20.0)
    det = CalibratedDetector(AlwaysSupportedDetector(), scaler)
    verdicts = det.check([Claim(text="anything at all here")], EVIDENCE)
    v = verdicts[0]
    assert v.label is Label.UNSUPPORTED
    assert v.support_prob < 0.5
    assert v.meta["raw_support_prob"] == 1.0
    assert "calibrated" in v.detector


def test_spread_flags_a_saturated_detector():
    sp = spread([0.001, 0.002, 0.998, 0.999])
    assert sp["frac_saturated"] == 1.0
    assert sp["frac_in_uncertain_band"] == 0.0


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------


class _Fixed(LexicalOverlapDetector):
    """A detector that returns a constant, for testing the combination rule."""

    def __init__(self, value: float, name: str) -> None:
        super().__init__()
        self.value = value
        self.name = name

    def score_batch(self, claims, passages):
        return [self.value] * len(claims)


def test_ensemble_combines_in_logit_space():
    ens = EnsembleDetector([
        ("a", _Fixed(0.9, "a"), 1.0),
        ("b", _Fixed(0.1, "b"), 1.0),
    ])
    p = ens.score_batch([Claim(text="a claim of some length")], EVIDENCE)[0]
    # Equal and opposite logits must cancel to exactly 0.5.
    assert abs(p - 0.5) < 1e-9


def test_ensemble_weights_shift_the_result():
    heavy = EnsembleDetector([
        ("a", _Fixed(0.9, "a"), 3.0),
        ("b", _Fixed(0.1, "b"), 1.0),
    ])
    p = heavy.score_batch([Claim(text="a claim of some length")], EVIDENCE)[0]
    assert p > 0.5


def test_ensemble_records_member_scores_and_agreement():
    ens = EnsembleDetector([
        ("a", _Fixed(0.95, "a"), 1.0),
        ("b", _Fixed(0.9, "b"), 1.0),
    ])
    v = ens.check([Claim(text="a claim of some length")], EVIDENCE)[0]
    assert v.meta["members"] == {"a": 0.95, "b": 0.9}
    assert v.meta["agreement"] > 0.0


def test_agreement_is_zero_on_a_split_decision():
    assert _agreement([0.9, 0.2]) == 0.0
    assert _agreement([0.9, 0.95]) > 0.0
    assert _agreement([0.51, 0.99]) == pytest.approx(0.02, abs=1e-6)


def test_ensemble_rejects_empty_membership():
    with pytest.raises(ValueError):
        EnsembleDetector([])


def test_logit_clipping_stops_one_saturated_member_swamping_the_pool():
    """Log-odds pooling amplifies saturated scores, so contributions are
    clipped. Without the clip a single 0.999 member overrules two moderates."""
    claim = [Claim(text="a claim of some length")]
    members = [
        ("hot", _Fixed(0.9999, "hot"), 1.0),
        ("mid1", _Fixed(0.35, "mid1"), 1.0),
        ("mid2", _Fixed(0.35, "mid2"), 1.0),
    ]
    unclipped = EnsembleDetector(members, clip_logit=50.0)
    clipped = EnsembleDetector(members, clip_logit=4.0)

    p_unclipped = unclipped.score_batch(claim, EVIDENCE)[0]
    p_clipped = clipped.score_batch(claim, EVIDENCE)[0]

    assert p_unclipped > 0.8, "unclipped pooling is dominated by the hot member"
    assert p_clipped < p_unclipped
    # With the clip, two moderates against one saturated member keep it down.
    assert p_clipped < 0.75


def test_ensemble_is_not_a_probability_mean():
    """Log-odds pooling is a different operator from averaging probabilities;
    asserting that keeps a future refactor from quietly swapping them."""
    ens = EnsembleDetector([
        ("a", _Fixed(0.95, "a"), 1.0),
        ("b", _Fixed(0.55, "b"), 1.0),
    ])
    p = ens.score_batch([Claim(text="a claim of some length")], EVIDENCE)[0]
    assert abs(p - (0.95 + 0.55) / 2) > 0.01


# --------------------------------------------------------------------------
# probe set integrity
# --------------------------------------------------------------------------


def test_probe_set_is_balanced_and_unambiguous():
    supported = [s for _, _, s in PROBE_SET]
    assert len(PROBE_SET) >= 8
    assert 0 < sum(supported) < len(supported)  # both classes present
    for doc, claim, _ in PROBE_SET:
        assert doc.strip() and claim.strip()


def test_every_template_candidate_has_both_slots():
    for t in TEMPLATE_CANDIDATES:
        assert "{doc}" in t and "{claim}" in t
        # Must not raise with real content.
        t.format(doc="d", claim="c")


def test_lexical_detector_separates_the_probe_set():
    """The cheap baseline must clear the probe, or the probe is broken."""
    from truthforge.eval.metrics import auroc

    det = LexicalOverlapDetector()
    probs = [
        det.score_batch([Claim(text=claim)], [Passage("p", doc)])[0]
        for doc, claim, _ in PROBE_SET
    ]
    assert auroc(probs, [s for _, _, s in PROBE_SET]) >= 0.85


# --------------------------------------------------------------------------
# build_detector wiring
# --------------------------------------------------------------------------


def test_build_detector_resolves_baselines_without_torch():
    from truthforge.detect.minicheck import build_detector

    cfg = Config()
    cfg.detector.name = "lexical"
    det = build_detector(cfg)
    assert det.score_batch([Claim(text="water boils at 100 degrees")], EVIDENCE)


def test_build_detector_rejects_unknown_names():
    from truthforge.detect.minicheck import build_detector

    cfg = Config()
    cfg.detector.name = "not_a_detector"
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector(cfg)


def test_build_detector_refuses_a_synthetic_scaler_on_real_data(tmp_path, monkeypatch):
    """A temperature fitted on the fixture must not silently be applied to a
    real run -- that would be a methodological error nothing else would catch."""
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "calibration_lexical.json").write_text(
        json.dumps({"temperature": 0.5, "bias": 0.0, "synthetic": True}), encoding="utf-8"
    )

    from truthforge.detect.minicheck import build_detector

    cfg = Config()
    cfg.detector.name = "lexical"

    cfg.dataset = "ragtruth"
    with pytest.warns(RuntimeWarning, match="synthetic fixture"):
        det = build_detector(cfg)
    assert not isinstance(det, CalibratedDetector)

    cfg.dataset = "ragtruth_sample"
    assert isinstance(build_detector(cfg), CalibratedDetector)


def test_calibration_can_be_explicitly_disabled(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "calibration_lexical.json").write_text(
        json.dumps({"temperature": 0.5, "bias": 0.0}), encoding="utf-8"
    )

    from truthforge.detect.minicheck import build_detector

    cfg = Config()
    cfg.detector.name = "lexical"
    cfg.detector.calibration_path = ""
    assert not isinstance(build_detector(cfg), CalibratedDetector)


# --------------------------------------------------------------------------
# hf compatibility shim
# --------------------------------------------------------------------------


def test_dtype_kwarg_matches_the_installed_transformers():
    from truthforge._hf import _dtype_kwarg

    assert _dtype_kwarg() in {"dtype", "torch_dtype"}


def test_with_dtype_fallback_swaps_the_keyword():
    from truthforge._hf import with_dtype_fallback

    def picky(name, *, torch_dtype=None):
        if torch_dtype is None:
            raise TypeError("got an unexpected keyword argument 'dtype'")
        return ("ok", torch_dtype)

    assert with_dtype_fallback(picky, "m", dtype="fp16") == ("ok", "fp16")


def test_with_dtype_fallback_reraises_unrelated_type_errors():
    from truthforge._hf import with_dtype_fallback

    def broken(name, **kw):
        raise TypeError("missing positional argument 'config'")

    with pytest.raises(TypeError, match="config"):
        with_dtype_fallback(broken, "m", dtype="fp16")


def test_cuda_summary_never_raises():
    from truthforge._hf import cuda_memory_summary

    s = cuda_memory_summary()
    assert "available" in s


def test_nli_detail_cites_the_most_specific_premise():
    """A whole passage usually entails marginally better than the one sentence
    inside it, so citing the top scorer shows the reader a wall of text when a
    single line carries the fact. Among near-equal premises, prefer the
    shortest."""
    from truthforge.detect.nli import NLIDetector, NLIScores

    det = NLIDetector.__new__(NLIDetector)       # no model load
    det.sentence_level = True
    det.max_premises = 64
    det.specificity_margin = 0.05
    det.flag_threshold = 0.5

    long_passage = ("Acme Corp Handbook. Section 4: Employees are entitled to 25 days "
                    "of paid annual leave. Section 5: Leave needs two weeks notice.")
    short_sentence = "Employees are entitled to 25 days of paid annual leave."

    class FakeBackend:
        def score_pairs(self, pairs):
            # the long passage edges out the short sentence, but only just
            return [
                NLIScores(entailment=0.99 if p is long_passage else 0.97,
                          neutral=0.0, contradiction=0.01)
                for p, _ in pairs
            ]

    det.backend = FakeBackend()
    det._premises = lambda passages: [long_passage, short_sentence]

    out = det.detail([Claim(text="Employees get 25 days of leave.")], [])
    assert out[0]["best_entailing_premise"] == short_sentence
    # the reported score is still the best one, not the cited premise's
    assert out[0]["entailment"] == 0.99


def test_nli_sentence_split_honours_line_breaks():
    """Uploaded documents are headings, numbered sections and bullets -- almost
    none of which end in a full stop. Splitting on terminal punctuation alone
    glued a whole section into one premise, so the citation shown to a reader
    was a block of text when a single line carried the fact."""
    from truthforge.detect.nli import _sentences

    doc = (
        "Acme Corp Employee Handbook\n"
        "Section 4: Annual Leave\n"
        "Employees are entitled to 25 days of paid annual leave per year.\n"
        "Section 5: Annual leave must be requested at least two weeks in advance.\n"
    )
    out = _sentences(doc)

    assert "Section 4: Annual Leave" in out
    assert "Employees are entitled to 25 days of paid annual leave per year." in out
    assert (
        "Section 5: Annual leave must be requested at least two weeks in advance."
        in out
    )
    # nothing spans a line break any more
    assert not any("\n" in s for s in out)


def test_nli_sentence_split_still_splits_within_a_line():
    from truthforge.detect.nli import _sentences

    out = _sentences(
        "The tower was completed in 1889. It stands 330 metres tall today."
    )
    assert out == [
        "The tower was completed in 1889.",
        "It stands 330 metres tall today.",
    ]
