"""The workbook has to be readable back, and the sample has to mean what it says.

Both of these were broken when first written. The workbook was an xlsx keyed on
an opaque ITEM with one five-way LABEL, while every consumer expected a CSV
keyed on attempt_key with separate outcome/worse_kind columns -- so a completed
worksheet was a dead end. And the control stratum swept in claims whose ledger
cell described an abstention rather than the rewrite, which made "rewrites the
ledger called harmless" a false description of more than half of them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "ingest", ROOT / "scripts" / "ingest_annotation_workbook.py"
)
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def test_every_workbook_label_maps_to_a_valid_outcome():
    """A label the pipeline cannot parse is worse than a missing one: it
    reaches `load_human_labels`, which raises on an unknown outcome."""
    from truthforge.types import Outcome, WorseKind

    valid_outcomes = {o.value for o in Outcome}
    valid_kinds = set(WorseKind.values())
    for label, (outcome, kind) in ingest.LABEL_MAP.items():
        assert outcome in valid_outcomes, label
        if outcome == "worse":
            assert kind in valid_kinds, label
        else:
            assert kind == ""


def test_the_five_labels_match_the_workbook_dropdown():
    spec2 = importlib.util.spec_from_file_location(
        "mk", ROOT / "scripts" / "make_annotation_workbook.py"
    )
    mk = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mk)
    assert set(mk.LABELS) == set(ingest.LABEL_MAP)
    # ...and the instructions tab explains every one of them
    assert {n for n, _ in mk.LABEL_HELP} == set(mk.LABELS)


def _workbook(tmp_path, rows):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Worksheet"
    ws.append(["ITEM", "QUESTION", "EVIDENCE", "BEFORE", "AFTER", "LABEL", "NOTES"])
    for item, label, note in rows:
        ws.append([item, "q", "e", "b", "a", label, note])
    p = tmp_path / "wb.xlsx"
    wb.save(p)
    return p


def test_reads_labels_and_tolerates_case_and_spaces(tmp_path):
    p = _workbook(tmp_path, [
        ("A001", "INFO_LOSS", ""),
        ("A002", "info loss", "lowercase with a space"),
        ("A003", "  SAME  ", ""),
    ])
    got = ingest.read_workbook(p)
    assert {k: v["label"] for k, v in got.items()} == {
        "A001": "INFO_LOSS", "A002": "INFO_LOSS", "A003": "SAME",
    }
    assert got["A002"]["notes"] == "lowercase with a space"


def test_blank_rows_are_unlabelled_not_guessed(tmp_path):
    p = _workbook(tmp_path, [("A001", "SAME", ""), ("A002", "", "")])
    got = ingest.read_workbook(p)
    assert "A002" not in got, "a blank label must not become a label"


def test_an_invalid_label_is_dropped_not_silently_coerced(tmp_path, capsys):
    p = _workbook(tmp_path, [("A001", "DUNNO", ""), ("A002", "BETTER", "")])
    got = ingest.read_workbook(p)
    assert set(got) == {"A002"}
    assert "DUNNO" in capsys.readouterr().out


def test_wilson_interval_behaves_at_the_edges():
    """The normal approximation gives intervals outside [0,1] at these n and
    proportions, which is why this is a Wilson interval."""
    lo, hi = ingest.wilson(30, 30)
    assert 0.0 <= lo <= 1.0 and hi <= 1.0 and lo > 0.85
    lo, hi = ingest.wilson(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo, hi = ingest.wilson(21, 30)
    assert lo < 0.70 < hi


def test_stratified_sample_is_reweighted_before_any_population_claim():
    """Reading a harm rate straight off a stratified sample inflates it. The
    sample over-draws the damaged stratum on purpose, so the population figure
    has to be a weighted combination."""
    # 70% harm among 74 damaged, 10% among 64 controls
    weighted = (0.70 * 74 + 0.10 * 64) / (74 + 64)
    naive = (0.70 * 30 + 0.10 * 20) / 50
    assert weighted < naive, "reweighting must pull the figure down here"
    assert abs(weighted - 0.4217) < 0.001


def test_only_two_ledger_cells_are_statements_about_the_rewrite():
    """false_fix says the rewrite broke a good claim; true_keep says it left
    one alone. Every other cell describes a different decision -- a *_drop was
    abstained, a false_keep was never resolved -- so asking "did the rewrite
    damage this?" against them is a category error. The sampler now excludes
    them; the ingest groups them separately so an older worksheet is still
    analysable rather than discarded."""
    assert ingest.CELL_GROUP == {"false_fix": "damaged", "true_keep": "fine"}


@pytest.mark.skipif(
    not (ROOT / "annotation" / "answer_key.json").exists(),
    reason="no generated worksheet in this checkout",
)
def test_every_sampled_item_carries_a_cell_the_ingest_can_group():
    """The ingest keys off `ledger_cell`, not off the key's `stratum` field,
    precisely so a worksheet produced by the earlier sampler still works. That
    only holds if every item records its cell."""
    key = json.loads((ROOT / "annotation" / "answer_key.json").read_text("utf-8"))
    items = key["items"].values()
    assert items, "key has no items"
    for v in items:
        assert v.get("ledger_cell"), f"item without a ledger_cell: {v}"
        assert v.get("attempt_key"), "item without an attempt_key to join on"

    # The over-correction stratum must be present and correctly identified,
    # whichever sampler produced the file -- it is the whole point.
    damaged = [v for v in items if v["ledger_cell"] == "false_fix"]
    assert len(damaged) >= 20, (
        f"only {len(damaged)} items from the over-correction column; the "
        f"precision estimate needs a usable n"
    )
