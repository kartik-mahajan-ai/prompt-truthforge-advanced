"""Read the completed annotation workbook and report what it establishes.

    python scripts/ingest_annotation_workbook.py annotation/worksheet.xlsx
    python scripts/ingest_annotation_workbook.py annotation/worksheet.xlsx --csv annotation/labels_human.csv

**Why this exists.** The workbook was built before anything could read it back.
`load_human_labels` expects a CSV keyed on `attempt_key` with separate
`outcome` and `worse_kind` columns; the workbook is an xlsx keyed on an opaque
`ITEM` with one five-way `LABEL`. Filling it in would have produced a file no
part of the pipeline could consume. This closes that loop: it joins ITEM back
to attempt_key through the answer key, splits LABEL into the two columns the
rest of the code expects, and emits the CSV.

**What it then reports, and the one correction that matters.** The sample is
stratified -- 30 drawn from the 74 rewrites the ledger calls damaging, 20 from
the 145 it calls harmless -- so the label mix here is NOT the population mix.
A worse-rate read straight off these answers would be inflated by roughly the
ratio of those sampling fractions. Every population-level figure below is
reweighted by stratum, and the per-stratum figures are printed beside it so
the correction is visible rather than taken on trust.

The headline it produces is the precision of the ledger's over-correction
column: of the rewrites it called damaging, what share does a human agree
were. That is the number the project's damage claim rests on and the only one
these fifty labels can settle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# LABEL in the workbook -> (outcome, worse_kind) the rest of the code uses.
LABEL_MAP = {
    "BETTER": ("better", ""),
    "SAME": ("same", ""),
    "INFO_LOSS": ("worse", "info_loss"),
    "HEDGING": ("worse", "hedging"),
    "NEW_ERROR": ("worse", "new_error"),
}


# Only two ledger cells are statements ABOUT THE REWRITE. Everything else
# describes a different decision: a *_drop claim was ultimately abstained, so
# its cell is about the abstention; a false_keep was never resolved. Asking
# "did the rewrite damage this?" against those is a category error, so they
# are reported in their own bucket rather than counted as controls.
CELL_GROUP = {
    "false_fix": "damaged",      # the over-correction column itself
    "true_keep": "fine",         # the clean control
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used because the normal approximation is wrong
    at the small n and near-0/near-1 proportions this sample produces."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_workbook(path: Path) -> dict[str, dict[str, str]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    if "Worksheet" not in wb.sheetnames:
        raise SystemExit(f"{path}: no 'Worksheet' tab (tabs: {wb.sheetnames})")
    ws = wb["Worksheet"]

    header = [str(c.value or "").strip().upper() for c in ws[1]]
    try:
        i_item, i_label = header.index("ITEM"), header.index("LABEL")
    except ValueError:
        raise SystemExit(f"{path}: expected ITEM and LABEL columns, found {header}")
    i_notes = header.index("NOTES") if "NOTES" in header else None

    out, bad = {}, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        item = str(row[i_item] or "").strip()
        if not item:
            continue
        label = str(row[i_label] or "").strip().upper().replace(" ", "_")
        if not label:
            continue                      # not yet labelled
        if label not in LABEL_MAP:
            bad.append((item, label))
            continue
        out[item] = {
            "label": label,
            "notes": str(row[i_notes] or "").strip() if i_notes is not None else "",
        }
    if bad:
        print("UNRECOGNISED LABELS (left out):")
        for item, label in bad:
            print(f"  {item}: {label!r}")
        print(f"  valid labels: {', '.join(LABEL_MAP)}\n")
    return out


def population_by_cell(runs: list[str]) -> dict[str, int]:
    """How many text-changing rewrites fall in each ledger cell, per run set.

    Recovers what an older answer key failed to record. Returns {} if the runs
    are not on disk, so a missing run degrades to "no population figure"
    rather than to a wrong one.
    """
    try:
        from truthforge.cli import _turn_from_dict
        from truthforge.data.ragtruth import load_ragtruth
        from truthforge.eval.comparison import claim_fates
    except Exception:  # noqa: BLE001
        return {}

    try:
        exs = {
            e.example_id: e
            for e in load_ragtruth(root=str(ROOT / "data/raw/ragtruth"),
                                   split="test", task_types={"QA"})
        }
    except Exception:  # noqa: BLE001
        return {}

    counts: Counter = Counter()
    for run in runs:
        path = ROOT / "runs" / run / "turns.jsonl"
        if not path.exists():
            return {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = _turn_from_dict(json.loads(line))
            ex = exs.get(t.example_id)
            if ex is None:
                continue
            gold = {c.claim_id: not ex.overlaps_span(c.start, c.end)
                    for c in t.claims if c.start is not None}
            cell_of = {f.claim_id: f.cell.value for f in claim_fates(t, gold)}
            for a in t.attempts:
                if a.before_text.strip() == a.after_text.strip():
                    continue
                cell = cell_of.get(a.claim_id)
                if cell:
                    counts[cell] += 1
    return dict(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workbook", nargs="?", default="annotation/worksheet.xlsx")
    ap.add_argument("--key", default="annotation/answer_key.json")
    ap.add_argument("--csv", default="annotation/labels_human.csv")
    ap.add_argument("--annotator", default="human1")
    ap.add_argument("--json", default="annotation/human_agreement.json")
    args = ap.parse_args()

    wb_path = ROOT / args.workbook if not Path(args.workbook).is_absolute() else Path(args.workbook)
    key_path = ROOT / args.key if not Path(args.key).is_absolute() else Path(args.key)
    if not wb_path.exists():
        raise SystemExit(f"no workbook at {wb_path}")
    if not key_path.exists():
        raise SystemExit(f"no answer key at {key_path} -- regenerate the workbook")

    key = json.loads(key_path.read_text(encoding="utf-8"))
    items = key["items"]
    labels = read_workbook(wb_path)

    missing = [i for i in items if i not in labels]
    print(f"{len(labels)} of {len(items)} items labelled")
    if missing:
        print(f"  not yet done: {', '.join(missing[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")
    unknown = [i for i in labels if i not in items]
    if unknown:
        print(f"  WARNING: {len(unknown)} item id(s) not in the key -- "
              f"was the workbook regenerated after labelling? {unknown[:5]}")
    if not labels:
        print("\nNothing to analyse yet.")
        return 0

    # ---- CSV for the rest of the pipeline -------------------------------
    out_csv = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["attempt_key", "outcome", "worse_kind", "annotator_id", "notes"])
        for item, rec in sorted(labels.items()):
            meta = items.get(item)
            if meta is None:
                continue
            outcome, kind = LABEL_MAP[rec["label"]]
            w.writerow([meta["attempt_key"], outcome, kind, args.annotator,
                        rec["notes"]])
    print(f"wrote {out_csv}")

    # ---- the question these labels exist to answer ----------------------
    pop = key["population_sizes"]
    # Group by the LEDGER CELL, not by the key's stratum label. Two reasons.
    #
    # First, only two cells are statements about the rewrite: false_fix (it
    # broke a claim that was fine) and true_keep (it left a fine claim fine).
    # A claim whose fate is *_drop was ultimately abstained, so its cell
    # describes the abstention; a false_keep was never resolved. An earlier
    # version of the sampler swept all of those into the control group, which
    # made "rewrites the ledger called harmless" a false description of more
    # than half of them.
    #
    # Second, grouping here rather than trusting the stratum field means a
    # worksheet generated by that earlier sampler still analyses correctly:
    # its mislabelled items land in their own bucket and are reported for what
    # they are, instead of being discarded or silently miscounted. Nobody has
    # to re-label anything.
    strata: dict[str, list[tuple[str, str]]] = {"damaged": [], "fine": [],
                                                "other": []}
    other_cells: Counter = Counter()
    for item, rec in labels.items():
        meta = items.get(item)
        if meta is None:
            continue
        cell = meta.get("ledger_cell", "")
        bucket = CELL_GROUP.get(cell, "other")
        if bucket == "other":
            other_cells[cell] += 1
        strata[bucket].append((item, rec["label"]))

    # Population sizes for reweighting. Prefer the exact per-cell counts when
    # the key carries them; fall back to the stratum names a newer key uses.
    n_damaged_pop = pop.get("damaged", pop.get("broke", 0))
    n_fine_pop = pop.get("control", 0)
    if not n_fine_pop:
        # Older key: "fine" pooled true_keep with cells that are not about the
        # rewrite, so its count cannot serve as the control population. Rather
        # than refuse to reweight, recount from the runs the key names -- the
        # information is sitting in them and costs one pass to recover.
        counts = population_by_cell(key.get("runs", []))
        if counts:
            n_damaged_pop = counts.get("false_fix", n_damaged_pop)
            n_fine_pop = counts.get("true_keep", 0)
            print(f"  (recounted population from the runs: "
                  f"{n_damaged_pop} false_fix, {n_fine_pop} true_keep)")

    print("\n" + "=" * 74)
    print("DID THE LEDGER CALL IT RIGHT?")
    print("=" * 74)

    confirmed = {}
    for bucket in ("damaged", "fine", "other"):
        rows = strata.get(bucket) or []
        if not rows:
            continue
        counts = Counter(lbl for _, lbl in rows)
        # "damage" as a human sees it: the rewrite lost, hedged or invented.
        harmed = sum(counts[k] for k in ("INFO_LOSS", "HEDGING", "NEW_ERROR"))
        lo, hi = wilson(harmed, len(rows))
        confirmed[bucket] = {"n": len(rows), "harmed": harmed,
                             "rate": harmed / len(rows), "ci": [lo, hi],
                             "labels": dict(counts)}
        name = {
            "damaged": "rewrites the ledger called DAMAGING (false_fix)",
            "fine": "rewrites the ledger called HARMLESS (true_keep)",
            "other": "claims later ABSTAINED -- cell is not about the rewrite",
        }[bucket]
        print(f"\n  {name}  (n={len(rows)})")
        print(f"    human says harmed : {harmed}/{len(rows)} = {harmed/len(rows):.1%} "
              f"  95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"    labels            : "
              + ", ".join(f"{k} {v}" for k, v in counts.most_common()))

    d = confirmed.get("damaged")
    if d:
        print("\n" + "-" * 74)
        print(f"  PRECISION of the over-correction column: {d['rate']:.1%} "
              f"[{d['ci'][0]:.1%}, {d['ci'][1]:.1%}]")
        print(f"  The ledger reports {n_damaged_pop} damaging rewrites across both")
        print(f"  arms. At this precision the true figure is about "
              f"{round(n_damaged_pop * d['rate'])} "
              f"(range {round(n_damaged_pop * d['ci'][0])}"
              f"-{round(n_damaged_pop * d['ci'][1])}).")

    o = confirmed.get("other")
    if o:
        print("\n" + "-" * 74)
        print(f"  {o['n']} sampled items are NOT statements about the rewrite")
        print("  (ledger cells: "
              + ", ".join(f"{k} {v}" for k, v in other_cells.most_common()) + ")")
        print("  Those claims were abstained or left unresolved, so the ledger")
        print("  cell describes a different decision. Their labels are")
        print(f"  reported here -- {o['harmed']}/{o['n']} judged harmful -- but")
        print("  they are NOT evidence about the over-correction column and are")
        print("  excluded from the reweighted figure below.")

    f = confirmed.get("fine")
    if f and f["rate"] > 0 and n_fine_pop:
        print(f"\n  The ledger also MISSED damage: {f['harmed']}/{f['n']} rewrites it")
        print(f"  called harmless were judged harmful. Scaled to the {n_fine_pop}")
        print(f"  harmless rewrites that is about {round(n_fine_pop * f['rate'])} more.")

    # ---- reweighted population rate -------------------------------------
    if d and f and n_damaged_pop and n_fine_pop and f["n"]:
        total = n_damaged_pop + n_fine_pop
        weighted = (d["rate"] * n_damaged_pop + f["rate"] * n_fine_pop) / total
        naive = sum(c["harmed"] for c in confirmed.values()) / sum(
            c["n"] for c in confirmed.values())
        print("\n" + "-" * 74)
        print(f"  Harm rate among text-changing rewrites, REWEIGHTED: {weighted:.1%}")
        print(f"  The same number read straight off the sample: {naive:.1%} -- which")
        print(f"  is wrong, because the sample deliberately over-draws damaging")
        print(f"  rewrites. Quote the reweighted figure, and say the sampling.")
    elif d:
        print("\n" + "-" * 74)
        print("  No population-level harm rate is reported: this worksheet's")
        print("  control group cannot be sized, because the key pools true_keep")
        print("  with cells that describe abstentions rather than rewrites.")
        print("  The precision figure above is unaffected -- it only uses the")
        print("  false_fix items, which are correctly identified either way.")

    # ---- agreement with the heuristic -----------------------------------
    pairs = [
        (items[i]["heuristic_outcome"], LABEL_MAP[r["label"]][0])
        for i, r in labels.items()
        if i in items and items[i].get("heuristic_outcome")
    ]
    if pairs:
        from truthforge.eval.outcomes import cohens_kappa

        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        k = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
        print("\n" + "=" * 74)
        print("HEURISTIC CLASSIFIER vs HUMAN")
        print("=" * 74)
        print(f"  n={len(pairs)}  raw agreement {agree:.1%}  "
              f"kappa {'n/a' if k != k else f'{k:.3f}'}")
        print("  This is the validity anchor for the damage taxonomy. Report it")
        print("  with its n attached, and never tune the heuristic on these labels")
        print("  and then quote this number.")

    out_json = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
    out_json.write_text(json.dumps({
        "workbook": str(wb_path),
        "n_labelled": len(labels),
        "n_items": len(items),
        "annotator": args.annotator,
        "sampling": key.get("sampling"),
        "population_sizes": pop,
        "by_stratum": confirmed,
        "heuristic_vs_human": (
            {"n": len(pairs), "raw_agreement": round(agree, 4),
             "kappa": None if k != k else round(k, 4)} if pairs else None
        ),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
