"""Build the blind human-annotation workbook, and the answer key it joins on.

    python scripts/make_annotation_workbook.py
    python scripts/make_annotation_workbook.py --n-broke 30 --n-fine 20

**What this is for.** The ledger's over-correction column -- "fix-everything
broke 69 claims that were already correct, TruthForge broke 2" -- is the number
the project's damage claim rests on, and it is the one with the weakest label
source. It cannot be otherwise: RAGTruth's spans annotate the ORIGINAL
response, so no dataset can say whether a rewrite broke text that did not exist
when the dataset was labelled. The detector fills that gap at precision 0.184.
Fifty human labels turn that column from detector-judged into measured.

**Why only rewrites that changed something.** Of 879 correction attempts in the
always-correct arm, 741 returned the claim byte-identical. A human has nothing
to judge there and labelling them would burn the budget on non-events. The
population is the 138 + 84 attempts whose text actually changed.

**Why stratified, and what that costs.** Sampling 50 at random from 222 would
draw roughly 16 of the 71 the ledger calls broken -- too few to estimate that
column's precision. So the sample over-weights them. The consequence must
travel with any number computed from it: the label distribution in this sample
is NOT the population distribution. A worse-rate read straight off these
labels would be inflated. Reweight by stratum, or report it as "worse-rate
among sampled attempts" and state the sampling.

**Why blind.** The workbook carries no heuristic label, no arm name, no
stratum, and no detector probability. Showing any of them would anchor the
annotator to the thing the annotation exists to check, and the agreement
number would measure compliance rather than validity. The answer key is
written to a separate file that is not part of the worksheet.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthforge.cli import _turn_from_dict  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth  # noqa: E402
from truthforge.eval.comparison import claim_fates  # noqa: E402

LABELS = ["BETTER", "SAME", "INFO_LOSS", "HEDGING", "NEW_ERROR"]

LABEL_HELP = [
    ("BETTER", "The rewrite fixed a real error, and lost nothing the evidence supported."),
    ("SAME", "Nothing meaningful changed, or the change neither helped nor hurt."),
    ("INFO_LOSS", "The rewrite dropped a detail the evidence DID support."),
    ("HEDGING", "A clear statement became vague or qualified, without cause in the evidence."),
    ("NEW_ERROR", "The rewrite introduced a claim the evidence does not support."),
]


def collect(runs: list[str]) -> list[dict]:
    """Every rewrite whose text actually changed, with its ledger verdict."""
    exs = {
        e.example_id: e
        for e in load_ragtruth(root=str(ROOT / "data/raw/ragtruth"),
                               split="test", task_types={"QA"})
    }
    out = []
    for run in runs:
        path = ROOT / "runs" / run / "turns.jsonl"
        if not path.exists():
            print(f"  skipping {run}: no turns.jsonl")
            continue
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
            evidence = "\n\n".join(p.text for p in t.passages)
            for a in t.attempts:
                if a.before_text.strip() == a.after_text.strip():
                    continue          # nothing for a human to judge
                cell = cell_of.get(a.claim_id)
                if cell is None:
                    continue
                out.append({
                    "run": run,
                    "attempt_key": f"{a.claim_id}:{a.iteration}:{a.action.value}",
                    "claim_id": a.claim_id,
                    "example_id": t.example_id,
                    "question": t.question,
                    "evidence": evidence,
                    "before": a.before_text,
                    "after": a.after_text,
                    "ledger_cell": cell,
                    # The stratum. Never written to the worksheet.
                    #
                    # Only two cells are statements ABOUT THE REWRITE:
                    #   false_fix -- it broke a claim that was fine
                    #   true_keep -- it left a claim that was fine, fine
                    # Everything else describes a different decision. A claim
                    # whose fate is *_drop was ultimately abstained, so its
                    # cell is about the abstention; a false_keep was never
                    # resolved. Lumping those into the control group was the
                    # first version of this and it made "rewrites the ledger
                    # called harmless" a false description of 11 of 20
                    # controls. They are excluded, not relabelled.
                    "stratum": {
                        "false_fix": "damaged",
                        "true_keep": "control",
                    }.get(cell, "not_about_the_rewrite"),
                    "heuristic_outcome": a.outcome.value if a.outcome else "",
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+",
                    default=["rt300_always_correct", "rt300_rule"])
    ap.add_argument("--n-broke", type=int, default=30)
    ap.add_argument("--n-fine", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default="annotation/worksheet.xlsx")
    ap.add_argument("--key", default="annotation/answer_key.json")
    ap.add_argument("--evidence-chars", type=int, default=2200)
    args = ap.parse_args()

    print("collecting rewrites that changed the text...")
    rows = collect(args.runs)
    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    for k, v in sorted(by_stratum.items()):
        print(f"  {k:6s} {len(v)}")

    rng = random.Random(args.seed)
    damaged_pool = by_stratum.get("damaged", [])
    control_pool = by_stratum.get("control", [])
    skipped = len(by_stratum.get("not_about_the_rewrite", []))
    if skipped:
        print(f"  (excluded {skipped} whose ledger cell describes an "
              f"abstention, not the rewrite)")

    sample = (
        rng.sample(damaged_pool, min(args.n_broke, len(damaged_pool)))
        + rng.sample(control_pool, min(args.n_fine, len(control_pool)))
    )
    # Interleave, so a run of consecutive "broke" items cannot be inferred.
    rng.shuffle(sample)
    for i, r in enumerate(sample, 1):
        r["item"] = f"A{i:03d}"
    print(f"\nsampled {len(sample)} items "
          f"({sum(1 for r in sample if r['stratum'] == 'damaged')} the ledger "
          f"calls damaging, {sum(1 for r in sample if r['stratum'] == 'control')} "
          f"it calls harmless)")

    write_workbook(sample, ROOT / args.out, args.evidence_chars)

    key_path = ROOT / args.key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({
        "seed": args.seed,
        "runs": args.runs,
        "sampling": "stratified; NOT the population distribution -- reweight "
                    "before quoting a worse-rate",
        "population_sizes": {k: len(v) for k, v in by_stratum.items()},
        "stratum_meaning": {
            "damaged": "ledger false_fix -- the rewrite broke a claim that was fine",
            "control": "ledger true_keep -- the rewrite left a fine claim fine",
            "not_about_the_rewrite": "claim was abstained or left unresolved, so "
                                     "its cell describes a different decision; "
                                     "excluded from sampling",
        },
        "items": {
            r["item"]: {
                "run": r["run"],
                "attempt_key": r["attempt_key"],
                "claim_id": r["claim_id"],
                "stratum": r["stratum"],
                "ledger_cell": r["ledger_cell"],
                "heuristic_outcome": r["heuristic_outcome"],
            }
            for r in sample
        },
    }, indent=2), encoding="utf-8")
    print(f"wrote {key_path}  (the answer key -- do NOT open before labelling)")
    return 0


def write_workbook(sample: list[dict], path: Path, evidence_chars: int) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # ---------------------------------------------------------- instructions
    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 18
    ins.column_dimensions["B"].width = 104

    def line(a, b="", bold=False, size=11, fill=None):
        r = ins.max_row + 1
        ins[f"A{r}"] = a
        ins[f"B{r}"] = b
        ins[f"A{r}"].font = Font(name="Arial", size=size, bold=True)
        ins[f"B{r}"].font = Font(name="Arial", size=size, bold=bold)
        ins[f"B{r}"].alignment = Alignment(wrap_text=True, vertical="top")
        if fill:
            ins[f"A{r}"].fill = ins[f"B{r}"].fill = PatternFill(
                "solid", fgColor=fill)
        return r

    ins["A1"] = "How to label"
    ins["A1"].font = Font(name="Arial", size=16, bold=True)
    line("")
    line("The task",
         "Each row on the 'Worksheet' tab shows a sentence before and after an "
         "automatic correction. Decide what the correction did. Fill in the "
         "LABEL column from the dropdown. About 30 seconds per row.")
    line("")
    line("THE ONE RULE",
         "Judge ONLY against the EVIDENCE column. Ignore what you know about "
         "the world. If the evidence does not support something, it is "
         "unsupported even when it is true in reality.",
         bold=True, fill="FFF2CC")
    line("")
    line("The labels", "")
    for name, desc in LABEL_HELP:
        line(f"   {name}", desc)
    line("")
    line("If unsure",
         "Pick the closest label and say why in NOTES. 'I could not tell' in "
         "NOTES is a useful answer; a guessed label with no note is not.")
    line("Order",
         "Rows are deliberately shuffled and carry no hint of what the system "
         "thought. That is the point: this is checking the system's judgement, "
         "so seeing it first would defeat the exercise.")
    line("")
    line("When done", "Save the file and send it back. Nothing else to do.")
    line("")
    line("Example", "")
    ex = ins.max_row + 1
    ins[f"A{ex}"] = "   BEFORE"
    ins[f"B{ex}"] = "The tower was completed in 1889 and stands 330 metres tall."
    ins[f"A{ex+1}"] = "   AFTER"
    ins[f"B{ex+1}"] = "The tower was completed in 1889."
    ins[f"A{ex+2}"] = "   EVIDENCE"
    ins[f"B{ex+2}"] = "The Eiffel Tower opened in 1889. It stands 330 metres tall."
    ins[f"A{ex+3}"] = "   LABEL"
    ins[f"B{ex+3}"] = ("INFO_LOSS  — the height was supported by the evidence "
                       "and the rewrite dropped it.")
    for r in range(ex, ex + 4):
        ins[f"A{r}"].font = Font(name="Arial", size=11, bold=True)
        ins[f"B{r}"].font = Font(name="Arial", size=11,
                                 bold=(r == ex + 3), italic=(r != ex + 3))
        ins[f"B{r}"].alignment = Alignment(wrap_text=True, vertical="top")

    # ------------------------------------------------------------- worksheet
    ws = wb.create_sheet("Worksheet")
    headers = ["ITEM", "QUESTION", "EVIDENCE", "BEFORE", "AFTER", "LABEL", "NOTES"]
    widths = [8, 34, 70, 44, 44, 16, 30]
    ws.append(headers)
    for i, (h, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(row=1, column=i)
        c.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E79")
        c.alignment = Alignment(vertical="center", horizontal="left")
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    for r in sample:
        ws.append([
            r["item"], r["question"], r["evidence"][:evidence_chars],
            r["before"], r["after"], "", "",
        ])

    body = Font(name="Arial", size=10)
    wrap = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=7):
        for c in row:
            c.font = body
            c.alignment = wrap
        row[5].fill = PatternFill("solid", fgColor="FFF2CC")   # LABEL
        row[5].alignment = Alignment(vertical="center", horizontal="center")
        row[5].font = Font(name="Arial", size=10, bold=True)
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 150

    dv = DataValidation(
        type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True,
        # OOXML inverts this one: showDropDown=0 means the in-cell arrow IS
        # shown. The other two are not inverted, and openpyxl defaults them to
        # off -- so without them the error box and the hint never appear and a
        # typo is accepted silently, which is the one thing validation is for.
        showDropDown=False,
        showErrorMessage=True,
        showInputMessage=True,
    )
    dv.error = "Pick one of: " + ", ".join(LABELS)
    dv.errorTitle = "Not a valid label"
    dv.prompt = "Judge against the EVIDENCE column only."
    dv.promptTitle = "Label"
    ws.add_data_validation(dv)
    dv.add(f"F2:F{ws.max_row}")

    # Progress counter, so the annotator can see how far through they are.
    prog = wb.create_sheet("Progress")
    prog["A1"] = "Labelled"
    prog["B1"] = f"=COUNTA(Worksheet!F2:F{ws.max_row})"
    prog["A2"] = "Total"
    prog["B2"] = ws.max_row - 1
    prog["A3"] = "Remaining"
    prog["B3"] = "=B2-B1"
    for r in range(1, 4):
        prog[f"A{r}"].font = Font(name="Arial", size=11, bold=True)
        prog[f"B{r}"].font = Font(name="Arial", size=11)
    prog.column_dimensions["A"].width = 14
    prog.column_dimensions["B"].width = 10

    wb.save(path)
    print(f"wrote {path}  ({ws.max_row - 1} items)")


if __name__ == "__main__":
    raise SystemExit(main())
