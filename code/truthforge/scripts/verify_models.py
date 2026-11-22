"""Health check for the model layer.

    python scripts/verify_models.py                 # detectors only (fast)
    python scripts/verify_models.py --all           # + generator
    python scripts/verify_models.py --json runs/model_report.json

Loads each model, reports VRAM before and after, and runs a probe set through
it. This is the thing to run after changing a model, a dtype, or a transformers
version -- every failure mode it catches (wrong label polarity, wrong input
template, silent CPU fallback, OOM) produces numbers that look fine and mean
nothing.

Exit code is non-zero if any component fails its probe, so it works in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge._hf import cuda_memory_summary, empty_cache  # noqa: E402
from truthforge.detect.minicheck import PROBE_SET  # noqa: E402
from truthforge.types import Claim, Passage  # noqa: E402

PROBE_PASSAGE = Passage(
    "probe",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
    "It stands 330 metres tall including its antennas.",
)
PROBE_CLAIMS = [
    ("The Eiffel Tower was completed in 1889.", True),
    ("The Eiffel Tower stands 330 metres tall.", True),
    ("The Eiffel Tower was completed in 1923.", False),
    ("The Eiffel Tower cost 45 million francs to build.", False),
]


def banner(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def probe_detector(det, name: str) -> dict:
    """Run the shared probe and report separation."""
    from truthforge.eval.metrics import auroc

    claims = [Claim(text=t) for t, _ in PROBE_CLAIMS]
    gold = [g for _, g in PROBE_CLAIMS]

    t0 = time.perf_counter()
    verdicts = det.check(claims, [PROBE_PASSAGE])
    elapsed = time.perf_counter() - t0

    probs = [v.support_prob for v in verdicts]
    sep = auroc(probs, gold)
    acc = sum(1 for p, g in zip(probs, gold) if (p >= 0.5) == g) / len(gold)

    print(f"  {'claim':<52} {'P(sup)':>8}  gold")
    for (text, g), p in zip(PROBE_CLAIMS, probs):
        mark = "ok " if (p >= 0.5) == g else "MISS"
        print(f"  {mark} {text[:48]:<48} {p:>8.3f}  {'sup' if g else 'uns'}")

    print(f"\n  AUROC {sep:.3f} | accuracy {acc:.2f} | {elapsed:.2f}s "
          f"({elapsed / len(claims) * 1000:.0f} ms/claim)")

    from truthforge.detect.calibration import spread

    sp = spread(probs)
    print(f"  score spread: {sp['min']:.3f}-{sp['max']:.3f}, "
          f"{sp['frac_saturated']:.0%} saturated (<0.05 or >0.95)")
    if sp["frac_saturated"] >= 0.75:
        print("  NOTE: heavily saturated scores. The router's uncertain band "
              "will be nearly empty -- calibrate before trusting thresholds.")

    # Pass/fail on RANKING (AUROC), not on accuracy at a fixed 0.5 threshold.
    # A detector can rank every claim perfectly while its scores all sit above
    # 0.5 -- that is a calibration defect, fixed by TemperatureScaler, not a
    # broken model. Gating on accuracy here would reject a working judge.
    ranks = sep == sep and sep >= 0.9
    miscalibrated = ranks and acc < 0.75
    if miscalibrated:
        print("  NOTE: ranks correctly (AUROC high) but accuracy at the 0.5 "
              "threshold is poor -- this detector NEEDS calibration before "
              "its scores feed the router. Run scripts/calibrate_detector.py.")

    return {
        "name": name,
        "auroc": None if sep != sep else round(sep, 4),
        "accuracy": round(acc, 3),
        "seconds": round(elapsed, 3),
        "probs": [round(p, 4) for p in probs],
        "spread": sp,
        "needs_calibration": miscalibrated or sp["frac_saturated"] >= 0.75,
        "passed": bool(ranks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also load the generator")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip", nargs="*", default=[], help="component names to skip")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    report: dict = {"components": [], "cuda": cuda_memory_summary()}

    banner("ENVIRONMENT")
    mem = report["cuda"]
    if mem.get("available"):
        print(f"  GPU        {mem['device_name']}")
        print(f"  VRAM       {mem['free_mb']} MB free / {mem['total_mb']} MB total")
    else:
        print(f"  no CUDA: {mem.get('reason')}")
        print("  everything will run on CPU -- correctness holds, latency "
              "numbers do not")
    try:
        import torch
        import transformers

        print(f"  torch      {torch.__version__}")
        print(f"  transformers {transformers.__version__}")
        report["versions"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    except ImportError as exc:
        print(f"  MISSING: {exc}")
        return 1

    ok = True

    # -- lexical baseline (no model) ------------------------------------
    if "lexical" not in args.skip:
        banner("BASELINE: lexical overlap (no model, no GPU)")
        from truthforge.detect.baselines import LexicalOverlapDetector

        r = probe_detector(LexicalOverlapDetector(), "lexical")
        report["components"].append(r)
        ok &= r["passed"]

    # -- NLI -------------------------------------------------------------
    if "nli" not in args.skip:
        banner("NLI: DeBERTa-v3-base-mnli-fever-anli")
        try:
            from truthforge.detect.nli import ContradictionAnalyzer, NLIDetector

            det = NLIDetector(device=args.device)
            det.backend.ensure_loaded()
            print(f"  loaded on {det.backend.device}, "
                  f"label map {det.backend._idx}")
            print(f"  VRAM after load: {cuda_memory_summary().get('used_mb')} MB used")
            r = probe_detector(det, "nli")
            report["components"].append(r)
            ok &= r["passed"]

            print("\n  contradiction analyzer (sources disagreeing with each other):")
            analyzer = ContradictionAnalyzer(device=args.device)
            conflicting = [
                Passage("a", "The Amazon River is approximately 6,400 kilometres long."),
                Passage("b", "The Amazon River measures about 7,000 kilometres in length."),
            ]
            agreeing = [
                Passage("a", "The Amazon River is approximately 6,400 kilometres long."),
                Passage("b", "The Amazon is roughly 6,400 km from source to mouth."),
            ]
            claim = "The Amazon River is about 6,400 kilometres long."
            c_conf = analyzer.analyze(claim, conflicting)
            c_agree = analyzer.analyze(claim, agreeing)
            print(f"    conflicting sources -> {c_conf['contradiction']:.3f}")
            print(f"    agreeing sources    -> {c_agree['contradiction']:.3f}")
            discriminates = c_conf["contradiction"] > c_agree["contradiction"]
            print(f"    discriminates: {discriminates}")
            report["contradiction"] = {
                "conflicting": round(c_conf["contradiction"], 4),
                "agreeing": round(c_agree["contradiction"], 4),
                "discriminates": discriminates,
            }
            ok &= discriminates
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            report["components"].append({"name": "nli", "passed": False, "error": str(exc)})
            ok = False

    # -- MiniCheck --------------------------------------------------------
    if "minicheck" not in args.skip:
        banner("MINICHECK: Flan-T5-Large (with empirical template verification)")
        try:
            from truthforge.detect.minicheck import MiniCheckDetector

            det = MiniCheckDetector(device=args.device)
            t0 = time.perf_counter()
            det._ensure_loaded()
            print(f"  loaded in {time.perf_counter() - t0:.1f}s "
                  f"| backend={det.backend} | device={det.device}")

            rep = det.template_report
            if rep.get("source") == "empirical_probe":
                print(f"\n  template search over {len(PROBE_SET)} probe pairs:")
                for cand in rep["all_candidates"]:
                    star = "->" if cand["template"] == rep["chosen"] else "  "
                    print(f"   {star} AUROC {cand['auroc']:.3f}  {cand['template'][:52]!r}")
                print(f"\n  chosen: {rep['chosen']!r}")
                print(f"  probe AUROC {rep['probe_auroc']:.3f} | "
                      f"accuracy {rep['probe_accuracy']:.2f}")
            else:
                print(f"  template source: {rep.get('source')}")

            print(f"  VRAM after load: {cuda_memory_summary().get('used_mb')} MB used")
            r = probe_detector(det, "minicheck")
            r["template_report"] = rep
            report["components"].append(r)
            ok &= r["passed"]
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            report["components"].append(
                {"name": "minicheck", "passed": False, "error": str(exc)}
            )
            ok = False

    # -- generator --------------------------------------------------------
    if args.all and "generator" not in args.skip:
        banner("GENERATOR / CORRECTOR")
        try:
            empty_cache()
            from truthforge.config import GeneratorConfig
            from truthforge.correct.rewrite import Rewriter
            from truthforge.models import LocalHFGenerator

            gen = LocalHFGenerator(GeneratorConfig(device=args.device))
            t0 = time.perf_counter()
            gen._ensure_loaded()
            print(f"  {gen.cfg.model} loaded in {time.perf_counter() - t0:.1f}s "
                  f"on {gen.device}")
            print(f"  VRAM after load: {cuda_memory_summary().get('used_mb')} MB used")

            rw = Rewriter(gen)
            claim = Claim(text="The Eiffel Tower was completed in 1923.")
            t0 = time.perf_counter()
            res = rw.rewrite(claim, claim.text, [PROBE_PASSAGE])
            dt = time.perf_counter() - t0
            print(f"\n  before: {res.original}")
            print(f"  after : {res.rewritten}")
            print(f"  {dt:.1f}s | {res.cost.total_tokens} tokens | "
                  f"changed={res.changed}")
            passed = res.changed and "1889" in res.rewritten
            print(f"  corrected the date: {passed}")
            report["components"].append(
                {
                    "name": "generator",
                    "model": gen.cfg.model,
                    "passed": passed,
                    "seconds": round(dt, 2),
                    "before": res.original,
                    "after": res.rewritten,
                }
            )
            ok &= passed
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            report["components"].append(
                {"name": "generator", "passed": False, "error": str(exc)}
            )
            ok = False

    banner("SUMMARY")
    for c in report["components"]:
        status = "PASS" if c.get("passed") else "FAIL"
        extra = f" (AUROC {c['auroc']})" if c.get("auroc") is not None else ""
        cal = "  [needs calibration]" if c.get("needs_calibration") else ""
        print(f"  {status}  {c['name']}{extra}{cal}")
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")

    report["overall_passed"] = ok
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
