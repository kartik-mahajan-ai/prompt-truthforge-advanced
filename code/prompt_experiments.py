"""Prompt iteration experiments for the Milestone 2 log.

Runs every saved version of the two final prompts on the same fixed inputs and
records what each version actually produced:

  P2 corrector  v1 -> v2 -> v3 (final)   on 40 flagged claims (RAGTruth train)
  P1 generator  v1 -> v2 (final)         on 20 questions (RAGTruth train)

    python "Milestone 2/code/prompt_experiments.py"

Outputs: test-data/prompt_runs/*.json (every raw model output) and
test-data/prompt_runs/summary.json (the numbers in the iteration log).
The prompts are read from prompts/v*/ and prompts/final/, so the files in the
folder are exactly what was run.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # Milestone 2/
REPO = HERE.parent                                   # TruthForge/
sys.path.insert(0, str(REPO / "src"))

import truthforge.correct.rewrite as rw              # noqa: E402
from truthforge.claims.splitter import RuleSplitter  # noqa: E402
from truthforge.config import Config, GeneratorConfig  # noqa: E402
from truthforge.data.ragtruth import load_ragtruth   # noqa: E402
from truthforge.detect.minicheck import build_detector  # noqa: E402
from truthforge.models import build_generator        # noqa: E402

OUT = HERE / "test-data" / "prompt_runs"
N_CLAIMS, N_QUESTIONS = 40, 20


def prompt(path: str) -> str:
    return (HERE / "prompts" / path).read_text(encoding="utf-8").strip("\n")


def flagged_claims(detector):
    """First N_CLAIMS claims the NLI verifier flags (P(supported) < 0.5)."""
    exs = load_ragtruth(root=str(REPO / "data/raw/ragtruth"), split="train", task_types={"QA"}, limit=400)
    splitter, out = RuleSplitter(), []
    for ex in exs:
        claims = splitter.split(ex.response)
        if not claims:
            continue
        scores = detector.score_batch(claims, ex.passages)
        for c, s in zip(claims, scores):
            if s < 0.5:
                out.append((ex, c, s))
            if len(out) == N_CLAIMS:
                return out
    return out


def run_corrector(gen, detector, items, template: str, tag: str):
    rw.REWRITE_PROMPT = template                     # the corrector reads this constant
    rewriter = rw.Rewriter(gen)
    rows = []
    for ex, c, s in items:
        r = rewriter.rewrite(c, ex.response, ex.passages)
        after = None
        if r.changed and not r.abstained:
            after = detector.score_batch([rw.Claim(text=r.rewritten)], ex.passages)[0]
        rows.append({
            "example_id": ex.example_id, "claim": c.text, "support_before": round(s, 3),
            "raw_output": r.raw_output, "result": r.rewritten,
            "outcome": ("UNSUPPORTED" if r.abstained else "rejected:" + r.rejection if r.rejected
                        else "changed" if r.changed else "unchanged"),
            "support_after": None if after is None else round(after, 3),
            "output_tokens": r.cost.completion_tokens,
        })
    (OUT / f"corrector_{tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    n = len(rows)
    rej = [r for r in rows if r["outcome"].startswith("rejected")]
    chg = [r for r in rows if r["outcome"] == "changed"]
    reasons: dict[str, int] = {}
    for r in rej:
        reasons[r["outcome"][9:]] = reasons.get(r["outcome"][9:], 0) + 1
    multi = sum(1 for r in rows if len(re.findall(r"[.!?](\s|$)", r["raw_output"].strip())) > 1)
    return {
        "n": n,
        "rejected_by_guard": len(rej), "rejection_reasons": reasons,
        "raw_output_multi_sentence": multi,
        "unsupported_sentinel": sum(r["outcome"] == "UNSUPPORTED" for r in rows),
        "changed": len(chg), "unchanged": sum(r["outcome"] == "unchanged" for r in rows),
        "changed_and_verifier_supported": sum(1 for r in chg if (r["support_after"] or 0) >= 0.5),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in rows) / n, 1),
    }


def run_generator(gen, questions, template: str, tag: str):
    rows = []
    for ex in questions:
        evidence = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(ex.passages))
        raw = (gen(template.format(evidence=evidence, question=ex.question)) or "").strip()
        words = raw.split()
        # a "copy" = an 8-word run of the output found verbatim in the passages
        joined = " ".join(" ".join(p.text.split()) for p in ex.passages)
        copied = any(" ".join(words[i:i + 8]) in joined for i in range(max(0, len(words) - 7)))
        rows.append({"example_id": ex.example_id, "question": ex.question, "output": raw,
                     "copied_8gram": copied, "has_bracket_marker": bool(re.search(r"\[\d+\]", raw)),
                     "sentences": len(re.findall(r"[.!?](\s|$)", raw)), "words": len(words)})
    (OUT / f"generator_{tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    n = len(rows)
    return {"n": n, "copied_8gram": sum(r["copied_8gram"] for r in rows),
            "bracket_markers": sum(r["has_bracket_marker"] for r in rows),
            "over_three_sentences": sum(r["sentences"] > 3 for r in rows),
            "mean_words": round(sum(r["words"] for r in rows) / n, 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.detector.name, cfg.detector.device = "nli", "cuda"
    detector = build_detector(cfg)
    gen = build_generator("local", GeneratorConfig(device="cuda"))
    t0 = time.time()

    items = flagged_claims(detector)
    print(f"{len(items)} flagged claims")
    summary = {"corrector": {}, "generator": {}, "settings": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "decoding": "greedy (temperature 0)",
        "max_new_tokens": GeneratorConfig().max_new_tokens, "verifier": "NLI DeBERTa-v3, flag below 0.5",
        "inputs": f"first {N_CLAIMS} NLI-flagged claims / first {N_QUESTIONS} questions, RAGTruth QA train split"}}
    for tag, path in [("v1", "v1/P2_corrector.txt"), ("v2", "v2/P2_corrector.txt"), ("v3", "final/P2_corrector.txt")]:
        summary["corrector"][tag] = run_corrector(gen, detector, items, prompt(path), tag)
        print("corrector", tag, summary["corrector"][tag])

    qs = load_ragtruth(root=str(REPO / "data/raw/ragtruth"), split="train", task_types={"QA"}, limit=N_QUESTIONS)
    for tag, path in [("v1", "v1/P1_generator.txt"), ("v2", "final/P1_generator.txt")]:
        summary["generator"][tag] = run_generator(gen, qs, prompt(path), tag)
        print("generator", tag, summary["generator"][tag])

    summary["seconds"] = round(time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
