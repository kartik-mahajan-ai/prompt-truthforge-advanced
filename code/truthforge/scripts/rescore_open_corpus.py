"""Re-score a saved open-corpus run without re-running the models.

    python scripts/rescore_open_corpus.py runs/open_corpus.json

The answers are in the artifact, so a scoring bug is fixable offline. This
exists because one was: `score_turn` counted only router-driven abstention and
reported 0.0 correct refusals for every arm, while the generator was declining
on more than half the unanswerable questions in its own words. Re-running five
arms on the GPU to fix an `if` would have been the wrong trade.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truthforge.claims.splitter import answer_is_refusal  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/open_corpus.json")
    d = json.loads(path.read_text(encoding="utf-8"))

    from truthforge.config import Config

    # A saved row does not carry the loop's `abstained` flag, but a turn the
    # answerability gate refused has exactly the system's abstain message as
    # its answer and no claims. That is the system speaking, not the model,
    # so it counts as a ROUTER refusal.
    system_msg = Config().loop.abstain_message.strip()

    for arm, r in d["arms"].items():
        for row in r["rows"]:
            ans = (row.get("answer") or "").strip()
            gate = ans == system_msg and not row.get("n_claims")
            row["router_refused"] = bool(row.get("router_refused")) or gate
            row["text_refused"] = answer_is_refusal(ans) and not gate
            row["refused"] = row["router_refused"] or row["text_refused"]

        una = [x for x in r["rows"] if not x["answerable"]]
        ans = [x for x in r["rows"] if x["answerable"]]

        def rate(xs, key):
            return round(sum(1 for x in xs if x[key]) / len(xs), 4) if xs else None

        r["summary"].update({
            "correct_refusal": rate(una, "refused"),
            "over_refusal": rate(ans, "refused"),
            "correct_refusal_by_router": rate(una, "router_refused"),
            "over_refusal_by_router": rate(ans, "router_refused"),
        })

    d["config"]["rescored"] = "refusal counted at answer level, not router only"
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    head = (f"{'arm':<16}{'correct_ref':>13}{'(router)':>10}{'over_ref':>11}"
            f"{'(router)':>10}{'reretr/turn':>13}")
    print(head)
    print("-" * len(head))
    for arm, r in d["arms"].items():
        s = r["summary"]
        print(f"{arm:<16}{s['correct_refusal']:>13}{s['correct_refusal_by_router']:>10}"
              f"{s['over_refusal']:>11}{s['over_refusal_by_router']:>10}"
              f"{s['reretrieve_per_turn']:>13}")
    print(f"\nrewrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
