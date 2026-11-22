"""An LLM annotator for correction outcomes, and what it can and cannot prove.

The project's plan was ~300 correction attempts double-labelled by two people,
with Cohen's kappa between them. Two humans agreeing is evidence that the
*construct* is well defined -- that "this rewrite lost information" means
something another careful reader arrives at independently. That is what makes
the heuristic classifier's agreement with them meaningful.

**Two language models agreeing proves something much weaker.** It shows the
models are similar. Two models can agree perfectly and both be wrong in the
same direction, and modern instruct models are correlated by training data,
architecture and RLHF style. Swapping human annotators for model annotators
and reporting the kappa as if it were the same number would be a
methodological error, so this module does not pretend otherwise:

* Every label records `outcome_source` -- "heuristic", "llm:<model>", or
  "human" -- and `agreement_report` refuses to average across kinds.
* The agreement that is actually worth reporting here is between the
  RULE-BASED heuristic and an LLM judge. Those two are methodologically
  independent: one counts tokens and hedge words, the other reads. When
  independent methods agree, that is informative. Two LLMs agreeing is mostly
  a statement about LLMs.
* A human anchor is still needed, but it does not need to be 300 items. Fifty
  labelled by one person is enough to say whether the automatic labels track
  human judgement at all, and the paper can report exactly that.

So: use this to scale labelling and to cross-check the heuristic. Do not use
it to replace the human anchor entirely, and never report model-model kappa
without saying what it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..types import CorrectionAttempt, Outcome, Passage, WorseKind

JUDGE_PROMPT = """\
You are grading whether an automatic correction improved a sentence.

Judge ONLY against the EVIDENCE below. Ignore anything you know about the \
world. If the evidence does not support something, it is unsupported even if \
it is true in reality.

EVIDENCE:
{evidence}

BEFORE: {before}
AFTER:  {after}

Choose exactly one label:
BETTER     - the rewrite fixed a real error, and lost nothing the evidence supported
SAME       - nothing meaningful changed, or the change neither helped nor hurt
INFO_LOSS  - the rewrite dropped a detail the evidence DID support
HEDGING    - a clear statement became vague or qualified without cause
NEW_ERROR  - the rewrite introduced a claim the evidence does not support

Reply with the label only, on one line. No explanation.

LABEL:"""

_LABELS = {
    "BETTER": (Outcome.BETTER, None),
    "SAME": (Outcome.SAME, None),
    "INFO_LOSS": (Outcome.WORSE, WorseKind.INFO_LOSS),
    "HEDGING": (Outcome.WORSE, WorseKind.HEDGING),
    "NEW_ERROR": (Outcome.WORSE, WorseKind.NEW_ERROR),
}

# Longest first, so INFO_LOSS is not matched as part of some other token.
_LABEL_RE = re.compile("|".join(sorted(_LABELS, key=len, reverse=True)), re.IGNORECASE)


@dataclass
class JudgeResult:
    outcome: Optional[Outcome]
    worse_kind: Optional[WorseKind]
    raw: str
    parsed: bool


class LLMJudge:
    """Labels a correction attempt with an instruct model.

    Takes a plain `generate_fn` so it never imports torch and so the judge can
    be a different model from the corrector -- which it must be. Grading a
    model's work with the same model measures self-consistency, not quality.
    """

    def __init__(self, generate_fn: Callable[[str], str], name: str = "llm") -> None:
        self.generate = generate_fn
        self.name = name

    def judge(
        self,
        before: str,
        after: str,
        passages: Sequence[Passage],
        evidence_chars: int = 4000,
    ) -> JudgeResult:
        if before.strip() == after.strip():
            # Not worth a model call, and the model would only guess.
            return JudgeResult(Outcome.SAME, None, "(unchanged)", True)

        evidence = "\n\n".join(p.text for p in passages)[:evidence_chars]
        raw = (
            self.generate(
                JUDGE_PROMPT.format(evidence=evidence, before=before, after=after)
            )
            or ""
        ).strip()

        m = _LABEL_RE.search(raw)
        if not m:
            # An unparseable reply is not a SAME. Recording it as one would
            # quietly inflate agreement with the heuristic, which also favours
            # SAME. Leave it unlabelled and count it.
            return JudgeResult(None, None, raw, False)

        outcome, kind = _LABELS[m.group(0).upper()]
        return JudgeResult(outcome, kind, raw, True)

    def label_attempt(
        self, attempt: CorrectionAttempt, passages: Sequence[Passage]
    ) -> CorrectionAttempt:
        """Fill in `outcome` / `worse_kind` on a copy-safe basis."""
        r = self.judge(attempt.before_text, attempt.after_text, passages)
        attempt.outcome = r.outcome
        attempt.worse_kind = r.worse_kind
        attempt.outcome_source = f"llm:{self.name}"
        attempt.notes = r.raw[:200]
        return attempt


def label_batch(
    judge: LLMJudge,
    attempts: Sequence[CorrectionAttempt],
    passages_by_claim: dict[str, Sequence[Passage]],
) -> dict:
    """Label many attempts and report how often the judge was parseable."""
    labelled, unparsed = {}, 0
    for a in attempts:
        key = f"{a.claim_id}:{a.iteration}:{a.action.value}"
        r = judge.judge(
            a.before_text, a.after_text, passages_by_claim.get(a.claim_id, [])
        )
        if not r.parsed:
            unparsed += 1
            continue
        labelled[key] = {
            "outcome": r.outcome.value,
            "worse_kind": r.worse_kind.value if r.worse_kind else "",
            "annotator_id": f"llm:{judge.name}",
            "notes": r.raw[:200],
        }
    return {
        "labels": labelled,
        "n_attempts": len(attempts),
        "n_labelled": len(labelled),
        "n_unparseable": unparsed,
        "annotator": f"llm:{judge.name}",
        # Carried into every downstream report so the kind of agreement being
        # claimed is never ambiguous.
        "label_kind": "automatic",
    }
