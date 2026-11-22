"""Correction actions: rewrite and abstain.

Design rule that the whole over-correction story depends on: a rewrite edits
ONE claim in place and splices it back into the answer. It never regenerates
the whole answer. Whole-answer regeneration makes `info_loss` and `new_error`
unattributable -- you cannot tell which flagged claim caused the damage -- and
it is the reason existing work cannot report this taxonomy.

**Small instruct models do not reliably obey "output one sentence".** Observed
with Qwen2.5-1.5B on the very first real run: asked to correct the third
sentence, it echoed the entire answer back. Splicing that over one sentence's
span duplicated the first two sentences into the final answer, and because
every word of the duplicate appears in the evidence, the outcome classifier
scored it SAME -- a corrupted answer recorded as a harmless no-op.

So the model's output is never trusted verbatim. `_extract_corrected_sentence`
recovers the intended sentence where it can and REJECTS the rewrite where it
cannot, and every rejection is counted in `RewriteResult.rejection`. A rising
rejection rate is a real finding about the corrector, not noise to hide: report
it next to the worse-rate.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..types import Claim, CostRecord, Passage

REWRITE_PROMPT = """\
Correct ONE sentence so that it matches the evidence.

EVIDENCE:
{evidence}

SURROUNDING ANSWER (context only -- never reproduce this):
{answer}

SENTENCE TO CORRECT:
{claim}

Rules:
- Output exactly ONE sentence: the corrected version of SENTENCE TO CORRECT.
- Do NOT repeat the surrounding answer. Do NOT output more than one sentence.
- No preamble, no explanation, no quotes, no bullet points.
- Keep every detail the evidence supports. Do not shorten to play safe.
- Do not add facts absent from the evidence.
- Do not hedge. If the evidence does not settle the point, output exactly:
  UNSUPPORTED
- Keep the original wording wherever the evidence does not require a change.

CORRECTED SENTENCE:"""

ABSTAIN_SENTINEL = "UNSUPPORTED"

# A corrected sentence may legitimately grow (adding a supported detail), but
# not without limit. Beyond this multiple of the original it is an echo, not a
# correction.
MAX_LENGTH_RATIO = 2.5

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    abstained: bool
    changed: bool
    cost: CostRecord
    # None when the model's output was used; otherwise why it was thrown away.
    rejection: Optional[str] = None
    raw_output: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.rejection is not None


class Rewriter:
    """Single-claim corrector.

    Takes a plain `generate_fn` so this module never imports torch, and so the
    corrector can be swapped for a different model than the judge without
    touching the loop (trap #2: corrector and judge must not be the same model).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        abstain_message: str = "I could not verify this from the retrieved sources.",
        count_tokens_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.generate = generate_fn
        self.abstain_message = abstain_message
        self._count = count_tokens_fn

    def rewrite(
        self, claim: Claim, answer: str, passages: list[Passage], evidence_chars: int = 6000
    ) -> RewriteResult:
        evidence = "\n\n".join(p.text for p in passages)[:evidence_chars]
        prompt = REWRITE_PROMPT.format(
            evidence=evidence, answer=answer, claim=claim.text
        )

        cost = CostRecord(stage="rewrite", llm_calls=1)
        # Time it. This record is built by hand rather than through
        # CostLedger.track, so without this latency_s stays 0 and the
        # rewriting stage -- usually the most expensive one -- reports as free
        # in the latency column.
        _t0 = time.perf_counter()
        raw = (self.generate(prompt) or "").strip()
        cost.latency_s = time.perf_counter() - _t0
        cost.prompt_tokens = self._tok(prompt)
        cost.completion_tokens = self._tok(raw)

        # Models like to wrap the answer in quotes or prefix it with a label.
        cleaned = _strip_wrapper(raw)

        if not cleaned or cleaned.upper().startswith(ABSTAIN_SENTINEL):
            return RewriteResult(
                original=claim.text,
                rewritten=self.abstain_message,
                abstained=True,
                changed=True,
                cost=cost,
                raw_output=raw,
            )

        candidate, rejection = _extract_corrected_sentence(cleaned, claim.text, answer)

        if rejection is not None:
            # Keep the original. An unchanged claim is an honest SAME outcome;
            # splicing a malformed rewrite would corrupt the answer and be
            # scored as a no-op, which is far worse.
            return RewriteResult(
                original=claim.text,
                rewritten=claim.text,
                abstained=False,
                changed=False,
                cost=cost,
                rejection=rejection,
                raw_output=raw,
            )

        return RewriteResult(
            original=claim.text,
            rewritten=candidate,
            abstained=False,
            changed=candidate.strip() != claim.text.strip(),
            cost=cost,
            raw_output=raw,
        )

    def _tok(self, text: str) -> int:
        if self._count:
            return self._count(text)
        from ..cost import count_tokens

        return count_tokens(text)


def _strip_wrapper(text: str) -> str:
    """Remove a leading label and surrounding quotes from a model output."""
    out = text.strip()
    for prefix in ("CORRECTED SENTENCE:", "Corrected sentence:", "Answer:", "Output:"):
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
    # A corrected sentence is one line; chatty models append commentary.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    out = lines[0] if lines else ""
    # Only strip quotes if they wrap the entire string.
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    return out


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(a: str, b: str) -> float:
    """Jaccard overlap of content tokens. Cheap and adequate for picking the
    sentence a model meant to be the rewrite."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _extract_corrected_sentence(
    output: str, claim: str, answer: str
) -> tuple[str, Optional[str]]:
    """Recover the single corrected sentence, or explain why we cannot.

    Returns `(sentence, None)` on success and `("", reason)` on rejection.

    The failure this exists for: a small instruct model, given the surrounding
    answer as context, echoes the whole answer instead of one sentence. Pasting
    that over a single sentence's span duplicates text and the damage is
    invisible to an attempt-level outcome classifier, because every duplicated
    word is genuinely supported by the evidence.
    """
    text = output.strip()
    if not text:
        return "", "empty_output"

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    if len(sentences) > 1:
        # Which of the produced sentences is the rewrite? The one closest to
        # the claim -- but only if it is meaningfully closer than the others,
        # otherwise the model produced prose rather than a correction.
        scored = sorted(sentences, key=lambda s: _overlap(s, claim), reverse=True)
        best = scored[0]
        if _overlap(best, claim) < 0.25:
            return "", f"no_sentence_matches_claim({len(sentences)}_sentences)"

        # If the other sentences are lifted verbatim from the surrounding
        # answer, the model echoed the answer back at us.
        others = [s for s in sentences if s is not best]
        echoed = sum(1 for s in others if s and s in answer)
        if echoed >= 1:
            recoverable = (
                _overlap(best, claim) >= 0.4
                and len(best) <= len(claim) * MAX_LENGTH_RATIO
            )
            if not recoverable:
                return "", f"echoed_surrounding_answer({echoed}_sentences_copied)"
            if best.strip() == claim.strip():
                # The echo contains the target sentence completely unchanged:
                # the model produced no correction at all. Recovering it would
                # look like a deliberate "no change needed", which is a
                # different thing and would hide the instruction failure.
                return "", f"echoed_answer_no_correction({echoed}_sentences_copied)"
            text = best
        else:
            text = best

    if len(text) > max(80, len(claim) * MAX_LENGTH_RATIO):
        return "", (
            f"output_too_long({len(text)}_chars_vs_claim_{len(claim)})"
        )

    # A rewrite that shares almost nothing with the claim is answering a
    # different question.
    if _overlap(text, claim) < 0.15:
        return "", "output_unrelated_to_claim"

    return text, None


def splice(answer: str, claim: Claim, replacement: str) -> str:
    """Replace one claim's span in the answer, preserving everything else.

    Prefers character offsets from the splitter. Falls back to an exact
    substring replace of the claim's surface form, and returns the answer
    unchanged if neither is available -- an unchanged answer is recorded as a
    SAME outcome, which is the correct accounting for a failed splice.
    """
    surface = claim.meta.get("surface", claim.text)

    if claim.start is not None and claim.end is not None:
        if 0 <= claim.start < claim.end <= len(answer):
            return answer[: claim.start] + replacement + answer[claim.end :]

    if surface and surface in answer:
        return answer.replace(surface, replacement, 1)
    if claim.text in answer:
        return answer.replace(claim.text, replacement, 1)
    return answer
