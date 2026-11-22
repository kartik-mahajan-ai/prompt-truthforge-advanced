"""The TruthForge loop.

    retrieve -> generate -> split -> detect -> ROUTE -> act -> re-verify

Everything here is assembly. The router is the contribution; this module's job
is to execute its decisions faithfully, never silently override them, and
account for every token spent.

Three invariants worth defending in review:

  1. Re-verification always runs on the *spliced* answer, not on the rewritten
     claim in isolation. A rewrite that fixes a sentence but breaks the
     paragraph around it must be visible as such.
  2. A claim whose attempt budget is exhausted abstains. The loop never
     silently keeps the last bad version -- except in the `always_correct`
     arm, whose whole point is that it does.
  3. Actions the environment cannot perform (RERETRIEVE on a fixed-passage
     dataset) are downgraded explicitly, with the original decision preserved
     in the log. They are never quietly dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .claims.splitter import RuleSplitter, Splitter
from .config import Config
from .correct.rewrite import Rewriter, splice
from .cost import CostLedger
from .detect.base import Detector
from .detect.baselines import contradiction_signal, coverage, question_coverage
from .retrieve.base import Retriever
from .router.base import ClaimState
from .types import (
    Action,
    Claim,
    CorrectionAttempt,
    Passage,
    TurnResult,
    Verdict,
)


@dataclass
class LoopComponents:
    """Everything the loop needs, injected rather than constructed.

    Keeping construction outside the loop is what lets the Gate 1 script run a
    detector with no generator, and what enforces trap #2 (corrector and judge
    are separate objects, chosen by the caller).
    """

    detector: Detector
    router: object                       # duck-typed Router
    retriever: Retriever
    splitter: Splitter = None            # type: ignore[assignment]
    rewriter: Optional[Rewriter] = None
    generate_fn: Optional[Callable[[str], str]] = None
    # Do the SOURCES disagree with each other about this claim? Signature:
    # (claim_text, passages) -> float in [0, 1]. This feeds the ABSTAIN
    # branch, and it is a different question from "is the claim supported":
    # silent evidence means re-retrieve, self-contradicting evidence means
    # abstain. Defaults to the lexical placeholder in `detect.baselines`;
    # pass `detect.nli.ContradictionAnalyzer(...).score` for the real thing.
    contradiction_fn: Optional[Callable[[str, list], float]] = None
    # Optional second verifier that must CONFIRM a flag before the router may
    # withhold a claim: (claim, passages) -> True if it also judges the claim
    # unsupported. Every abstention inherits the precision of the flag behind
    # it, and on RAGTruth QA a single verifier is right about one flag in five.
    # On the training split, requiring NLI and MiniCheck to agree raised flag
    # precision from 0.228 to 0.308 (scripts/verifier_agreement.py). An
    # unconfirmed abstention becomes an ACCEPT, with the reason recorded.
    confirm_fn: Optional[Callable[[object, list], bool]] = None

    def __post_init__(self) -> None:
        if self.splitter is None:
            self.splitter = RuleSplitter()
        if self.contradiction_fn is None:
            self.contradiction_fn = contradiction_signal

    @property
    def contradiction_is_real(self) -> bool:
        """False while the lexical placeholder is in use.

        Weak link #2 in DECISIONS.md: the placeholder compares numeric
        literals and misses most real contradictions, so the
        `contradictory_sources` routing reason is not evidence of anything
        until a real NLI analyzer is supplied.
        """
        return self.contradiction_fn is not contradiction_signal


class TruthForgeLoop:
    def __init__(self, components: LoopComponents, cfg: Optional[Config] = None) -> None:
        self.c = components
        self.cfg = cfg or Config()

    # ----------------------------------------------------------------- run

    def run(
        self,
        question: str,
        answer: Optional[str] = None,
        passages: Optional[Sequence[Passage]] = None,
        example_id: Optional[str] = None,
        dataset: str = "unknown",
    ) -> TurnResult:
        """Run one turn.

        Pass `answer` to evaluate a pre-existing response (RAGTruth mode); omit
        it to generate one. Pass `passages` to fix the evidence; omit it to
        retrieve.
        """
        t_start = time.perf_counter()
        ledger = CostLedger()

        # -- 1. retrieve ---------------------------------------------------
        if passages is not None:
            current_passages = list(passages)
        else:
            with ledger.track("retrieve") as rec:
                current_passages = self.c.retriever.retrieve(
                    question, self.cfg.retrieval.top_k
                )
                rec.retrieval_calls = 1

        # -- 1b. answerability gate ----------------------------------------
        # Before spending a generation, ask whether the evidence is about the
        # question at all. This is a different question from "is each claim
        # supported", and claim-level verification cannot answer it: evidence
        # about X plus a question about Y produces an answer about X in which
        # every claim is genuinely supported by the evidence.
        #
        # It applies ONLY when this system performed the retrieval. When the
        # caller hands in fixed passages -- RAGTruth, or any benchmark that
        # pairs a question with its own evidence -- relevance was decided
        # elsewhere and the gate would be measuring the benchmark's
        # construction rather than the system's judgement.
        gate_applies = (
            passages is None
            and getattr(self.cfg.router, "answerability_gate", False)
        )
        q_cov = None
        if gate_applies:
            q_cov = question_coverage(question, current_passages)

            # Low coverage has two causes with the same symptom: the corpus
            # does not contain the answer, or the retriever did not find it.
            # Only the first justifies refusing. So widen the search once
            # before giving up -- the same "go look again before concluding
            # you cannot know" principle the claim router applies at rule 4,
            # and the one case where RE-RETRIEVE can prevent a wrong refusal
            # rather than merely feed another rewrite.
            if (
                q_cov < self.cfg.router.min_question_coverage
                and getattr(self.c.retriever, "supports_reretrieval", False)
            ):
                with ledger.track("reretrieve") as rec:
                    widened = self.c.retriever.retrieve(
                        question, self.cfg.retrieval.reretrieve_top_k
                    )
                    rec.retrieval_calls = 1
                current_passages = _merge_passages(current_passages, widened)
                widened_cov = question_coverage(question, current_passages)
                result_meta_widened = {
                    "answerability_reretrieved": True,
                    "question_coverage_before_reretrieval": round(q_cov, 4),
                }
                q_cov = widened_cov
            else:
                result_meta_widened = {}

            if q_cov < self.cfg.router.min_question_coverage:
                result = TurnResult(
                    question=question,
                    initial_answer="",
                    final_answer=self.cfg.loop.abstain_message,
                    dataset=dataset,
                    example_id=example_id,
                    passages=list(current_passages),
                    abstained=True,
                )
                result.wall_clock_s = time.perf_counter() - t_start
                result.costs = ledger.records
                result.meta.update({
                    "abstained_claim_ids": [],
                    "refused_before_generating": True,
                    "question_coverage": round(q_cov, 4),
                    "note": "evidence does not address the question",
                    **result_meta_widened,
                })
                return result

        # -- 2. generate ---------------------------------------------------
        if answer is None:
            if self.c.generate_fn is None:
                raise ValueError("no answer supplied and no generate_fn configured")
            with ledger.track("generate") as rec:
                answer = self._generate_answer(question, current_passages)
                rec.llm_calls = 1
                rec.prompt_tokens = _tok(question) + sum(
                    _tok(p.text) for p in current_passages
                )
                rec.completion_tokens = _tok(answer)

        initial_answer = answer
        current_answer = answer

        # -- 3. split ------------------------------------------------------
        with ledger.track("split") as rec:
            claims = self.c.splitter.split(current_answer)
            if getattr(self.c.splitter, "name", "") == "llm":
                rec.llm_calls = 1

        result = TurnResult(
            question=question,
            initial_answer=initial_answer,
            final_answer=current_answer,
            dataset=dataset,
            example_id=example_id,
            passages=list(current_passages),
            claims=list(claims),
        )
        if not claims:
            result.wall_clock_s = time.perf_counter() - t_start
            result.costs = ledger.records
            result.meta["note"] = "splitter produced no claims"
            return result

        # -- 4..7. the loop -------------------------------------------------
        history: dict[str, list[Action]] = {c.claim_id: [] for c in claims}
        # Whether the most recent REWRITE on each claim left it unchanged.
        # Feeds the router's futile-retry rule.
        rewrite_failed: dict[str, bool] = {c.claim_id: False for c in claims}
        by_id = {c.claim_id: c for c in claims}
        abstained_claims: set[str] = set()
        latest_verdict: dict[str, Verdict] = {}

        for iteration in range(self.cfg.loop.max_iterations):
            with ledger.track("detect") as rec:
                verdicts = self.c.detector.check(claims, current_passages)
                rec.detector_calls = len(claims)
                rec.prompt_tokens = _detector_tokens(claims, current_passages)
            result.verdicts.extend(verdicts)
            latest_verdict.update({v.claim_id: v for v in verdicts})

            actionable = []
            for v in verdicts:
                if v.claim_id in abstained_claims:
                    continue
                claim = by_id[v.claim_id]
                state = ClaimState(
                    claim_id=v.claim_id,
                    claim_text=claim.text,
                    verdict=v,
                    coverage=coverage(claim.text, current_passages),
                    contradiction=self.c.contradiction_fn(claim.text, current_passages),
                    iteration=iteration,
                    history=history[v.claim_id],
                    last_rewrite_failed=rewrite_failed[v.claim_id],
                )
                d = self.c.router.route(state, current_passages)
                d = self._downgrade_if_unavailable(d)
                d = self._confirm_abstain(d, claim, current_passages, ledger)
                result.decisions.append(d)
                if d.action in (Action.REWRITE, Action.RERETRIEVE):
                    actionable.append((state, d))
                elif d.action is Action.ABSTAIN:
                    abstained_claims.add(v.claim_id)

            result.iterations_used = iteration + 1

            if not actionable:
                break

            # -- execute ---------------------------------------------------
            for state, d in actionable:
                claim = by_id[state.claim_id]
                history[state.claim_id].append(d.action)

                if d.action is Action.RERETRIEVE:
                    with ledger.track("reretrieve") as rec:
                        widened = self.c.retriever.retrieve(
                            f"{question} {claim.text}",
                            self.cfg.retrieval.reretrieve_top_k,
                        )
                        rec.retrieval_calls = 1
                    current_passages = _merge_passages(current_passages, widened)
                    result.passages = list(current_passages)
                    # New evidence -> a rewrite retry is meaningful again.
                    rewrite_failed[state.claim_id] = False
                    # No text changed, so no CorrectionAttempt text diff; the
                    # attempt is still logged so its cost and outcome count.
                    result.attempts.append(
                        CorrectionAttempt(
                            claim_id=state.claim_id,
                            action=Action.RERETRIEVE,
                            iteration=iteration,
                            before_text=claim.text,
                            after_text=claim.text,
                            before_support_prob=state.verdict.support_prob,
                            cost=ledger.records[-1],
                        )
                    )
                    continue

                # REWRITE
                if self.c.rewriter is None:
                    raise ValueError("router chose REWRITE but no rewriter configured")
                rw = self.c.rewriter.rewrite(claim, current_answer, current_passages)
                ledger.add(rw.cost)
                # A rewrite that changed nothing (or was rejected) has failed.
                # Greedy decoding makes an identical retry deterministic.
                rewrite_failed[state.claim_id] = not rw.changed
                new_answer = splice(current_answer, claim, rw.rewritten)

                result.attempts.append(
                    CorrectionAttempt(
                        claim_id=state.claim_id,
                        action=Action.REWRITE,
                        iteration=iteration,
                        before_text=claim.text,
                        after_text=rw.rewritten,
                        before_support_prob=state.verdict.support_prob,
                        cost=rw.cost,
                        rejection=rw.rejection,
                    )
                )
                if rw.abstained:
                    abstained_claims.add(state.claim_id)

                current_answer = new_answer
                # The claim object must track its new surface text, or the next
                # iteration re-checks the stale version.
                by_id[state.claim_id] = _reanchor(claim, rw.rewritten, new_answer)

            # -- follow up on failed corrections ---------------------------
            # A rewrite that changed nothing is a failed correction, and the
            # router may want to abstain rather than leave a claim it believes
            # is false standing as fact. Re-route those claims NOW instead of
            # waiting for the next pass: at max_iterations=1 there is no next
            # pass, and the claim would silently survive. Routing is pure
            # logic, so this costs no tokens and no model calls.
            #
            # `always_correct` never returns ABSTAIN, so its behaviour is
            # unchanged -- that asymmetry is the point of the comparison.
            for state, d in actionable:
                cid = state.claim_id
                if d.action is not Action.REWRITE or not rewrite_failed[cid]:
                    continue
                if cid in abstained_claims:
                    continue
                follow_up = ClaimState(
                    claim_id=cid,
                    claim_text=by_id[cid].text,
                    verdict=state.verdict,
                    coverage=state.coverage,
                    contradiction=state.contradiction,
                    iteration=iteration,
                    history=history[cid],
                    last_rewrite_failed=True,
                )
                fd = self._downgrade_if_unavailable(
                    self.c.router.route(follow_up, current_passages)
                )
                fd = self._confirm_abstain(fd, by_id[cid], current_passages, ledger)
                if fd.action is Action.ABSTAIN:
                    result.decisions.append(fd)
                    abstained_claims.add(cid)

            claims = [by_id[c.claim_id] for c in claims]
            result.claims = list(claims)

        # -- 8. final re-verification + abstention -------------------------
        with ledger.track("reverify") as rec:
            final_verdicts = self.c.detector.check(claims, current_passages)
            rec.detector_calls = len(claims)
            rec.prompt_tokens = _detector_tokens(claims, current_passages)
        result.verdicts.extend(final_verdicts)
        final_by_id = {v.claim_id: v for v in final_verdicts}

        for attempt in result.attempts:
            v = final_by_id.get(attempt.claim_id)
            if v is not None and attempt.after_support_prob is None:
                attempt.after_support_prob = v.support_prob

        if abstained_claims:
            current_answer = self._apply_abstentions(
                current_answer, [by_id[cid] for cid in abstained_claims]
            )
            result.abstained = True

        result.final_answer = current_answer
        result.costs = ledger.records
        result.wall_clock_s = time.perf_counter() - t_start
        result.meta["abstained_claim_ids"] = sorted(abstained_claims)
        if q_cov is not None:
            result.meta["question_coverage"] = round(q_cov, 4)
            if gate_applies and result_meta_widened:
                result.meta.update(result_meta_widened)
        result.meta["contradiction_signal_is_real"] = self.c.contradiction_is_real
        return result

    # ------------------------------------------------------------- helpers

    def _confirm_abstain(self, d, claim, passages, ledger):
        """Withhold a claim only if a second verifier also flags it.

        An abstention that the confirming verifier does not back becomes an
        ACCEPT, and the original reason is kept in the new reason so the
        ledger can count how often confirmation overruled the router. With no
        confirming verifier configured this is a no-op.
        """
        if self.c.confirm_fn is None or d.action is not Action.ABSTAIN:
            return d
        with ledger.track("confirm") as rec:
            flagged = bool(self.c.confirm_fn(claim, list(passages)))
            rec.detector_calls = 1
            rec.prompt_tokens = _detector_tokens([claim], passages)
        if flagged:
            d.reason = f"{d.reason}+confirmed"
            return d
        d.action = Action.ACCEPT
        d.reason = f"abstain_not_confirmed({d.reason})"
        return d

    def _downgrade_if_unavailable(self, d):
        """RERETRIEVE on a fixed-passage dataset cannot do anything.

        Rather than pretending it ran, the decision is rewritten to ABSTAIN and
        the original is kept in the reason string, so RAGTruth rows are
        distinguishable from genuine abstentions in the error analysis.
        """
        if d.action is Action.RERETRIEVE and not getattr(
            self.c.retriever, "supports_reretrieval", True
        ):
            d.reason = f"reretrieve_unavailable({d.reason})"
            d.action = Action.ABSTAIN
        return d

    def _generate_answer(self, question: str, passages: Sequence[Passage]) -> str:
        """Draft an answer from the retrieved passages.

        The instructions are blunt about not copying because a small model
        will otherwise return the passage verbatim. That failure is quietly
        catastrophic for this system: an answer that IS the evidence is
        trivially supported by the evidence, so every claim passes
        verification, nothing is flagged, the router never acts, and the whole
        pipeline reports a perfect result while having answered nothing.
        """
        evidence = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
        prompt = (
            "Answer the QUESTION using only the PASSAGES.\n\n"
            f"PASSAGES:\n{evidence}\n\n"
            f"QUESTION: {question}\n\n"
            "Rules:\n"
            "- Answer in your own words, in at most three sentences.\n"
            "- Do NOT copy the passages. Do NOT repeat headings, section "
            "numbers or bracketed markers.\n"
            "- Use only facts stated in the passages. Add nothing else.\n"
            "- If the passages do not answer the question, reply exactly:\n"
            "  The documents do not contain this information.\n\n"
            "ANSWER:"
        )
        raw = (self.c.generate_fn(prompt) or "").strip()
        return _strip_echoed_passages(raw, passages)

    def _apply_abstentions(self, answer: str, claims: Sequence[Claim]) -> str:
        """Render abstained claims, last span first so earlier splices cannot
        invalidate later offsets."""
        import re

        style = getattr(self.cfg.loop, "abstain_style", "message")
        replacement = "" if style == "remove" else self.cfg.loop.abstain_message

        ordered = sorted(
            claims, key=lambda c: (c.start if c.start is not None else -1), reverse=True
        )
        out = answer
        for c in ordered:
            out = splice(out, c, replacement)

        if style == "remove":
            # Deleting a sentence leaves doubled spaces and a possible leading
            # or trailing gap. Tidy without touching the surviving text.
            out = re.sub(r"[ 	]{2,}", " ", out).strip()
        return out


def _reanchor(claim: Claim, new_text: str, new_answer: str) -> Claim:
    """Rebuild a claim's offsets after its text was spliced into the answer."""
    pos = new_answer.find(new_text)
    return Claim(
        text=new_text,
        claim_id=claim.claim_id,
        start=pos if pos >= 0 else None,
        end=pos + len(new_text) if pos >= 0 else None,
        sent_index=claim.sent_index,
        meta={**claim.meta, "surface": new_text, "rewritten": True},
    )


def _merge_passages(current: Sequence[Passage], new: Sequence[Passage]) -> list[Passage]:
    """Union by doc_id, keeping the original evidence.

    Replacing rather than merging would let a re-retrieval *remove* the
    passage that supported a good claim, which shows up later as info loss
    caused by the retriever rather than the rewriter -- a confound worth
    designing out.
    """
    seen = {p.doc_id for p in current}
    merged = list(current)
    for p in new:
        if p.doc_id not in seen:
            merged.append(p)
            seen.add(p.doc_id)
    return merged


def _tok(text: str) -> int:
    from .cost import count_tokens

    return count_tokens(text)


def _detector_tokens(claims: Sequence[Claim], passages: Sequence[Passage]) -> int:
    """Input tokens the judge consumes for one pass over all claims.

    The evidence is re-encoded per claim, which is exactly why the detector
    dominates the token column on long contexts. Leaving this out would make
    the cost comparison flattering to whichever arm detects most -- i.e. to
    every arm in this project.
    """
    evidence = sum(_tok(p.text) for p in passages)
    return sum(_tok(c.text) + evidence for c in claims)


def _strip_echoed_passages(answer: str, passages: Sequence[Passage]) -> str:
    """Drop lines the model copied verbatim out of the evidence.

    Belt and braces for the prompt above. A copied line is not an answer, and
    leaving it in makes the claim trivially supported by the evidence it was
    copied from -- so verification passes, the router finds nothing to do, and
    the system reports success on a non-answer.

    Only exact line matches are removed. Anything the model actually composed,
    even if it reuses wording, is left alone.
    """
    if not answer:
        return answer

    evidence_lines = {
        ln.strip()
        for p in passages
        for ln in p.text.splitlines()
        if len(ln.strip()) >= 25
    }
    if not evidence_lines:
        return answer

    kept = [
        ln for ln in answer.splitlines()
        if ln.strip() and ln.strip() not in evidence_lines
    ]
    cleaned = "\n".join(kept).strip()

    # If stripping removed everything, the model produced nothing of its own.
    # Say so rather than returning the evidence back to the user.
    return cleaned if cleaned else "The documents do not contain this information."
