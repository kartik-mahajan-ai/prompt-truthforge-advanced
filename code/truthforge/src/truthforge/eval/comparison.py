"""Head-to-head evaluation: on what basis can we say one arm beats another?

The awkward fact about this project is that TruthForge does strictly LESS work
than the always-correct baseline. It cannot win by fixing more things. So a
single "accuracy" column cannot decide the comparison, and the usual move --
report accuracy and hope the cost column is noticed -- is exactly the trap the
brief warns about.

This module supplies the three things a defensible claim needs.

1. A CLAIM-FATE LEDGER that makes over-correction and over-abstention
   symmetric and countable.

   Every claim in the original answer has a gold status (was it supported by
   the evidence?) and a fate (kept / rewritten / dropped). Crossing those gives
   six cells, and each cell is either a win or a loss:

                      | kept unchanged | rewritten        | dropped
       ---------------+----------------+------------------+-----------------
       was SUPPORTED  | TRUE_KEEP  ok  | FALSE_FIX   BAD  | FALSE_DROP  BAD
       was UNSUPPORTED| FALSE_KEEP BAD | TRUE_FIX    ok   | TRUE_DROP   ok

   FALSE_FIX is over-correction: a claim that was fine, rewritten into one that
   is not. FALSE_DROP is information loss. FALSE_KEEP is a missed hallucination.
   No system can minimise all three at once, which is the point.

2. TWO RATES THAT CANNOT BOTH BE GAMED.

       error_removal  = unsupported claims fixed or dropped / all unsupported
       retention      = supported claims still present and supported / all supported

   Abstain on everything: error_removal 1.0, retention 0.0.
   Change nothing:        retention 1.0, error_removal 0.0.
   Their harmonic mean (`net_quality`) punishes both, so it is the single
   number to lead with -- and unlike raw faithfulness it cannot be inflated by
   deleting whatever the system fails to fix.

3. PAIRED SIGNIFICANCE AND COST-MATCHED COMPARISON.

   Arms run on the same examples, so differences are paired: a paired bootstrap
   over turns is both more powerful and more honest than comparing two means.
   `compare_arms` reports the delta, its confidence interval, and whether the
   arms are Pareto-comparable at all.

A note on gold labels. With RAGTruth, `was supported` comes from the annotated
spans. Without it, this module falls back to the detector's own verdict on the
ORIGINAL answer, which is weaker -- the judge grades the work its own flags
caused. That fallback is recorded in `gold_source` on every result, and a
number computed from it must be reported as such.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..types import Label, TurnResult


class Fate(str, enum.Enum):
    """What the system did with a claim that was in the original answer."""

    KEPT = "kept"          # untouched
    REWRITTEN = "rewritten"
    DROPPED = "dropped"    # abstained on, or removed


class Cell(str, enum.Enum):
    """One cell of the claim-fate ledger. Three are wins, three are losses."""

    TRUE_KEEP = "true_keep"        # supported, left alone            ok
    FALSE_KEEP = "false_keep"      # unsupported, left alone          missed
    TRUE_FIX = "true_fix"          # unsupported, rewritten, now ok   ok
    FALSE_FIX = "false_fix"        # supported, rewritten, now broken OVER-CORRECTION
    TRUE_DROP = "true_drop"        # unsupported, removed             ok
    FALSE_DROP = "false_drop"      # supported, removed               INFO LOSS

    @property
    def is_win(self) -> bool:
        return self in (Cell.TRUE_KEEP, Cell.TRUE_FIX, Cell.TRUE_DROP)


WINS = (Cell.TRUE_KEEP, Cell.TRUE_FIX, Cell.TRUE_DROP)
LOSSES = (Cell.FALSE_KEEP, Cell.FALSE_FIX, Cell.FALSE_DROP)


@dataclass
class ClaimFate:
    claim_id: str
    was_supported: bool
    fate: Fate
    ends_supported: bool
    cell: Cell


def _classify(was_supported: bool, fate: Fate, ends_supported: bool) -> Cell:
    if fate is Fate.DROPPED:
        return Cell.FALSE_DROP if was_supported else Cell.TRUE_DROP
    if fate is Fate.REWRITTEN:
        # A rewrite is judged on where it LANDED, not on intent. Rewriting a
        # supported claim into an unsupported one is the failure this project
        # is named after; rewriting it into another supported one is harmless.
        if was_supported:
            return Cell.TRUE_KEEP if ends_supported else Cell.FALSE_FIX
        return Cell.TRUE_FIX if ends_supported else Cell.FALSE_KEEP
    return Cell.TRUE_KEEP if was_supported else Cell.FALSE_KEEP


def claim_fates(
    turn: TurnResult,
    gold: Optional[dict[str, bool]] = None,
    trust_post_rewrite: bool = True,
) -> list[ClaimFate]:
    """Resolve every claim in one turn to a ledger cell.

    **An asymmetry that must travel with every over-correction count.** "Was
    supported" can come from gold annotations; "ends supported" cannot, and
    never will, because the annotations describe the ORIGINAL response and a
    rewrite produces text that did not exist when the dataset was labelled. So
    `ends_supported` is always the detector's judgement.

    That matters because the detector's precision is 0.184 on this data. A
    gold-supported claim that a rewrite left fine, but which the detector flags
    afterwards, is counted as an over-correction. The over-correction column is
    therefore an UPPER bound, and the LLM-judge pass over the same rewrites
    exists to give it a second, independent opinion.

    **Human labels then showed it is far worse than an upper bound.** Fifty
    rewrites were labelled blind. Of 30 the ledger scored as over-corrections,
    the annotator judged 2 harmful -- precision 6.7% [1.8%, 21.3%] -- and 17
    were outright improvements, mostly the corrector cleaning up splitter
    artifacts. Of 9 the ledger scored as harmless, 4 were harmful. The
    detector's verdict on edited text does not just overcount damage, it runs
    roughly backwards.

    `trust_post_rewrite=False` is the correction those labels license: a
    rewritten claim keeps its GOLD status instead of taking the detector's
    post-rewrite verdict. The labels support that directly -- 28 of the 30
    "damaging" rewrites left the claim's support unchanged. It is
    conservative in one direction only: a rewrite can no longer be credited
    with fixing an unsupported claim, so error removal BY REWRITING is
    understated. Error removal by abstention is unaffected, because a dropped
    claim is scored against gold either way.

    Args:
        turn: a completed turn.
        gold: claim_id -> was_supported. When omitted, the detector's FIRST
            verdict on each claim stands in, which is the weaker fallback the
            module docstring warns about.
        trust_post_rewrite: score a rewritten claim by the detector's verdict
            on the new text (the pre-registered behaviour) or by its gold
            status (the human-validated correction).
    """
    first: dict[str, bool] = {}
    last: dict[str, bool] = {}
    for v in turn.verdicts:
        supported = v.label is Label.SUPPORTED
        first.setdefault(v.claim_id, supported)
        last[v.claim_id] = supported

    dropped = set(turn.meta.get("abstained_claim_ids", []))
    rewritten = {
        a.claim_id
        for a in turn.attempts
        if a.before_text.strip() != a.after_text.strip()
    }

    out: list[ClaimFate] = []
    for claim_id in first:
        if gold is not None and claim_id not in gold:
            # Falling back to the detector's verdict here would mix two gold
            # sources under one `gold_source: "annotated"` label. On a 60-turn
            # slice where the arms differ by 8 claims, one silently
            # detector-scored claim is enough to move a column. Skip it; the
            # caller counts the skips.
            continue
        was = gold[claim_id] if gold else first[claim_id]
        if claim_id in dropped:
            fate = Fate.DROPPED
        elif claim_id in rewritten:
            fate = Fate.REWRITTEN
        else:
            fate = Fate.KEPT
        if fate is Fate.REWRITTEN and not trust_post_rewrite:
            ends = was
        else:
            ends = last.get(claim_id, was)
        out.append(
            ClaimFate(claim_id, was, fate, ends, _classify(was, fate, ends))
        )
    return out


# --------------------------------------------------------------------------
# Per-arm scoring
# --------------------------------------------------------------------------


@dataclass
class ArmScore:
    arm: str
    n_turns: int
    n_claims: int
    cells: dict[str, int]
    error_removal: float
    retention: float
    net_quality: float
    tokens_per_turn: float
    latency_per_turn: float
    gold_source: str
    # Claims excluded because no gold label covered them (a claim with no
    # character offsets cannot be joined to a span annotation). Reported
    # rather than back-filled -- see `claim_fates`.
    n_unscorable: int = 0
    # Per-turn net_quality, kept so the bootstrap can pair turns across arms.
    per_turn: list[float] = field(default_factory=list)
    per_turn_tokens: list[float] = field(default_factory=list)
    # Per-turn cell counts, so a bootstrap can resample turns and
    # recompute the POOLED statistic rather than averaging per-turn ones.
    per_turn_cells: list[dict[str, int]] = field(default_factory=list)

    @property
    def net_claims_improved(self) -> int:
        """Errors removed minus correct claims damaged, in raw claims.

        Not the pre-registered headline and not a replacement for it. It is
        here because it stays interpretable where `net_quality` does not: it
        is monotone in both directions and a reader can check it by counting.
        """
        g = self.cells.get
        return (g("true_fix", 0) + g("true_drop", 0)) - (
            g("false_fix", 0) + g("false_drop", 0)
        )

    @property
    def net_quality_unstable(self) -> bool:
        return net_quality_is_unstable(self.cells)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "n_turns": self.n_turns,
            "n_claims": self.n_claims,
            "error_removal": round(self.error_removal, 4),
            "retention": round(self.retention, 4),
            "net_quality": round(self.net_quality, 4),
            "tokens_per_turn": round(self.tokens_per_turn, 1),
            "latency_per_turn": round(self.latency_per_turn, 4),
            "gold_source": self.gold_source,
            "n_unscorable": self.n_unscorable,
            "net_claims_improved": self.net_claims_improved,
            "net_quality_unstable": self.net_quality_unstable,
            **{k: v for k, v in self.cells.items()},
        }


# Below this, net_quality stops ordering arms sensibly -- see
# `net_quality_is_unstable`.
NEAR_ZERO_REMOVAL = 0.05


def net_quality_is_unstable(cells: dict[str, int]) -> bool:
    """Is `net_quality` in the regime where it ranks arms wrongly?

    net_quality is the harmonic mean of error removal and retention, which was
    chosen because it refuses to reward either degenerate strategy: delete
    everything scores 0, change nothing scores 0. That property holds at the
    ends. It fails in the middle-of-nothing:

        removal 0.0000, retention 1.0000  ->  0.0000   (changed nothing)
        removal 0.0082, retention 0.9648  ->  0.0163   (fixed 1, broke 69)
        removal 0.0082, retention 0.5000  ->  0.0161   (fixed 1, broke 500)

    A harmonic mean is pinned near zero whenever either term is near zero, so
    once error removal is tiny the comparison between "exactly zero" and
    "slightly above zero" swamps everything retention can say. Fixing one
    claim and destroying five hundred outranks doing nothing.

    This is not hypothetical: it is why the pooled statistic scored the
    always-correct arm ABOVE vanilla at n=300 while the pre-registered
    per-turn test found it significantly worse. The metric is not being
    changed -- it is pre-registered and the results are already in -- but
    every report flags the regime, and the raw counts (errors removed, claims
    broken) are printed beside it because those stay interpretable.
    """
    g = cells.get
    unsupported = g("false_keep", 0) + g("true_fix", 0) + g("true_drop", 0)
    if not unsupported:
        return True
    removal = (g("true_fix", 0) + g("true_drop", 0)) / unsupported
    return removal < NEAR_ZERO_REMOVAL


def _rates(cells: dict[str, int]) -> tuple[float, float, float]:
    g = cells.get
    unsupported = g("false_keep", 0) + g("true_fix", 0) + g("true_drop", 0)
    supported = g("true_keep", 0) + g("false_fix", 0) + g("false_drop", 0)

    removal = (
        (g("true_fix", 0) + g("true_drop", 0)) / unsupported if unsupported else 1.0
    )
    retention = g("true_keep", 0) / supported if supported else 1.0

    if removal + retention == 0:
        net = 0.0
    else:
        net = 2 * removal * retention / (removal + retention)
    return removal, retention, net


def score_arm(
    arm: str,
    turns: Sequence[TurnResult],
    gold: Optional[dict[str, bool]] = None,
    trust_post_rewrite: bool = True,
) -> ArmScore:
    """Score one arm over its turns. See `claim_fates` for trust_post_rewrite."""
    totals = {c.value: 0 for c in Cell}
    per_turn, per_turn_tokens, per_turn_cells = [], [], []
    n_claims = 0

    n_skipped = 0
    for t in turns:
        fates = claim_fates(t, gold, trust_post_rewrite)
        if gold is not None:
            n_skipped += len({v.claim_id for v in t.verdicts}) - len(fates)
        n_claims += len(fates)
        turn_cells = {c.value: 0 for c in Cell}
        for f in fates:
            totals[f.cell.value] += 1
            turn_cells[f.cell.value] += 1
        per_turn.append(_rates(turn_cells)[2])
        per_turn_cells.append(turn_cells)
        per_turn_tokens.append(sum(c.total_tokens for c in t.costs))

    removal, retention, net = _rates(totals)
    n = max(1, len(turns))
    return ArmScore(
        arm=arm,
        n_turns=len(turns),
        n_claims=n_claims,
        cells=totals,
        error_removal=removal,
        retention=retention,
        net_quality=net,
        tokens_per_turn=sum(per_turn_tokens) / n,
        latency_per_turn=sum(
            t.wall_clock_s or sum(c.latency_s for c in t.costs) for t in turns
        ) / n,
        gold_source=(
            ("annotated" if trust_post_rewrite else "annotated_rewrite_neutral")
            if gold else "detector_on_original"
        ),
        n_unscorable=n_skipped,
        per_turn=per_turn,
        per_turn_tokens=[float(x) for x in per_turn_tokens],
        per_turn_cells=per_turn_cells,
    )


# --------------------------------------------------------------------------
# Head-to-head
# --------------------------------------------------------------------------


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 20260920,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap on (a - b) per turn.

    Arms are run on the SAME examples, so the per-turn differences are paired.
    Pairing removes between-example variance, which on a few hundred turns is
    the difference between a detectable effect and noise. Returns the mean
    difference, a percentile CI, and a two-sided p-value by sign-flipping.
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length sequences")
    n = len(a)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan")}

    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]

    # Two-sided permutation p-value: under the null the sign of each paired
    # difference is arbitrary.
    rng2 = random.Random(seed + 1)
    extreme = 0
    for _ in range(n_resamples):
        s = sum(d if rng2.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= abs(observed):
            extreme += 1
    p = (extreme + 1) / (n_resamples + 1)

    return {
        "n": n,
        "mean_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "significant": bool(lo > 0 or hi < 0),
    }



def cluster_bootstrap_pooled(
    cells_a: Sequence[dict[str, int]],
    cells_b: Sequence[dict[str, int]],
    n_resamples: int = 10000,
    seed: int = 20260920,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap on the POOLED net quality, resampling turns.

    **Why this exists, added 2026-10-03 after seeing p=0.0858.** The reported
    headline is the pooled net quality -- every claim from every turn in one
    ledger. The pre-registered significance test was a paired bootstrap over
    PER-TURN net quality. Those are different statistics, and on RAGTruth they
    disagree in direction:

        pooled      rule 0.749   always 0.000   (rule far ahead)
        per-turn    rule - always = -0.070      (rule slightly behind)

    The cause is not noise, it is `_rates`: a turn containing no unsupported
    claims has no error-removal to measure, so removal defaults to 1.0. Most
    RAGTruth turns are clean, so on most turns an arm that changes nothing
    scores a perfect 1.0 and any arm that drops a claim scores less. The
    per-turn average therefore rewards inaction on clean turns and barely sees
    the turns that contain the errors -- it is measuring something other than
    the quantity in the table.

    This resamples turns with replacement and recomputes the pooled rates on
    each resample, which is the standard cluster bootstrap for a ratio
    statistic with claims nested in turns. It is the CI for the number
    actually reported.

    **The pre-registered per-turn test is NOT replaced.** It was fixed in
    advance, this one was written after seeing its p-value, and a reader is
    entitled to know that. `compare_arms` returns both, and the verdict still
    follows the pre-registered one. See the DECISIONS.md change log.
    """
    if len(cells_a) != len(cells_b):
        raise ValueError("cluster bootstrap needs equal-length sequences")
    n = len(cells_a)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan")}

    def pooled(cells: Sequence[dict[str, int]], idx: Sequence[int]) -> float:
        total: dict[str, int] = {}
        for i in idx:
            for k, v in cells[i].items():
                total[k] = total.get(k, 0) + v
        return _rates(total)[2]

    allidx = list(range(n))
    observed = pooled(cells_a, allidx) - pooled(cells_b, allidx)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(pooled(cells_a, idx) - pooled(cells_b, idx))
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_resamples)]
    hi = diffs[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]

    # Two-sided: how often does a resample fall on the other side of zero?
    # (A sign-flip permutation is not available here -- the statistic is not a
    # mean of paired differences -- so this is the CI-based test.)
    below = sum(1 for d in diffs if d <= 0.0)
    p = 2.0 * min(below, n_resamples - below) / n_resamples
    p = min(1.0, p)

    return {
        "n": n,
        "statistic": "pooled_net_quality",
        "mean_diff": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(p, 4),
        "significant": bool(lo > 0 or hi < 0),
        "preregistered": False,
    }

def compare_arms(
    ours: ArmScore, baseline: ArmScore, n_resamples: int = 10000
) -> dict:
    """The head-to-head verdict, and the honest verdict is often 'no claim'.

    Reports quality and cost deltas with paired CIs, then applies a decision
    rule that refuses to call a win on a quality difference that is not
    significant.
    """
    if ours.n_turns != baseline.n_turns:
        raise ValueError(
            f"arms must cover the same turns: {ours.arm} has {ours.n_turns}, "
            f"{baseline.arm} has {baseline.n_turns}"
        )

    quality = paired_bootstrap(ours.per_turn, baseline.per_turn, n_resamples)
    quality["statistic"] = "per_turn_net_quality"
    quality["preregistered"] = True
    # The CI for the number actually printed in the table. Reported alongside,
    # never instead of -- see cluster_bootstrap_pooled's docstring.
    quality_pooled = cluster_bootstrap_pooled(
        ours.per_turn_cells, baseline.per_turn_cells, n_resamples
    )
    cost = paired_bootstrap(
        ours.per_turn_tokens, baseline.per_turn_tokens, n_resamples
    )

    q_better = quality.get("significant") and quality["mean_diff"] > 0
    q_worse = quality.get("significant") and quality["mean_diff"] < 0
    cheaper = cost.get("significant") and cost["mean_diff"] < 0

    saving = (
        (baseline.tokens_per_turn - ours.tokens_per_turn) / baseline.tokens_per_turn
        if baseline.tokens_per_turn
        else 0.0
    )

    if q_worse:
        verdict = "loses on quality"
        claim = (
            f"{ours.arm} scores significantly LOWER quality than {baseline.arm}. "
            "No efficiency argument rescues this; report it."
        )
    elif q_better and cheaper:
        verdict = "dominates"
        claim = (
            f"{ours.arm} beats {baseline.arm} on quality AND cost -- Pareto "
            "dominance, the strongest available result."
        )
    elif q_better:
        verdict = "better quality, not cheaper"
        claim = f"{ours.arm} improves quality over {baseline.arm} at comparable cost."
    elif cheaper:
        verdict = "equal quality, cheaper"
        claim = (
            f"{ours.arm} matches {baseline.arm} on quality "
            f"(difference not significant, 95% CI "
            f"[{quality['ci_low']:+.4f}, {quality['ci_high']:+.4f}]) "
            f"while using {saving:.0%} fewer tokens. This is the efficiency claim."
        )
    else:
        verdict = "no detectable difference"
        claim = (
            f"No significant difference between {ours.arm} and {baseline.arm} on "
            "either quality or cost. Do not claim a win."
        )

    # The pre-registered metric and the raw count encode different value
    # judgements, and when they point opposite ways saying only the flattering
    # one would be a choice, not a report.
    #
    #   net_quality  -- harmonic mean of two RATES. Removing 73 of 125 errors
    #                   is 58% of a small denominator; dropping 274 of 1964
    #                   good claims is 14% of a large one. It therefore values
    #                   a removed error well above a kept claim.
    #   net_claims   -- raw difference. Values them exactly 1:1.
    #
    # Neither is objectively right: the exchange rate between "a hallucination
    # survived" and "a true statement was withheld" is a property of the
    # application. So both are returned and a conflict is named.
    net_claims_conflict = (
        verdict in ("dominates", "better quality, not cheaper", "equal quality, cheaper")
        and ours.net_claims_improved < baseline.net_claims_improved
    )
    if net_claims_conflict:
        claim += (
            f" NOTE: on raw claim counts {ours.arm} is WORSE "
            f"({ours.net_claims_improved} vs {baseline.net_claims_improved} net "
            f"claims improved). net_quality weighs rates, so removing "
            f"{ours.cells.get('true_fix', 0) + ours.cells.get('true_drop', 0)} "
            f"errors out of a small pool outweighs withholding "
            f"{ours.cells.get('false_drop', 0) + ours.cells.get('false_fix', 0)} "
            f"correct claims out of a large one. Whether that is the right "
            f"exchange rate is an application question, not a measurement one. "
            f"Report both."
        )

    return {
        "ours": ours.arm,
        "baseline": baseline.arm,
        "verdict": verdict,
        "claim": claim,
        "net_claims_ours": ours.net_claims_improved,
        "net_claims_baseline": baseline.net_claims_improved,
        "net_claims_conflict": net_claims_conflict,
        "quality_delta": quality,
        "quality_delta_pooled": quality_pooled,
        "cost_delta": cost,
        "token_saving_frac": round(saving, 4),
        "over_correction_ours": ours.cells.get("false_fix", 0),
        "over_correction_baseline": baseline.cells.get("false_fix", 0),
        "info_loss_ours": ours.cells.get("false_drop", 0),
        "info_loss_baseline": baseline.cells.get("false_drop", 0),
        "gold_source": ours.gold_source,
    }


def pareto_front(scores: Sequence[ArmScore]) -> list[str]:
    """Arms not dominated on both quality (higher) and tokens (lower)."""
    front = []
    for s in scores:
        dominated = any(
            o is not s
            and o.net_quality >= s.net_quality
            and o.tokens_per_turn <= s.tokens_per_turn
            and (o.net_quality > s.net_quality or o.tokens_per_turn < s.tokens_per_turn)
            for o in scores
        )
        if not dominated:
            front.append(s.arm)
    return front


def format_ledger(score: ArmScore) -> str:
    """The six-cell table for one arm, as markdown."""
    c = score.cells
    rows = [
        ("was supported",   c["true_keep"],  c["false_fix"],  c["false_drop"]),
        ("was unsupported", c["false_keep"], c["true_fix"],   c["true_drop"]),
    ]
    out = [
        f"**{score.arm}** — {score.n_claims} claims over {score.n_turns} turns",
        "",
        "| | kept | rewritten | dropped |",
        "|---|---|---|---|",
    ]
    for label, keep, rew, drop in rows:
        out.append(f"| {label} | {keep} | {rew} | {drop} |")
    out += [
        "",
        f"- error removal : {score.error_removal:.3f}",
        f"- retention     : {score.retention:.3f}",
        f"- **net quality**: **{score.net_quality:.3f}**",
        f"- over-correction (supported claim broken by a rewrite): {c['false_fix']}",
        f"- information loss (supported claim dropped): {c['false_drop']}",
        f"- missed (unsupported claim left alone): {c['false_keep']}",
    ]
    return "\n".join(out)
