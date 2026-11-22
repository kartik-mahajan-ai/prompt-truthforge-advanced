"""Claim splitting: answer text -> list of standalone, checkable claims.

Gate 2 in DECISIONS.md points here. If the detector cannot beat a trivial
baseline, the splitter is the first suspect, because every downstream stage
inherits its errors:

  * under-splitting -> a sentence with one bad fact and three good ones gets
    one verdict, and any rewrite risks destroying the good facts (this is
    exactly the `info_loss` failure mode the taxonomy measures);
  * over-splitting  -> fragments lose their referents, the detector marks them
    unsupported, and the over-correction rate is inflated by an artifact.

Two implementations:
  * `RuleSplitter`  -- sentence segmentation + pronoun resolution heuristics.
                       No model, no GPU, deterministic. The baseline.
  * `LLMSplitter`   -- prompts the generator model for atomic decontextualised
                       claims. Better, costs tokens (which the ledger records).
"""

from __future__ import annotations

import re
from typing import Optional, Protocol

from ..types import Claim

# Abbreviations that must not end a sentence. Checked in Python rather than in
# the regex: variable-width lookbehind is not supported by `re`.
_ABBREV = frozenset(
    """mr mrs ms dr prof sr jr st mt rev hon gen col sgt capt lt inc ltd co
    corp dept univ ave blvd no vs etc al fig eq ch pp cf approx est i.e e.g
    u.s u.k a.m p.m""".split()
)

_SENT_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])         # after terminal punctuation
    ["')\]]*           # allow trailing quotes/brackets
    \s+                # whitespace
    (?=[A-Z0-9"'(\[])  # next sentence starts with a capital, digit or quote
    """,
    re.VERBOSE,
)

# A bullet or numbered-list marker at the start of a line.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•·●▪]|\d+[.)]|[a-z][.)])\s+")


def _line_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every line, offsets relative to `text`."""
    spans, cursor = [], 0
    while cursor <= len(text):
        nl = text.find("\n", cursor)
        if nl == -1:
            spans.append((cursor, len(text)))
            break
        spans.append((cursor, nl))
        cursor = nl + 1
    return [(s, e) for s, e in spans if e > s]


# The token immediately before a candidate boundary.
_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.[\"')\]]*\s*$")

# Leading discourse markers that carry no checkable content.
_DISCOURSE = re.compile(
    r"^\s*(?:however|moreover|furthermore|additionally|also|in addition|"
    r"therefore|thus|consequently|that said|on the other hand|finally|"
    r"overall|in summary|in conclusion)\s*,?\s*",
    re.IGNORECASE,
)

# A claim must assert something. These are pure meta-commentary.
_NON_CLAIM = re.compile(
    r"^\s*(?:based on the (?:provided |retrieved )?(?:context|passages?|documents?|sources?)"
    r"|according to the (?:context|passages?|documents?|sources?)"
    r"|here (?:is|are)\b|sure[,!]|certainly[,!]|i hope this helps)",
    re.IGNORECASE,
)

_PRONOUNS = frozenset(
    """he she it they them this that these those his her its their him we us
    our you your i me my one ones there""".split()
)

_PRONOUN_START = re.compile(
    r"^\s*(?:he|she|it|they|this|that|these|those|his|her|its|their)\b",
    re.IGNORECASE,
)


class Splitter(Protocol):
    name: str

    def split(self, answer: str) -> list[Claim]: ...


class RuleSplitter:
    """Deterministic sentence-level splitter with light decontextualisation.

    Keeps character offsets into the original answer, which is what lets the
    rewriter splice a corrected claim back in without regenerating the whole
    answer, and what lets eval score claims against RAGTruth's word spans.
    """

    name = "rule"

    def __init__(
        self,
        min_chars: int = 2,
        min_words: int = 2,
        drop_non_claims: bool = True,
        resolve_pronouns: bool = True,
    ) -> None:
        # Filter on CONTENT, not length. min_chars was 12, which silently
        # deleted "He danced." -- a real, checkable, and in that instance
        # false claim -- before the fact-checker ever saw it. A length cutoff
        # cannot tell a short claim from a formatting fragment.
        #
        # min_words counts alphabetic tokens of 2+ letters, which keeps
        # "He danced." (2) and still drops "1." (0) and "(Passage 3)" (1).
        self.min_chars = min_chars
        self.min_words = min_words
        self.drop_non_claims = drop_non_claims
        self.resolve_pronouns = resolve_pronouns

    def split(self, answer: str) -> list[Claim]:
        claims: list[Claim] = []
        last_subject: Optional[str] = None

        for idx, (start, end) in enumerate(self._sentence_spans(answer)):
            raw = answer[start:end]
            stripped = raw.strip()
            if len(stripped) < self.min_chars or not _is_claimlike(
                stripped, self.min_words
            ):
                continue
            if self.drop_non_claims and _NON_CLAIM.match(stripped):
                continue

            text = _DISCOURSE.sub("", stripped).strip()
            if not text:
                continue

            resolved = text
            if self.resolve_pronouns and last_subject and _PRONOUN_START.match(text):
                # Prefix rather than substitute: the detector needs the
                # referent, and prefixing cannot corrupt the original wording.
                resolved = f"({last_subject}) {text}"

            # Trim the span to the stripped text on BOTH sides. Leaving
            # trailing whitespace inside the span makes `splice` swallow the
            # separator between sentences.
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            # A leading "* " or "3. " is formatting, not content. Drop it from
            # the claim text AND advance the offset, so the span still indexes
            # exactly the text the claim represents.
            marker = _LIST_MARKER.match(raw[lead:])
            if marker:
                lead += marker.end()
                stripped = raw[lead : len(raw) - trail]
                text = _DISCOURSE.sub("", stripped).strip()
                resolved = text
                if self.resolve_pronouns and last_subject and _PRONOUN_START.match(text):
                    resolved = f"({last_subject}) {text}"
                if not text or not _is_claimlike(stripped, self.min_words):
                    continue
            claims.append(
                Claim(
                    text=resolved,
                    start=start + lead,
                    end=end - trail,
                    sent_index=idx,
                    meta={
                        "surface": stripped,
                        "decontextualised": resolved != text,
                        "splitter": self.name,
                    },
                )
            )
            subject = _leading_noun_phrase(text)
            if subject:
                last_subject = subject

        return claims

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Split into claim spans, honouring BOTH line breaks and sentences.

        Lines come first. Half of RAGTruth's QA answers contain newlines and a
        third are bulleted lists, whose items carry no terminal punctuation:

            * Diarrhea (often violent episodes)
            * Constipation
            * Belly pains or cramps after meals

        Sentence-boundary splitting alone glues that entire list into one
        claim -- measured at up to 154 words -- so a single verdict covers one
        bad fact and a dozen good ones. That is the under-splitting failure
        Gate 1 exists to catch, and it was producing 32% "mixed" claims.
        """
        if not text.strip():
            return []

        spans: list[tuple[int, int]] = []
        for line_start, line_end in _line_spans(text):
            line = text[line_start:line_end]
            if not line.strip():
                continue
            for s, e in self._sentences_within(line):
                spans.append((line_start + s, line_start + e))
        return spans

    def _sentences_within(self, line: str) -> list[tuple[int, int]]:
        """Sentence spans inside one line, relative to that line."""
        spans, cursor = [], 0
        for m in _SENT_BOUNDARY.finditer(line):
            if _is_abbreviation_boundary(line, m.start()):
                continue
            spans.append((cursor, m.start()))
            cursor = m.end()
        spans.append((cursor, len(line)))
        return [(s, e) for s, e in spans if e > s]


def _is_abbreviation_boundary(text: str, pos: int) -> bool:
    """True when the period at `pos` closes a known abbreviation, not a sentence."""
    m = _TRAILING_TOKEN.search(text[:pos])
    if not m:
        return False
    token = m.group(1).lower().rstrip(".")
    # Single initials ("J. R. R. Tolkien") are never sentence ends either.
    return token in _ABBREV or len(token) == 1


_ALPHA_TOKEN = re.compile(r"[A-Za-z]{2,}")

# Sentence terminators, for length checks only -- not for splitting.
_SENT_END = re.compile(r"[.!?](?:\s|$)")

# "(Passage 3)", "Passage 2:" -- a citation marker, not an assertion.
_CITATION_ONLY = re.compile(r"^[\s(\[]*passages?\s*\d*\s*[)\]:.]*$", re.IGNORECASE)

# A stock tail a refusal usually carries: "...based on the given passages".
_REFUSAL_TAIL = (
    r"(?:\s*(?:based (?:up)?on|from|in|with|using|given)"
    r"\s*(?:the\s*)?(?:given|provided|available|supplied|above)?"
    r"\s*(?:passages?|context|information|documents?|sources?|text)\s*)?"
)

# Whole-sentence refusals ONLY. Each pattern must match the entire sentence
# (minus that stock tail), never merely appear inside one.
#
# The looser version of this -- anything containing "does not mention" -- would
# also swallow "The passages do not mention that the tower is 500m tall", which
# smuggles a factual assertion inside refusal-shaped wording. Excluding that
# from verification is exactly the failure this filter must not introduce, so
# the patterns are anchored and the open-ended ones are left out.
_REFUSAL_ONLY = re.compile(
    r"^(?:"
    r"(?:i\s*(?:am|'m)?\s*)?unable to answer(?:\s*(?:the|this)?\s*question)?"
    r"|(?:i\s*)?(?:can\s*not|cannot|can't)\s*answer(?:\s*(?:the|this)?\s*question)?"
    r"|(?:there\s*is\s*)?(?:not enough|insufficient|no sufficient)"
    r"\s*(?:information|evidence|context|detail)"
    r"|no (?:relevant )?(?:information|evidence|answer)"
    r"\s*(?:is\s*)?(?:available|found|provided|given)?"
    r"|(?:the\s*)?answer (?:is not|isn't) (?:provided|available|given|present)"
    r")" + _REFUSAL_TAIL + r"[\s.!]*$",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    """Is this sentence a refusal rather than an assertion about the world?

    "Unable to answer based on the given passages" says something about the
    system's knowledge state, not about the evidence. Scoring it for support
    is a category error: the detector marks it unsupported (p ~ 0.02-0.07,
    measured), the router dutifully sends it to the corrector, and a 1.5B
    model is asked to make a refusal more faithful to passages it is refusing
    to use. On RAGTruth QA this accounted for 12 of 81 flags and a comparable
    share of wasted corrector calls.

    Excluding it does NOT delete it. Text covered by no claim stays in the
    answer untouched; it simply stops being routed.
    """
    return bool(_REFUSAL_ONLY.match(text.strip()))


# Broader than `_REFUSAL_ONLY`, and deliberately so -- see `answer_is_refusal`.
_REFUSAL_ANYWHERE = re.compile(
    r"(?:do(?:es)?\s*n[o']?t\s*(?:contain|provide|mention|include|specify|discuss|cover|have)"
    r"|no\s*(?:relevant\s*)?information"
    r"|not\s*(?:enough|sufficient)\s*information"
    r"|unable to answer|can\s*not answer|cannot answer|can't answer"
    r"|is not (?:provided|mentioned|specified|available|discussed))",
    re.IGNORECASE,
)


def answer_is_refusal(text: str, max_sentences: int = 3) -> bool:
    """Did the system decline to answer, in whatever words?

    This is a different question from `is_refusal`, and the two must not be
    merged even though they look alike.

    `is_refusal` decides whether one sentence should be ROUTED. Getting that
    wrong in the permissive direction exempts a factual assertion from
    verification -- "The passages do not mention that the tower is 500m tall"
    would stop being checked -- so it is anchored to whole-sentence templates
    and refuses to match anything with content hanging off it.

    This decides whether a whole ANSWER counts as a refusal when scoring
    refusal rates. Getting that wrong in the strict direction is what actually
    happened: the open-corpus evaluation counted only router-driven abstention
    and scored every arm at 0.0 correct refusals, when the generator was in
    fact declining on 56% of unanswerable questions in its own words
    ("The passages do not contain information about oil stains on concrete
    driveways"). The measurement said the system never refuses; it refuses
    often, and the instrument could not see it.

    So this one matches the phrase anywhere, and guards against swallowing a
    real answer by requiring the reply to be SHORT. A three-sentence reply
    whose first clause is "the passages do not mention X" is a decline; a
    twelve-sentence essay containing that phrase in passing is not.
    """
    t = (text or "").strip()
    if not t:
        return True
    if len(_SENT_END.findall(t)) > max_sentences:
        return False
    return bool(_REFUSAL_ANYWHERE.search(t))


def _is_claimlike(text: str, min_words: int) -> bool:
    """Does this span assert something, or is it formatting or a refusal?

    Replaces a character-count cutoff, which could not distinguish a short
    claim from a list marker and silently dropped the former.
    """
    stripped = text.strip()
    if _CITATION_ONLY.match(stripped):
        return False
    if is_refusal(stripped):
        return False
    return len(_ALPHA_TOKEN.findall(text)) >= min_words


# Closed-class words, and verbs that commonly open an instruction. A sentence
# beginning with one of these is capitalised because it BEGINS A SENTENCE, not
# because it names anything. Nouns are open-class and cannot be listed, so
# they are accepted by default and these are rejected.
_NOT_A_SUBJECT = frozenset("""
    a an the this that these those some any each every another such no
    many most several few all both either neither
    in on at by for from with without about above below under over into onto
    of to through during before after since until while within between among
    across against along around behind beyond despite like near per via
    according based depending given regarding following including
    and or but nor so yet if then else when where whether because although
    though unless once also however therefore thus hence moreover furthermore
    meanwhile otherwise instead next first second third finally lastly
    additionally overall generally typically usually often sometimes
    here there now today yes not only just even still
    use add attach click select open close remove place put set make take
    turn cook bake roast grill boil heat preheat mix stir pour cut chop
    wash rinse dry apply check ensure keep let try avoid start stop begin
    seeking finding using making taking having being getting doing
    step steps passage passages note notes answer summary conclusion
    it's its i'm you're we're they're there's that's what's here's
    how what why who whom whose which can could should would will shall may
    might must do does did is are was were be been
""".split())

_DETERMINERS = frozenset("the a an this that these those".split())

# What follows an instruction verb: its object or an infinitive. "Rub the
# meat", "Dissolve a teaspoon", "Remember to", "Pry your lid". A noun subject
# is followed by a verb instead -- "Chard is", "Magellan sailed". Listing every
# English imperative is impossible; this one syntactic cue covers the open set.
_OBJECT_OPENERS = frozenset(
    "the a an to your my our their his her its this that these those it them "
    "each every some any all both off out up down in on into onto over".split()
)

# "Passage 3", "Step 4", "Section 2" -- a citation or a list label, not a
# referent. Prefixing a claim with "(Passage 3)" tells the detector nothing.
_LABEL_PHRASE = re.compile(
    r"^(?:passage|step|section|part|item|option|source)\s*\d+$", re.IGNORECASE
)


def _leading_noun_phrase(sentence: str, max_words: int = 5) -> Optional[str]:
    """The named subject of a sentence, or None when there is not a clear one.

    **This was wrong, and the human annotation found it.** The first version
    took any capitalised run at the start of a sentence as the subject and
    rejected only pronouns. But English capitalises the first word of EVERY
    sentence, so "The tower was completed in 1889" yielded the subject "The",
    and the next sentence -- "It stands 330 metres tall" -- became the claim
    "(The) It stands 330 metres tall". Across 300 RAGTruth answers, 6.5% of
    claims carried such a prefix and about two-thirds were garbage: "Based",
    "The", "In", "If", "A", "Then", "Use", "Passage 3".

    The damage was not cosmetic. Garbled claims went to the detector, scored
    low, and were sent to the corrector, which spent its call deleting the
    artifact. Prefixed claims were 4x over-represented among rewrites (26% of
    text-changing rewrites against 6.5% of claims), and the detector then
    scored the cleaned text differently -- so the ledger recorded clean-up as
    damage. A human labelling 30 of those "damaging" rewrites judged 2 harmful
    and 17 actual improvements.

    The rule now: a wrong prefix is worse than no prefix, so when in doubt
    return None. That is cheap, because the NLI detector already resolves
    coreference by keeping whole passages in its premise set -- the prefix was
    never its only defence. Specifically:

      * a leading closed-class word, instruction verb or gerund opener is not
        a subject ("The", "Based", "In", "Use", "Seeking")
      * a leading determiner IS allowed when a capitalised name follows it,
        and the name is taken without it ("The Eiffel Tower" -> "Eiffel Tower")
      * a citation or list label is not a referent ("Passage 3", "Step 4")
      * pronouns are rejected, as before -- they are what needs resolving
    """
    cores = [w.strip(".,;:\"'()[]!?") for w in sentence.split()[: max_words + 1]]
    cores = [c for c in cores if c]
    if not cores:
        return None

    start = 0
    first = cores[0].lower()
    if first in _PRONOUNS:
        return None
    if first in _NOT_A_SUBJECT:
        # A determiner may introduce a proper name: "The Eiffel Tower ...".
        # Anything else in the closed class introduces no subject at all.
        if first in _DETERMINERS and len(cores) > 1 and cores[1][:1].isupper():
            start = 1
        else:
            return None

    picked: list[str] = []
    for c in cores[start : start + max_words]:
        # Stop at a closed-class word even when capitalised: "Smiling If you
        # ..." is a subject running into a new clause, not a two-word name.
        if picked and c.lower() in _NOT_A_SUBJECT:
            break
        if c[:1].isupper() or c.isdigit():
            picked.append(c)
        else:
            break
    if not picked:
        return None
    head = picked[0].lower()
    if head in _PRONOUNS or head in _NOT_A_SUBJECT:
        return None
    # "-ly" openers are adverbs: "Lightly", "Similarly", "Proportionately".
    if len(picked) == 1 and head.endswith("ly") and len(head) > 4:
        return None
    # A single word followed by an object is an instruction, not a subject.
    # Multi-word capitalised runs ("Eiffel Tower", "New York City") and
    # acronyms ("DNA") are names whatever follows them.
    nxt = start + len(picked)
    if (
        len(picked) == 1
        and not picked[0].isupper()
        and nxt < len(cores)
        and cores[nxt].lower() in _OBJECT_OPENERS
    ):
        return None

    phrase = " ".join(picked)
    if _LABEL_PHRASE.match(phrase):
        return None
    return phrase


CLAIM_SPLIT_PROMPT = """\
Break the ANSWER into standalone factual claims.

Rules:
- One claim per line, no numbering, no bullets.
- Each claim must be understandable on its own: replace every pronoun and \
definite reference with the full name it refers to.
- Copy the wording of the answer where you can. Do not add facts, do not \
infer, do not merge claims.
- Skip pure meta-commentary ("Based on the context...", "I hope this helps").
- If the answer contains no factual claim, output NONE.

ANSWER:
{answer}

CLAIMS:"""


class LLMSplitter:
    """Prompt-based splitter. Loses character offsets; keeps fidelity.

    Offsets are recovered by exact-substring search where possible and left
    None otherwise -- downstream code must handle `start is None`.
    """

    name = "llm"

    def __init__(self, generate_fn, min_chars: int = 12) -> None:
        """Args:
        generate_fn: callable(prompt: str) -> str. Kept as a plain callable so
            this module never imports torch.
        """
        self.generate = generate_fn
        self.min_chars = min_chars
        self.fallback = RuleSplitter(min_chars=min_chars)

    def split(self, answer: str) -> list[Claim]:
        raw = self.generate(CLAIM_SPLIT_PROMPT.format(answer=answer)) or ""
        lines = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip()
            for ln in raw.splitlines()
        ]
        lines = [ln for ln in lines if len(ln) >= self.min_chars and ln.upper() != "NONE"]

        if not lines:
            # An empty or malformed generation must not silently drop the
            # answer from evaluation.
            return self.fallback.split(answer)

        claims = []
        for i, line in enumerate(lines):
            pos = answer.find(line)
            claims.append(
                Claim(
                    text=line,
                    start=pos if pos >= 0 else None,
                    end=pos + len(line) if pos >= 0 else None,
                    sent_index=i,
                    meta={"splitter": self.name, "offset_recovered": pos >= 0},
                )
            )
        return claims


def get_splitter(name: str = "rule", generate_fn=None) -> Splitter:
    if name == "rule":
        return RuleSplitter()
    if name == "llm":
        if generate_fn is None:
            raise ValueError("LLMSplitter needs a generate_fn")
        return LLMSplitter(generate_fn)
    raise ValueError(f"unknown splitter: {name!r}")
