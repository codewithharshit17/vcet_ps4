"""[5b] Aggressive word-level stripping -- OFF by default, measured separately.

Both deck diagrams put "prune redundant words" in the main path. It is built
here with its own toggle instead, because removing modifiers is genuinely
dangerous: "non-critical failure", "unapproved vendor" and "third-quarter
revenue" all invert or lose meaning when the modifier goes.

The job of this module is to give the eval harness something to measure, not to
assume the answer. Guardrails: never touch structured content, never remove a
negation, never remove a token adjacent to a numeral.
"""

from __future__ import annotations

import re

from .score import ScoredSentence
from .sentences import Kind
from .timing import Timeline
from .tokens import get_counter

# Discourse filler that carries no propositional content.
FILLER = {
    "actually", "basically", "certainly", "essentially", "generally", "however",
    "indeed", "moreover", "nevertheless", "particularly", "perhaps", "quite",
    "rather", "really", "relatively", "significantly", "simply", "specifically",
    "substantially", "therefore", "thus", "typically", "usually", "very",
    "furthermore", "additionally", "accordingly", "consequently", "notably",
}

# Never removable: these flip or scope the meaning of a clause.
NEGATIONS = {
    "no", "not", "never", "none", "nor", "neither", "without", "cannot",
    "except", "unless", "excluding", "less", "fewer", "non", "un", "any",
}

_TOKEN = re.compile(r"\w+|\W+")
_HAS_DIGIT = re.compile(r"\d")


def strip_sentence(text: str) -> str:
    parts = _TOKEN.findall(text)
    words = [(i, p) for i, p in enumerate(parts) if p.isalpha()]

    remove: set[int] = set()
    for pos, (i, word) in enumerate(words):
        low = word.lower()
        if low not in FILLER or low in NEGATIONS:
            continue
        # Refuse to touch anything sitting next to a number or a negation.
        neighbours = [words[pos - 1][1] if pos > 0 else "", words[pos + 1][1] if pos + 1 < len(words) else ""]
        if any(_HAS_DIGIT.search(n) or n.lower() in NEGATIONS for n in neighbours):
            continue
        remove.add(i)

    if not remove:
        return text
    out = "".join(p for i, p in enumerate(parts) if i not in remove)
    return re.sub(r"\s{2,}", " ", out).strip()


def apply_strip(
    selected: list[ScoredSentence], timeline: Timeline | None = None
) -> tuple[list[str], int]:
    """Return (stripped_texts, tokens_saved). Structured content is untouched."""

    def _run() -> tuple[list[str], int]:
        counter = get_counter()
        before = sum(s.token_count for s in selected)
        texts: list[str] = []
        for s in selected:
            if s.sentence.kind is not Kind.PROSE:
                texts.append(s.sentence.text)
            else:
                texts.append(strip_sentence(s.sentence.text))
        after = sum(counter.count_many(texts)) if texts else 0
        return texts, before - after

    if timeline is not None:
        with timeline.span("strip_ms"):
            return _run()
    return _run()
