"""[2] Sentence decomposition.

Provenance (doc_id, chunk_id, sentence_index) rides on every sentence so
citations survive to the final answer.

Two things here exist to stop naive compressors destroying answers:

* **Structured content is atomic.** Tables, code blocks and bullet items are
  never split mid-structure. A table row severed from its header is worse than
  useless -- it becomes unattributed numbers.
* **Short lines are not automatically junk.** Sentences below the minimum token
  count are dropped as headers/fragments *unless* they carry a numeral or a
  proper noun. "Net sales were $391.0 billion." is short and is the answer.

Retrieved chunks overlap by 15% at ingest, so the same sentence genuinely
appears in two chunks. We de-duplicate exactly, keeping the earliest position,
rather than leaving it for MMR -- an exact duplicate is a bookkeeping artefact
of our own chunking, not a property of the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import syntok.segmenter as segmenter

from .config import settings
from .corpus.fetch_sec import TABLE_CLOSE, TABLE_OPEN
from .retrieve import RetrievedChunk
from .timing import Timeline
from .tokens import get_counter


class Kind(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    CODE = "code"
    LIST = "list"


@dataclass
class Sentence:
    doc_id: str
    chunk_id: int
    sentence_index: int
    text: str
    char_offset: int  # absolute offset in the source document
    token_count: int
    kind: Kind

    @property
    def sid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}.{self.sentence_index}"

    @property
    def atomic(self) -> bool:
        return self.kind is not Kind.PROSE


_BULLET = re.compile(r"^\s*(?:[•●▪‣\-\*·]|\(?[a-z0-9]{1,3}[\.\)])\s+")
_CODE_FENCE = re.compile(r"^\s*(?:```|~~~)")
_HAS_DIGIT = re.compile(r"\d")
_HAS_ALPHA_WORD = re.compile(r"[A-Za-z]{2,}")
# A capitalised word that is not the first word of the line: a cheap proper-noun
# signal that avoids a POS-tagger dependency.
_PROPER_NOUN = re.compile(r"\S+\s+.*\b[A-Z][a-zA-Z]{2,}")


def _is_informative_short(text: str) -> bool:
    """Keep a short line if it carries a numeral or a proper noun.

    A numeral only counts when it sits alongside actual words: a bare "84" is a
    page number, not a fact. Without this, every page number and table-artifact
    digit in a 10-K survives the minimum-length filter and then competes for
    budget, because a 1-token sentence has unbeatable score-per-token.
    """
    has_word = bool(_HAS_ALPHA_WORD.search(text))
    if _HAS_DIGIT.search(text) and has_word:
        return True
    return has_word and bool(_PROPER_NOUN.search(text))


def _split_prose(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Sentence-split prose with syntok (handles abbreviations, decimals)."""
    out: list[tuple[str, int]] = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            toks = [t for t in sentence]
            if not toks:
                continue
            start = toks[0].offset
            last = toks[-1]
            end = last.offset + len(last.value)
            raw = text[start:end].strip()
            if raw:
                out.append((raw, base_offset + start))
    return out


def segment_chunk_text(text: str, base_offset: int) -> list[tuple[str, int, Kind]]:
    """Return (text, absolute_offset, kind) units for one chunk."""
    units: list[tuple[str, int, Kind]] = []
    pos = 0

    table_re = re.compile(re.escape(TABLE_OPEN) + r".*?" + re.escape(TABLE_CLOSE), re.DOTALL)
    for m in table_re.finditer(text):
        _segment_nontable(text[pos : m.start()], base_offset + pos, units)
        units.append((m.group(0).strip(), base_offset + m.start(), Kind.TABLE))
        pos = m.end()
    _segment_nontable(text[pos:], base_offset + pos, units)
    return units


def _segment_nontable(
    segment: str, base_offset: int, out: list[tuple[str, int, Kind]]
) -> None:
    if not segment.strip():
        return

    lines = segment.split("\n")
    buf: list[str] = []
    buf_start = 0
    cursor = 0
    in_code = False

    def flush_prose() -> None:
        nonlocal buf, buf_start
        if buf:
            blob = "\n".join(buf)
            if blob.strip():
                for s, off in _split_prose(blob, base_offset + buf_start):
                    out.append((s, off, Kind.PROSE))
            buf = []

    for line in lines:
        line_start = cursor
        cursor += len(line) + 1  # +1 for the newline we split on

        if _CODE_FENCE.match(line):
            flush_prose()
            in_code = not in_code
            out.append((line.strip(), base_offset + line_start, Kind.CODE))
            continue
        if in_code:
            out.append((line, base_offset + line_start, Kind.CODE))
            continue

        # Each bullet item is its own atomic unit: never split a bullet into
        # fragments, but still allow the budget to pick some bullets over others.
        if _BULLET.match(line) and line.strip():
            flush_prose()
            out.append((line.strip(), base_offset + line_start, Kind.LIST))
            continue

        if not buf:
            buf_start = line_start
        buf.append(line)

    flush_prose()


def decompose(
    hits: list[RetrievedChunk], timeline: Timeline | None = None
) -> list[Sentence]:
    """Retrieved chunks -> de-duplicated, provenance-carrying sentences."""

    def _run() -> list[Sentence]:
        counter = get_counter()
        raw: list[Sentence] = []

        for hit in hits:
            ch = hit.chunk
            units = segment_chunk_text(ch.text, ch.char_offset)
            texts = [u[0] for u in units]
            counts = counter.count_many(texts) if texts else []
            for idx, ((text, offset, kind), ntok) in enumerate(zip(units, counts)):
                if kind is Kind.PROSE:
                    if ntok < settings.min_sentence_tokens and not _is_informative_short(text):
                        continue
                elif not _HAS_ALPHA_WORD.search(text):
                    # Structured units skip the length filter because they are
                    # atomic, which let page numbers wrapped in table markup
                    # ("[TABLE] 22 [/TABLE]") through as high-density
                    # candidates. A table with no words is a layout artefact.
                    continue
                raw.append(
                    Sentence(
                        doc_id=ch.doc_id,
                        chunk_id=ch.chunk_id,
                        sentence_index=idx,
                        text=text,
                        char_offset=offset,
                        token_count=ntok,
                        kind=kind,
                    )
                )

        # Exact de-duplication from 15% chunk overlap: keep the earliest copy.
        seen: dict[str, Sentence] = {}
        for s in raw:
            key = re.sub(r"\s+", " ", s.text).strip().lower()
            prev = seen.get(key)
            if prev is None or (s.doc_id, s.char_offset) < (prev.doc_id, prev.char_offset):
                seen[key] = s
        return sorted(seen.values(), key=lambda s: (s.doc_id, s.char_offset, s.sentence_index))

    if timeline is not None:
        with timeline.span("sentence_split_ms"):
            return _run()
    return _run()
