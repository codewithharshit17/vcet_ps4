"""[6] Reassembly.

Surviving sentences are re-sorted into **original document order**, grouped by
source document. Handing the LLM a relevance-sorted jumble destroys narrative
coherence and measurably hurts answers -- the model loses the thread of which
clause qualifies which.

Elisions are marked with [...] so the model knows text was removed rather than
inferring that two adjacent sentences were adjacent in the source. Each group
carries its source identifier so the answer can cite it.
"""

from __future__ import annotations

from .score import ScoredSentence

ELISION = "[...]"


def assemble(selected: list[ScoredSentence]) -> str:
    """Source-ordered, cited, elision-marked context."""
    if not selected:
        return ""

    by_doc: dict[str, list[ScoredSentence]] = {}
    for s in selected:
        by_doc.setdefault(s.sentence.doc_id, []).append(s)

    blocks: list[str] = []
    for doc_id in sorted(by_doc):
        group = sorted(by_doc[doc_id], key=lambda s: s.sentence.char_offset)
        lines: list[str] = [f"[SOURCE: {doc_id}]"]

        prev_end: int | None = None
        for s in group:
            start = s.sentence.char_offset
            # A gap means intervening text was dropped; say so explicitly.
            if prev_end is not None and start > prev_end:
                lines.append(ELISION)
            lines.append(s.sentence.text)
            prev_end = start + len(s.sentence.text)

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def assemble_baseline(hits) -> str:
    """The uncompressed control: raw top-K chunks, same citation framing.

    Framing is identical to the compressed path so the A/B measures compression,
    not prompt-format differences.
    """
    blocks: list[str] = []
    for h in hits:
        blocks.append(f"[SOURCE: {h.chunk.doc_id}]\n{h.chunk.text}")
    return "\n\n".join(blocks)
