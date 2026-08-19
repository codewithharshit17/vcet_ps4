"""Corpus -> chunks -> Qdrant.

Chunking is ~400 tokens with ~15% overlap, respecting paragraph boundaries, and
never splitting a [TABLE]...[/TABLE] block. Every chunk keeps ``doc_id``,
``chunk_id`` and ``char_offset`` so provenance survives all the way to the
citation on the final answer.

Token counts come from the real tokenizer (tokens.py), not len(split()).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qdrant_client.models import PointStruct

from .config import CORPUS_DIR, settings
from .corpus.fetch_sec import TABLE_CLOSE, TABLE_OPEN
from .store import Chunk, ensure_collection, get_client, reset_storage
from .tokens import get_counter

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class Block:
    """A paragraph or an atomic table, with its offset in the source document."""

    text: str
    char_offset: int
    is_table: bool


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def split_blocks(text: str) -> list[Block]:
    """Split into paragraphs, keeping table blocks whole."""
    blocks: list[Block] = []
    pos = 0
    # Tables first, so their internal blank lines never split them.
    pattern = re.compile(
        re.escape(TABLE_OPEN) + r".*?" + re.escape(TABLE_CLOSE), re.DOTALL
    )
    for m in pattern.finditer(text):
        _emit_paragraphs(text[pos : m.start()], pos, blocks)
        blocks.append(Block(text=m.group(0), char_offset=m.start(), is_table=True))
        pos = m.end()
    _emit_paragraphs(text[pos:], pos, blocks)
    return blocks


def _emit_paragraphs(segment: str, base_offset: int, out: list[Block]) -> None:
    cursor = 0
    for para in re.split(r"\n\s*\n", segment):
        start = segment.find(para, cursor)
        if start == -1:
            start = cursor
        cursor = start + len(para)
        stripped = para.strip()
        if stripped:
            out.append(
                Block(text=stripped, char_offset=base_offset + start, is_table=False)
            )


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    counter = get_counter()
    target = settings.chunk_target_tokens
    overlap_target = int(target * settings.chunk_overlap_ratio)

    blocks = split_blocks(text)
    if not blocks:
        return []
    block_tokens = counter.count_many([b.text for b in blocks])

    chunks: list[Chunk] = []
    cur: list[Block] = []
    cur_tokens = 0
    idx = 0

    def flush() -> None:
        nonlocal cur, cur_tokens, idx
        if not cur:
            return
        body = "\n\n".join(b.text for b in cur)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=idx,
                char_offset=cur[0].char_offset,
                text=body,
                token_count=counter.count(body),
            )
        )
        idx += 1

    for block, ntok in zip(blocks, block_tokens):
        # A single oversized block (usually a big table) becomes its own chunk
        # rather than being split mid-structure.
        if ntok >= target:
            flush()
            cur, cur_tokens = [block], ntok
            flush()
            cur, cur_tokens = [], 0
            continue

        if cur_tokens + ntok > target and cur:
            flush()
            # Carry trailing blocks back as overlap.
            carry: list[Block] = []
            carried = 0
            for b, t in zip(reversed(cur), reversed(block_tokens[: len(cur)])):
                if carried >= overlap_target:
                    break
                carry.insert(0, b)
                carried += t
            cur = list(carry)
            cur_tokens = carried

        cur.append(block)
        cur_tokens += ntok

    flush()
    return chunks


def ingest(recreate: bool = True, corpus_dir: Path | None = None) -> list[Chunk]:
    from sentence_transformers import SentenceTransformer

    src = corpus_dir or CORPUS_DIR
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise RuntimeError(
            f"No .txt/.md/.pdf files in {src}. Run `tokendiet fetch-corpus` first."
        )

    all_chunks: list[Chunk] = []
    for path in files:
        text = read_document(path)
        cs = chunk_document(path.stem, text)
        print(f"[ingest] {path.name}: {len(text):,} chars -> {len(cs)} chunks")
        all_chunks.extend(cs)

    if len(all_chunks) < settings.min_corpus_chunks:
        raise RuntimeError(
            f"Corpus produced only {len(all_chunks)} chunks; need >= "
            f"{settings.min_corpus_chunks} so retrieval is non-trivial. Add more documents."
        )

    print(f"[ingest] embedding {len(all_chunks)} chunks with {settings.embed_model} ...")
    model = SentenceTransformer(settings.embed_model)
    vectors = model.encode(
        [c.text for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if recreate:
        # Must happen before the client opens the storage directory.
        reset_storage()
    client = get_client()
    ensure_collection(client, recreate=recreate)
    points = [
        PointStruct(
            id=c.uid,
            vector=vec.tolist(),
            payload={
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "char_offset": c.char_offset,
                "text": c.text,
                "token_count": c.token_count,
            },
        )
        for c, vec in zip(all_chunks, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(settings.qdrant_collection, points=points[i : i + 256])

    # Duplicate chunks would inflate the baseline context and flatter every
    # compression number, so the indexed count must match exactly.
    indexed = client.count(settings.qdrant_collection, exact=True).count
    if indexed != len(points):
        raise RuntimeError(
            f"Indexed {indexed} points but upserted {len(points)}. "
            "The collection was not clean; refusing to proceed with duplicate chunks."
        )

    print(f"[ingest] upserted {len(points)} chunks into {settings.qdrant_collection}")
    client.close()
    return all_chunks
