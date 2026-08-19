"""[1] Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

We deliberately over-fetch (K=20). Recall is cheap when compression sits
downstream -- that is the whole premise of the project, so the retriever is
tuned for recall and the compressor is trusted to cut the fat.

Dense search runs in Qdrant (the deck's choice) rather than a local matrix, so
we are not re-embedding the corpus on every query. BM25 is rebuilt in memory
from the same stored chunks, keeping Qdrant the single source of truth instead
of maintaining a parallel index that can drift out of sync.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import settings
from .store import Chunk, backend_name, get_client, iter_all_chunks
from .timing import Timeline, now_ns

_WORD = re.compile(r"[a-z0-9]+")


def tokenize_lexical(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rrf_score: float
    dense_rank: int | None  # None = absent from that ranker's pool, not "rank 0"
    bm25_rank: int | None

    @property
    def source(self) -> str:
        return f"{self.chunk.doc_id}#{self.chunk.chunk_id}"


@dataclass
class Retriever:
    chunks: list[Chunk] = field(default_factory=list)
    _index_by_uid: dict[int, int] = field(default_factory=dict)
    _bm25: object | None = None
    _embedder: object | None = None
    _client: object | None = None

    @classmethod
    def load(cls) -> "Retriever":
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        client = get_client()
        chunks = iter_all_chunks(client)
        if not chunks:
            raise RuntimeError("Index is empty. Run `tokendiet ingest` first.")

        r = cls(chunks=chunks, _client=client)
        r._index_by_uid = {c.uid: i for i, c in enumerate(chunks)}
        r._bm25 = BM25Okapi([tokenize_lexical(c.text) for c in chunks])
        r._embedder = SentenceTransformer(settings.embed_model)
        return r

    def warmup(self) -> float:
        """Force lazy model/ANN init so it is never billed to query latency."""
        t = now_ns()
        vec = self._embedder.encode(
            ["warmup"], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        self._client.query_points(
            collection_name=settings.qdrant_collection, query=vec.tolist(), limit=1
        )
        self._bm25.get_scores(tokenize_lexical("warmup"))
        return (now_ns() - t) / 1e6

    def retrieve(
        self, query: str, k: int | None = None, timeline: Timeline | None = None
    ) -> list[RetrievedChunk]:
        k = k or settings.top_k
        pool = max(50, k * 3)

        def _run() -> list[RetrievedChunk]:
            import numpy as np

            # --- dense (Qdrant, cosine over normalised vectors)
            qvec = self._embedder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0]
            hits = self._client.query_points(
                collection_name=settings.qdrant_collection,
                query=qvec.tolist(),
                limit=pool,
                with_payload=False,
            ).points
            dense_rank: dict[int, int] = {}
            for rank, hit in enumerate(hits, start=1):
                idx = self._index_by_uid.get(int(hit.id))
                if idx is not None:
                    dense_rank[idx] = rank

            # --- lexical (BM25)
            bm25_scores = self._bm25.get_scores(tokenize_lexical(query))
            bm25_rank = {
                int(idx): rank
                for rank, idx in enumerate(np.argsort(-bm25_scores)[:pool], start=1)
            }

            # --- Reciprocal Rank Fusion: score = sum 1 / (rrf_k + rank_i)
            fused: dict[int, float] = {}
            for ranks in (dense_rank, bm25_rank):
                for idx, rank in ranks.items():
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank)

            top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            return [
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    rrf_score=score,
                    dense_rank=dense_rank.get(idx),
                    bm25_rank=bm25_rank.get(idx),
                )
                for idx, score in top
            ]

        if timeline is not None:
            with timeline.span("retrieval_ms"):
                return _run()
        return _run()

    def describe(self) -> str:
        return (
            f"{len(self.chunks)} chunks from "
            f"{len({c.doc_id for c in self.chunks})} documents via {backend_name()}"
        )
