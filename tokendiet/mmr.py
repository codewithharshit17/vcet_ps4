"""[4] Redundancy suppression via Maximal Marginal Relevance.

    MMR(s) = lambda * rel(s) - (1 - lambda) * max_{s' in S} sim(s, s')

This is what kills "the same fact restated in three retrieved chunks", the
single biggest source of wasted tokens in real RAG. On a corpus of five 10-K
filings it matters even more than usual: every company writes near-identical
climate, cybersecurity and supply-chain risk boilerplate.

MMR here does two jobs. It rewrites each sentence's score to be
redundancy-aware (the knapsack then spends budget on that adjusted score), and
it explicitly *drops* sentences whose similarity to an already-accepted
sentence exceeds a threshold, recording which sentence they duplicate so the
dashboard can show "redundant with S4 (sim 0.91)".
"""

from __future__ import annotations

import numpy as np

from .cache import embed_key, get_vectors, put_vectors
from .config import settings
from .score import ScoredSentence
from .timing import Timeline, now_ns


class Embedder:
    """Sentence embeddings for the similarity term (same model as retrieval).

    Cached: profiling showed encoding dominated the MMR stage (2.8s of 3.5s).
    An embedding is query-independent, so it is reusable across every query
    that retrieves the same chunk.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.embed_model)
        self.stats = {"cache_hits": 0, "cache_misses": 0}

    def warmup(self) -> float:
        t = now_ns()
        self._model.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True)
        return (now_ns() - t) / 1e6

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")

        keys = [embed_key(t, settings.embed_model) for t in texts]
        cached = get_vectors(list(dict.fromkeys(keys)))
        self.stats["cache_hits"] += sum(1 for k in keys if k in cached)

        todo = [i for i, k in enumerate(keys) if k not in cached]
        self.stats["cache_misses"] += len(todo)
        if todo:
            fresh = self._encode_raw([texts[i] for i in todo])
            new = {keys[i]: fresh[j] for j, i in enumerate(todo)}
            put_vectors(new)
            cached.update(new)

        return np.vstack([cached[k] for k in keys]).astype("float32")


def apply_mmr(
    scored: list[ScoredSentence],
    embedder: Embedder,
    *,
    lam: float | None = None,
    redundancy_threshold: float | None = None,
    timeline: Timeline | None = None,
) -> list[ScoredSentence]:
    """Annotate sentences with mmr_score, and drop near-duplicates.

    Returns the same objects, mutated, in MMR-selection order followed by the
    dropped ones -- nothing is discarded, so the dashboard can render what was
    removed and why.
    """
    lam = settings.mmr_lambda if lam is None else lam
    thresh = (
        settings.mmr_redundancy_threshold
        if redundancy_threshold is None
        else redundancy_threshold
    )

    def _run() -> list[ScoredSentence]:
        if not scored:
            return []

        # Vectors are L2-normalised, so a dot product is cosine similarity.
        vecs = embedder.encode([s.sentence.text for s in scored])
        rel = np.array([s.relevance for s in scored], dtype="float32")

        n = len(scored)
        remaining = set(range(n))
        chosen: list[int] = []
        max_sim = np.zeros(n, dtype="float32")  # similarity to nearest chosen
        nearest = np.full(n, -1, dtype="int32")

        ordered: list[ScoredSentence] = []
        dropped: list[ScoredSentence] = []

        while remaining:
            idx_list = np.fromiter(remaining, dtype="int32", count=len(remaining))
            if not chosen:
                # First pick is pure relevance: nothing to be redundant with.
                mmr_vals = rel[idx_list]
            else:
                mmr_vals = lam * rel[idx_list] - (1.0 - lam) * max_sim[idx_list]

            best_pos = int(np.argmax(mmr_vals))
            best = int(idx_list[best_pos])
            s = scored[best]

            if chosen and max_sim[best] >= thresh:
                # Near-duplicate of something already kept.
                s.mmr_score = float(mmr_vals[best_pos])
                s.similarity = float(max_sim[best])
                s.redundant_with = scored[int(nearest[best])].sid
                s.drop_reason = (
                    f"redundant with {s.redundant_with} (sim {s.similarity:.2f})"
                )
                dropped.append(s)
                remaining.discard(best)
                continue

            s.mmr_score = float(mmr_vals[best_pos])
            if chosen:
                s.similarity = float(max_sim[best])
                s.redundant_with = scored[int(nearest[best])].sid
            chosen.append(best)
            ordered.append(s)
            remaining.discard(best)

            # Update running max-similarity against the newly chosen sentence.
            if remaining:
                rest = np.fromiter(remaining, dtype="int32", count=len(remaining))
                sims = vecs[rest] @ vecs[best]
                improved = sims > max_sim[rest]
                max_sim[rest] = np.where(improved, sims, max_sim[rest])
                nearest[rest] = np.where(improved, best, nearest[rest])

        return ordered + dropped

    if timeline is not None:
        with timeline.span("mmr_ms"):
            return _run()
    return _run()
