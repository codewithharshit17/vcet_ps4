"""[3] Parallel hybrid relevance scoring (the deck's stage).

Two signals, computed over sentences and blended:

* **Cross-encoder** (ms-marco-MiniLM) -- the dense semantic match, and the
  pipeline's hot loop. Batched, and cached on hash(query + sentence + model).
* **BM25** -- the lexical keyword match, fitted over the candidate sentences
  for this query.

Normalisation to [0,1] is deliberately asymmetric:

* Cross-encoder logits go through a **sigmoid**, which is an absolute mapping.
  Min-max would force the best sentence of every query to exactly 1.0, making
  a relevance *threshold* meaningless across queries -- a query whose best
  sentence is genuinely poor would still score 1.0.
* BM25 has no natural scale, so it is min-maxed within the query.

Setting score_weight_bm25 = 0 reproduces the brief's cross-encoder-only
pipeline exactly, so the eval can measure whether the BM25 term earns its keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .cache import get_scores, put_scores, score_key
from .config import settings
from .retrieve import tokenize_lexical
from .sentences import Sentence
from .timing import Timeline


@dataclass
class ScoredSentence:
    sentence: Sentence
    ce_raw: float | None      # None when the blend ignores the cross-encoder
    bm25_raw: float
    ce_norm: float
    bm25_norm: float
    relevance: float          # blended, [0,1]

    # filled in by later stages; None means "stage did not run"
    mmr_score: float | None = None
    redundant_with: str | None = None
    similarity: float | None = None
    selected: bool = False
    drop_reason: str | None = None

    @property
    def sid(self) -> str:
        return self.sentence.sid

    @property
    def token_count(self) -> int:
        return self.sentence.token_count

    @property
    def selection_score(self) -> float:
        """Non-negative score used for budget selection.

        MMR's raw value is lambda*rel - (1-lambda)*sim, which goes NEGATIVE for
        weakly-relevant or redundant sentences. Dividing a negative numerator by
        token_count makes longer sentences score *higher* -- the density
        heuristic silently inverts and starts preferring exactly the bloat we
        exist to remove. Clamping at zero keeps density monotone in length.
        """
        base = self.mmr_score if self.mmr_score is not None else self.relevance
        return max(0.0, base)

    @property
    def density(self) -> float:
        """Score per token -- the actual 'token diet' criterion."""
        return self.selection_score / max(1, self.token_count)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # All equal: carry no signal rather than inventing a spread.
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


@dataclass
class Scorer:
    _model: object | None = None
    stats: dict[str, int] = field(default_factory=lambda: {"cache_hits": 0, "cache_misses": 0})

    @classmethod
    def load(cls) -> "Scorer":
        from sentence_transformers import CrossEncoder

        s = cls()
        s._model = CrossEncoder(settings.cross_encoder_model)
        return s

    def warmup(self) -> float:
        from .timing import now_ns

        t = now_ns()
        self._model.predict([("warmup query", "warmup sentence")])
        return (now_ns() - t) / 1e6

    def score(
        self, query: str, sentences: list[Sentence], timeline: Timeline | None = None
    ) -> list[ScoredSentence]:
        def _run() -> list[ScoredSentence]:
            if not sentences:
                return []

            w_ce = settings.score_weight_cross_encoder
            w_bm = settings.score_weight_bm25
            total_w = w_ce + w_bm
            if total_w <= 0:
                raise ValueError("score weights must sum to > 0")
            w_ce, w_bm = w_ce / total_w, w_bm / total_w

            # ---- BM25 over this query's candidate sentences
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi([tokenize_lexical(s.text) for s in sentences])
            bm25_raw = list(bm25.get_scores(tokenize_lexical(query)))

            # ---- cross-encoder, cache-aware
            ce_raw: list[float | None] = [None] * len(sentences)
            if w_ce > 0:
                keys = [
                    score_key(query, s.text, settings.cross_encoder_model) for s in sentences
                ]
                cached = get_scores(list(dict.fromkeys(keys)))
                self.stats["cache_hits"] += sum(1 for k in keys if k in cached)

                todo = [i for i, k in enumerate(keys) if k not in cached]
                self.stats["cache_misses"] += len(todo)
                if todo:
                    preds = self._model.predict(
                        [(query, sentences[i].text) for i in todo],
                        batch_size=settings.cross_encoder_batch_size,
                        show_progress_bar=False,
                    )
                    fresh = {keys[i]: float(p) for i, p in zip(todo, preds)}
                    put_scores(fresh)
                    cached.update(fresh)
                ce_raw = [cached[k] for k in keys]

            ce_norm = [_sigmoid(v) if v is not None else 0.0 for v in ce_raw]
            bm_norm = _minmax(bm25_raw)

            return [
                ScoredSentence(
                    sentence=s,
                    ce_raw=ce_raw[i],
                    bm25_raw=bm25_raw[i],
                    ce_norm=ce_norm[i],
                    bm25_norm=bm_norm[i],
                    relevance=w_ce * ce_norm[i] + w_bm * bm_norm[i],
                )
                for i, s in enumerate(sentences)
            ]

        if timeline is not None:
            with timeline.span("rerank_ms"):
                return _run()
        return _run()
