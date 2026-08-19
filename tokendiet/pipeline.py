"""Compression pipeline: stages [2] through [6], plus optional [5b].

Owns no metric definitions of its own -- it measures token counts with the real
tokenizer and hands raw numbers to metrics.py. Every unmeasured field stays
None so the dashboard renders an em dash instead of a plausible-looking zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .assemble import assemble, assemble_baseline
from .config import settings
from .mmr import Embedder, apply_mmr
from .retrieve import RetrievedChunk, Retriever
from .score import ScoredSentence, Scorer
from .select import select
from .sentences import decompose
from .strip import apply_strip
from .timing import Timeline
from .tokens import get_counter


@dataclass
class CompressionResult:
    query: str
    hits: list[RetrievedChunk]
    sentences: list[ScoredSentence]      # every candidate, kept and dropped
    baseline_context: str
    compressed_context: str
    baseline_context_tokens: int
    compressed_context_tokens: int
    budget: int | None
    timeline: Timeline
    strip_tokens_saved: int | None = None
    cache_stats: dict[str, int] = field(default_factory=dict)

    # -- formulas live in metrics.py; these are the raw inputs it needs
    @property
    def tokens_saved(self) -> int:
        return self.baseline_context_tokens - self.compressed_context_tokens

    @property
    def compression_ratio(self) -> float | None:
        if not self.baseline_context_tokens:
            return None
        return 1.0 - (self.compressed_context_tokens / self.baseline_context_tokens)

    @property
    def kept(self) -> list[ScoredSentence]:
        return [s for s in self.sentences if s.selected]

    @property
    def dropped(self) -> list[ScoredSentence]:
        return [s for s in self.sentences if not s.selected]

    @property
    def pipeline_overhead_ms(self) -> float:
        """t_end_of_reassembly - t_end_of_retrieval: everything we added."""
        stages = ("sentence_split_ms", "rerank_ms", "mmr_ms", "select_ms", "strip_ms", "assemble_ms")
        return sum(self.timeline.duration_ms(s) or 0.0 for s in stages)


@dataclass
class Compressor:
    retriever: Retriever
    scorer: Scorer
    embedder: Embedder

    @classmethod
    def load(cls) -> "Compressor":
        return cls(retriever=Retriever.load(), scorer=Scorer.load(), embedder=Embedder())

    def warmup(self) -> dict[str, float]:
        """Explicitly warm every local model. Discarded from all timings."""
        return {
            "retriever_ms": self.retriever.warmup(),
            "cross_encoder_ms": self.scorer.warmup(),
            "embedder_ms": self.embedder.warmup(),
        }

    def run(
        self,
        query: str,
        *,
        budget: int | None = None,
        top_k: int | None = None,
        mmr_lambda: float | None = None,
        relevance_threshold: float | None = None,
        aggressive_strip: bool | None = None,
        timeline: Timeline | None = None,
    ) -> CompressionResult:
        counter = get_counter()
        tl = timeline or Timeline(label="compressed")
        budget = settings.token_budget if budget is None else budget
        do_strip = settings.aggressive_strip if aggressive_strip is None else aggressive_strip

        # [1] retrieval
        hits = self.retriever.retrieve(query, k=top_k, timeline=tl)
        baseline_context = assemble_baseline(hits)

        # [2] decomposition
        sents = decompose(hits, timeline=tl)

        # [3] hybrid scoring
        scored = self.scorer.score(query, sents, timeline=tl)

        # [4] redundancy suppression
        scored = apply_mmr(scored, self.embedder, lam=mmr_lambda, timeline=tl)

        # [5] budget knapsack
        result = select(
            scored, budget=budget, relevance_threshold=relevance_threshold, timeline=tl
        )

        # [5b] optional word stripping
        strip_saved: int | None = None
        texts: list[str] | None = None
        if do_strip and result.selected:
            texts, strip_saved = apply_strip(result.selected, timeline=tl)

        # [6] reassembly
        with tl.span("assemble_ms"):
            if texts is not None:
                originals = [s.sentence.text for s in result.selected]
                for s, t in zip(result.selected, texts):
                    s.sentence.text = t
                compressed_context = assemble(result.selected)
                for s, o in zip(result.selected, originals):
                    s.sentence.text = o  # restore for display/diff
            else:
                compressed_context = assemble(result.selected)

        return CompressionResult(
            query=query,
            hits=hits,
            sentences=result.all_sentences,
            baseline_context=baseline_context,
            compressed_context=compressed_context,
            baseline_context_tokens=counter.count(baseline_context),
            compressed_context_tokens=counter.count(compressed_context),
            budget=budget,
            timeline=tl,
            strip_tokens_saved=strip_saved,
            cache_stats={
                "ce_hits": self.scorer.stats["cache_hits"],
                "ce_misses": self.scorer.stats["cache_misses"],
                "emb_hits": self.embedder.stats["cache_hits"],
                "emb_misses": self.embedder.stats["cache_misses"],
            },
        )
