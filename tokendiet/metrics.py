"""Every number on the dashboard is computed here.

No other module defines a headline metric. Formulas are implemented exactly as
specified in PLAN.md section 4, and anything unmeasured stays None so the UI can
render an em dash rather than a plausible-looking zero.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from .config import settings
from .llm.base import Usage


# --------------------------------------------------------------------- stats
@dataclass
class Distribution:
    """Headline metrics are medians over N>=5 runs, with p95 alongside.

    A single run of a network call is a fantasy number: jitter alone can swing
    TTFT by hundreds of milliseconds.
    """

    n: int
    median: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None

    @classmethod
    def of(cls, values: list[float | None]) -> "Distribution":
        vals = [v for v in values if v is not None]
        if not vals:
            return cls(n=0, median=None, p95=None, minimum=None, maximum=None)
        ordered = sorted(vals)
        # Nearest-rank p95: with n=5 this is the max, which is honest -- it does
        # not interpolate a smoother number than the sample supports.
        idx = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) + 0.5)) - 1))
        return cls(
            n=len(vals),
            median=statistics.median(ordered),
            p95=ordered[idx],
            minimum=ordered[0],
            maximum=ordered[-1],
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- cost
def cost_usd(usage: Usage | None, price_in_per_mtok: float, price_out_per_mtok: float) -> float | None:
    """Actual billed cost from the server's own usage. None if unmeasured."""
    if usage is None:
        return None
    return (
        usage.prompt_tokens * price_in_per_mtok
        + usage.completion_tokens * price_out_per_mtok
    ) / 1_000_000.0


# ------------------------------------------------------------- per-path view
@dataclass
class PathRun:
    """One streamed generation on one path."""

    label: str                      # "baseline" | "compressed"
    context_tokens: int
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    stream_ms: float | None = None
    answer: str = ""
    usage: Usage | None = None
    stages: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None

    @property
    def output_tokens(self) -> int | None:
        return None if self.usage is None else self.usage.completion_tokens

    @property
    def prompt_tokens(self) -> int | None:
        """Server ground truth for the WHOLE request, not just the context."""
        return None if self.usage is None else self.usage.prompt_tokens

    @property
    def contaminated(self) -> bool:
        return self.usage is not None and self.usage.contaminated

    def cost(self) -> float | None:
        return cost_usd(
            self.usage,
            settings.answer_price_input_per_mtok,
            settings.answer_price_output_per_mtok,
        )


# ------------------------------------------------------------------ A/B pair
@dataclass
class ABRun:
    """One query run through both paths concurrently, sharing a submit stamp."""

    query: str
    baseline: PathRun
    compressed: PathRun
    pipeline_overhead_ms: float
    budget: int | None

    # ---- formulas (PLAN.md section 4)
    @property
    def compression_ratio(self) -> float | None:
        if not self.baseline.context_tokens:
            return None
        return 1.0 - (self.compressed.context_tokens / self.baseline.context_tokens)

    @property
    def tokens_saved(self) -> int:
        return self.baseline.context_tokens - self.compressed.context_tokens

    @property
    def ttft_delta_ms(self) -> float | None:
        """Positive = compression won."""
        if self.baseline.ttft_ms is None or self.compressed.ttft_ms is None:
            return None
        return self.baseline.ttft_ms - self.compressed.ttft_ms

    @property
    def output_token_delta(self) -> int | None:
        b, c = self.baseline.output_tokens, self.compressed.output_tokens
        return None if b is None or c is None else b - c

    @property
    def cost_saved_usd(self) -> float | None:
        """tokens_saved * input_price + output_token_delta * output_price."""
        delta = self.output_token_delta
        if delta is None:
            return None
        return (
            self.tokens_saved * settings.answer_price_input_per_mtok
            + delta * settings.answer_price_output_per_mtok
        ) / 1_000_000.0

    @property
    def cost_saved_actual_usd(self) -> float | None:
        """Difference of the two actually-billed costs.

        Kept alongside the brief's formula because they answer different
        questions: the formula prices the *context* we removed, this prices the
        whole request including the fixed system-prompt and query overhead that
        both paths pay.
        """
        b, c = self.baseline.cost(), self.compressed.cost()
        return None if b is None or c is None else b - c

    @property
    def contaminated(self) -> bool:
        """Either path served from a prefix cache invalidates the comparison."""
        return self.baseline.contaminated or self.compressed.contaminated


# ------------------------------------------------------------ aggregated view
@dataclass
class ABSummary:
    """Distributions over N runs of the same query. This is what gets reported."""

    query: str
    runs: list[ABRun]
    discarded_contaminated: int = 0

    def _f(self, fn) -> Distribution:
        return Distribution.of([fn(r) for r in self.runs])

    def report(self) -> dict:
        return {
            "query": self.query,
            "n_runs": len(self.runs),
            "discarded_contaminated": self.discarded_contaminated,
            "baseline_context_tokens": self._f(lambda r: r.baseline.context_tokens).to_dict(),
            "compressed_context_tokens": self._f(lambda r: r.compressed.context_tokens).to_dict(),
            "compression_ratio": self._f(lambda r: r.compression_ratio).to_dict(),
            "tokens_saved": self._f(lambda r: r.tokens_saved).to_dict(),
            "pipeline_overhead_ms": self._f(lambda r: r.pipeline_overhead_ms).to_dict(),
            "baseline_ttft_ms": self._f(lambda r: r.baseline.ttft_ms).to_dict(),
            "compressed_ttft_ms": self._f(lambda r: r.compressed.ttft_ms).to_dict(),
            "ttft_delta_ms": self._f(lambda r: r.ttft_delta_ms).to_dict(),
            "baseline_total_latency_ms": self._f(lambda r: r.baseline.total_latency_ms).to_dict(),
            "compressed_total_latency_ms": self._f(lambda r: r.compressed.total_latency_ms).to_dict(),
            "baseline_cost_usd": self._f(lambda r: r.baseline.cost()).to_dict(),
            "compressed_cost_usd": self._f(lambda r: r.compressed.cost()).to_dict(),
            "cost_saved_usd": self._f(lambda r: r.cost_saved_usd).to_dict(),
        }
