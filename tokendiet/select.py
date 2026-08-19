"""[5] Token-budget knapsack selection.

Greedy by **score-per-token density**, not raw score. This is the actual token
diet: a 300-token paragraph scoring 0.9 loses to three 40-token sentences
scoring 0.7 each, because the budget buys far more signal per token.

Two guarantees:

* The top-1 scoring sentence is always included, regardless of length. Without
  this, a single long sentence that *is* the answer can be starved out by
  cheaper, less relevant filler.
* Selection stops when adding the next sentence would exceed the budget, but
  the scan continues -- a later, smaller sentence can still fit. That is
  strictly better than stopping dead, at no quality cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .score import ScoredSentence
from .timing import Timeline


@dataclass
class SelectionResult:
    selected: list[ScoredSentence]
    dropped: list[ScoredSentence]
    budget: int | None
    selected_tokens: int

    @property
    def all_sentences(self) -> list[ScoredSentence]:
        return self.selected + self.dropped


def select(
    scored: list[ScoredSentence],
    *,
    budget: int | None = None,
    relevance_threshold: float | None = None,
    timeline: Timeline | None = None,
) -> SelectionResult:
    """Pick sentences under a token budget. budget=None means unlimited."""
    budget = settings.token_budget if budget is None else budget
    thresh = (
        settings.relevance_threshold if relevance_threshold is None else relevance_threshold
    )

    def _run() -> SelectionResult:
        # Sentences MMR already rejected keep their reason and never come back.
        candidates = [s for s in scored if s.drop_reason is None]
        pre_dropped = [s for s in scored if s.drop_reason is not None]

        below: list[ScoredSentence] = []
        keep: list[ScoredSentence] = []
        for s in candidates:
            if s.relevance < thresh:
                s.drop_reason = f"low relevance ({s.relevance:.2f})"
                below.append(s)
            else:
                keep.append(s)

        if not keep:
            return SelectionResult([], pre_dropped + below, budget, 0)

        # Guarantee: the single best sentence is in, whatever it costs.
        best = max(keep, key=lambda s: s.relevance)

        selected: list[ScoredSentence] = [best]
        used = best.token_count
        best.selected = True

        if budget is not None and used > budget:
            # The mandated sentence alone blows the budget. Keeping it is the
            # documented guarantee; record the overrun rather than hide it.
            for s in keep:
                if s is not best:
                    s.drop_reason = "over budget"
            return SelectionResult(selected, pre_dropped + below + [s for s in keep if s is not best], budget, used)

        for s in sorted(keep, key=lambda s: -s.density):
            if s is best:
                continue
            if budget is None or used + s.token_count <= budget:
                s.selected = True
                selected.append(s)
                used += s.token_count
            else:
                s.drop_reason = "over budget"

        rejected = [s for s in keep if not s.selected]
        return SelectionResult(selected, pre_dropped + below + rejected, budget, used)

    if timeline is not None:
        with timeline.span("select_ms"):
            return _run()
    return _run()
