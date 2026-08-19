"""Timing instrumentation.

Two rules this module exists to enforce:

1. A stage that did not run reports ``None``, never ``0.0``. The baseline path
   never reranks; reporting ``rerank_ms: 0`` would be a fabricated measurement.
2. Both paths share a single ``t_query_submitted`` stamp, taken before either
   path starts. The compressed path's TTFT therefore carries our own pipeline
   overhead, which is the entire point of the comparison.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# Every stage we know how to time. `Timeline.to_dict()` always emits all of
# these keys so a missing measurement is visibly None rather than absent.
STAGE_KEYS: tuple[str, ...] = (
    "retrieval_ms",
    "sentence_split_ms",
    "rerank_ms",
    "mmr_ms",
    "select_ms",
    "strip_ms",
    "assemble_ms",
    "llm_ttft_ms",
    "llm_stream_ms",
)


def now_ns() -> int:
    """Monotonic clock. Never use wall-clock for durations."""
    return time.perf_counter_ns()


def _ms(ns: int) -> float:
    return ns / 1_000_000.0


@dataclass
class Timeline:
    """Records stage durations and absolute event marks for a single path."""

    label: str = ""
    t0_ns: int | None = None  # shared query-submission stamp
    _durations_ns: dict[str, int] = field(default_factory=dict)
    _marks_ns: dict[str, int] = field(default_factory=dict)
    _open: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str):
        start = now_ns()
        try:
            yield
        finally:
            self._durations_ns[name] = self._durations_ns.get(name, 0) + (now_ns() - start)

    def start(self, name: str) -> None:
        self._open[name] = now_ns()

    def stop(self, name: str) -> None:
        if name not in self._open:
            raise KeyError(f"stop({name!r}) without matching start()")
        start = self._open.pop(name)
        self._durations_ns[name] = self._durations_ns.get(name, 0) + (now_ns() - start)

    # ------------------------------------------------------------------ marks
    def mark(self, name: str) -> int:
        """Stamp an absolute event (e.g. first token received)."""
        t = now_ns()
        self._marks_ns[name] = t
        return t

    def since_submit_ms(self, mark_name: str) -> float | None:
        """Milliseconds from the SHARED submit stamp to a marked event."""
        if self.t0_ns is None or mark_name not in self._marks_ns:
            return None
        return _ms(self._marks_ns[mark_name] - self.t0_ns)

    # ----------------------------------------------------------------- output
    def duration_ms(self, name: str) -> float | None:
        ns = self._durations_ns.get(name)
        return None if ns is None else _ms(ns)

    def elapsed_between_ms(self, a: str, b: str) -> float | None:
        """Duration between two absolute marks, or None if either is missing."""
        if a not in self._marks_ns or b not in self._marks_ns:
            return None
        return _ms(self._marks_ns[b] - self._marks_ns[a])

    def to_dict(self) -> dict[str, float | None]:
        """All known stages. Un-run stages are None, not 0."""
        return {k: self.duration_ms(k) for k in STAGE_KEYS}


def new_query_clock() -> int:
    """The single ``t_query_submitted`` stamp shared by both paths."""
    return now_ns()
