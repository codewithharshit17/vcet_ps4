"""Token counting: a fast local counter, plus proof that it is trustworthy.

Rule 2 of the brief says token counts must come from a real tokenizer matching
the target LLM. Two facts shape this module:

* The knapsack needs a count for every candidate sentence (~300-500 per query).
  That cannot be a network round-trip.
* Groq returns ``usage.prompt_tokens`` on the very call we are already measuring.
  That is server-side ground truth, free, with no extra request.

So: count locally with tiktoken, and *calibrate against the server* to prove the
local counter agrees. The calibration is not decoration -- if it disagrees, the
disagreement gets published as a measured number rather than hidden.

Calibration design
------------------
For a sample of texts we compare ``local(text)`` against ``usage.prompt_tokens``
from a single-user-message request containing exactly that text. The difference
is (chat-template overhead) + (tokenizer disagreement). Those are separable:

* If the delta is **constant** across wildly different lengths, the tokenizer
  agrees exactly and the constant is the fixed template overhead.
* If the delta **grows with length**, the tokenizer genuinely disagrees, and the
  slope quantifies by how much.

We therefore report mean delta, its spread, and the slope of delta vs. length.
A slope of ~0 means the local counter is exact and rule 2 is fully satisfied.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol

import tiktoken

from .config import settings


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass
class LocalTokenCounter:
    """tiktoken-backed counter. The encoding is resolved, not assumed."""

    encoding_name: str
    _enc: "tiktoken.Encoding"

    @classmethod
    def resolve(cls, candidates: tuple[str, ...] | None = None) -> "LocalTokenCounter":
        """Pick the first candidate encoding this tiktoken build actually has.

        We do not hardcode one: the GPT-OSS family's encoding name is verified
        empirically rather than recalled from memory.
        """
        cands = candidates or settings.tokenizer_encoding_candidates
        tried: list[str] = []
        for name in cands:
            try:
                enc = tiktoken.get_encoding(name)
            except Exception as exc:  # noqa: BLE001 - report, then try the next
                tried.append(f"{name}: {type(exc).__name__}")
                continue
            return cls(encoding_name=name, _enc=enc)
        raise RuntimeError(
            "No usable tiktoken encoding found. Tried:\n  " + "\n  ".join(tried)
        )

    def count(self, text: str) -> int:
        # disallowed_special=() so corpus text containing e.g. "<|endoftext|>"
        # is counted as ordinary text instead of raising.
        return len(self._enc.encode(text, disallowed_special=()))

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(ids) for ids in self._enc.encode_batch(texts, disallowed_special=())]


@dataclass
class CalibrationResult:
    encoding_name: str
    n_samples: int
    mean_delta: float          # server prompt_tokens - local count
    stdev_delta: float
    min_delta: int
    max_delta: int
    slope_per_1k_tokens: float  # delta growth per 1000 local tokens
    exact: bool                 # tokenizer agrees (slope ~0 and delta constant)

    def render(self) -> str:
        verdict = (
            "EXACT - local counter matches the server tokenizer; the constant "
            f"offset of {self.mean_delta:.0f} tokens is fixed chat-template overhead."
            if self.exact
            else "APPROXIMATE - local counter disagrees with the server; see slope."
        )
        return (
            f"Tokenizer calibration ({self.encoding_name}, n={self.n_samples})\n"
            f"  delta (server - local): mean {self.mean_delta:.2f}, "
            f"stdev {self.stdev_delta:.2f}, range [{self.min_delta}, {self.max_delta}]\n"
            f"  slope: {self.slope_per_1k_tokens:+.3f} tokens per 1000 local tokens\n"
            f"  verdict: {verdict}"
        )


def calibrate(
    counter: LocalTokenCounter,
    samples: list[str],
    server_prompt_tokens: list[int],
    *,
    slope_tolerance: float = 0.5,
    stdev_tolerance: float = 0.5,
) -> CalibrationResult:
    """Compare local counts against real server ``usage.prompt_tokens`` values.

    ``server_prompt_tokens[i]`` must come from an actual API call whose only
    user-message content was ``samples[i]``.
    """
    if len(samples) != len(server_prompt_tokens):
        raise ValueError("samples and server_prompt_tokens must be the same length")
    if len(samples) < 2:
        raise ValueError("need at least 2 samples to estimate a slope")

    local = [counter.count(s) for s in samples]
    deltas = [srv - loc for srv, loc in zip(server_prompt_tokens, local)]

    # Least-squares slope of delta vs. local length.
    mean_x = statistics.fmean(local)
    mean_y = statistics.fmean(deltas)
    denom = sum((x - mean_x) ** 2 for x in local)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(local, deltas)) / denom
        if denom
        else 0.0
    )

    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    slope_per_1k = slope * 1000.0

    return CalibrationResult(
        encoding_name=counter.encoding_name,
        n_samples=len(samples),
        mean_delta=mean_y,
        stdev_delta=stdev,
        min_delta=min(deltas),
        max_delta=max(deltas),
        slope_per_1k_tokens=slope_per_1k,
        exact=abs(slope_per_1k) < slope_tolerance and stdev < stdev_tolerance,
    )


_counter: LocalTokenCounter | None = None


def get_counter() -> LocalTokenCounter:
    """Process-wide local counter (resolving the encoding downloads a BPE file once)."""
    global _counter
    if _counter is None:
        _counter = LocalTokenCounter.resolve()
    return _counter
