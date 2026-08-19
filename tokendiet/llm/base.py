"""Provider-agnostic LLM interface.

Kept deliberately small so a second provider is a new file, not a refactor.
Streaming is mandatory: TTFT is meaningless without token-level delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class Usage:
    """Server-reported token accounting. This is our ground truth."""

    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0  # >0 means the run is cache-contaminated

    @property
    def contaminated(self) -> bool:
        """A cached prefix makes a run cheaper and faster than a cold one.

        Averaging such a run into an A/B comparison is exactly the 'remembered
        baseline' failure the brief forbids, so callers must discard these.
        """
        return self.cached_prompt_tokens > 0


@dataclass
class StreamEvent:
    delta: str | None = None
    usage: Usage | None = None  # arrives on the final event


class LLMClient(Protocol):
    model: str

    def stream(self, system: str, user: str, *, max_tokens: int) -> AsyncIterator[StreamEvent]: ...
