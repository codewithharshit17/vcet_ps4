"""Groq client (OpenAI-compatible).

Two things here exist specifically to protect metric honesty:

* ``probe_prompt_tokens`` gives the tokenizer calibration its ground truth.
* ``cache_buster`` defeats Groq's automatic prefix caching. Groq bills cached
  input at half rate, so across N=5 runs the baseline path would otherwise hit
  cache on runs 2-5 and report artificially low latency and cost -- a remembered
  baseline dressed as a live one. Every run gets a unique nonce at the FRONT of
  the prompt, which invalidates the shared prefix and keeps both paths cold.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI

from ..config import settings
from .base import StreamEvent, Usage


def cache_buster() -> str:
    """Unique per-run prefix. Must be first in the prompt to break the cache."""
    return f"[run:{uuid.uuid4().hex[:16]}]"


def _extract_usage(raw) -> Usage | None:
    if raw is None:
        return None
    cached = 0
    details = getattr(raw, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return Usage(
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens or 0,
        cached_prompt_tokens=cached,
    )


@dataclass
class GroqClient:
    model: str = ""

    def __post_init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
                "from https://console.groq.com/keys"
            )
        self.model = self.model or settings.answer_model
        self._sync = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
        self._async = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)

    # ------------------------------------------------------------ calibration
    def probe_prompt_tokens(self, text: str) -> int:
        """Server-side token count for a single user message containing `text`."""
        resp = self._sync.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,
            temperature=0,
        )
        return resp.usage.prompt_tokens

    # --------------------------------------------------------------- generate
    async def stream(self, system: str, user: str, *, max_tokens: int | None = None):
        """Yield StreamEvents. The final event carries server usage."""
        stream = await self._async.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens or settings.max_output_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = _extract_usage(getattr(chunk, "usage", None))
            delta = None
            if chunk.choices:
                delta = chunk.choices[0].delta.content
            if delta or usage:
                yield StreamEvent(delta=delta, usage=usage)
