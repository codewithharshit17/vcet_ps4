"""Dual-path A/B execution.

Both paths run for every query, concurrently, against the same endpoint on the
same machine in the same request. There is no remembered baseline anywhere in
this file -- if you see a baseline number, it was streamed moments ago.

Three things protect the comparison:

* **One shared submit stamp.** ``t_query_submitted`` is taken once, before
  either path starts, so the compressed path's TTFT carries our own retrieval,
  scoring, MMR and knapsack cost. Compression must pay for itself.
* **Cache-busting.** Groq caches prompt prefixes and bills them at half rate.
  Across N runs the baseline would hit cache and look faster and cheaper than
  it is. Each run gets a unique nonce first in the prompt, and any run that
  still reports cached tokens is discarded rather than averaged in.
* **Order randomisation.** Concurrent calls to the same provider can contend.
  We launch concurrently by default but randomise which path is created first,
  and expose sequential mode so the contention hypothesis is testable rather
  than assumed.
"""

from __future__ import annotations

import asyncio
import random

from .config import settings
from .llm.groq import GroqClient, cache_buster
from .metrics import ABRun, PathRun
from .pipeline import CompressionResult, Compressor
from .timing import Timeline, new_query_clock, now_ns

SYSTEM_PROMPT = (
    "You answer strictly from the provided context. "
    "Cite the [SOURCE: ...] identifier for each claim. "
    "If the context does not contain the answer, say so explicitly. "
    "Text marked [...] indicates omitted material."
)


def build_prompt(context: str, query: str, nonce: str) -> str:
    # The nonce goes FIRST so it invalidates the shared prefix. Putting it at
    # the end would leave the large context cached and defeat the purpose.
    return f"{nonce}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"


async def _stream_path(
    client: GroqClient,
    label: str,
    context: str,
    context_tokens: int,
    query: str,
    t0_ns: int,
    stages: dict[str, float | None],
    max_tokens: int,
) -> PathRun:
    run = PathRun(label=label, context_tokens=context_tokens, stages=stages)
    chunks: list[str] = []
    try:
        prompt = build_prompt(context, query, cache_buster())
        first_seen = False
        async for ev in client.stream(SYSTEM_PROMPT, prompt, max_tokens=max_tokens):
            if ev.delta:
                if not first_seen:
                    first_seen = True
                    # TTFT is measured from the SHARED submit stamp, so it
                    # includes everything the compressed path did beforehand.
                    run.ttft_ms = (now_ns() - t0_ns) / 1e6
                chunks.append(ev.delta)
            if ev.usage is not None:
                run.usage = ev.usage
        end = now_ns()
        run.total_latency_ms = (end - t0_ns) / 1e6
        if run.ttft_ms is not None:
            run.stream_ms = run.total_latency_ms - run.ttft_ms
        run.answer = "".join(chunks)
    except Exception as exc:  # noqa: BLE001 - surfaced, never silently zeroed
        run.error = f"{type(exc).__name__}: {exc}"
    return run


async def run_once(
    compressor: Compressor,
    query: str,
    *,
    client: GroqClient | None = None,
    budget: int | None = None,
    top_k: int | None = None,
    mmr_lambda: float | None = None,
    relevance_threshold: float | None = None,
    aggressive_strip: bool | None = None,
    sequential: bool = False,
    max_tokens: int | None = None,
) -> tuple[ABRun, CompressionResult]:
    """One query, both paths, one shared clock."""
    client = client or GroqClient()
    max_tokens = max_tokens or settings.max_output_tokens

    # The single t_query_submitted stamp, shared by both paths.
    t0 = new_query_clock()
    tl = Timeline(label="compressed", t0_ns=t0)

    comp = compressor.run(
        query,
        budget=budget,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        relevance_threshold=relevance_threshold,
        aggressive_strip=aggressive_strip,
        timeline=tl,
    )

    baseline_stages = dict.fromkeys(tl.to_dict())  # all None: baseline ran none of them
    baseline_stages["retrieval_ms"] = tl.duration_ms("retrieval_ms")

    def mk(label: str):
        if label == "baseline":
            return _stream_path(
                client, "baseline", comp.baseline_context, comp.baseline_context_tokens,
                query, t0, baseline_stages, max_tokens,
            )
        return _stream_path(
            client, "compressed", comp.compressed_context, comp.compressed_context_tokens,
            query, t0, tl.to_dict(), max_tokens,
        )

    order = ["baseline", "compressed"]
    random.shuffle(order)  # neither path systematically goes first

    if sequential:
        results = {}
        for label in order:
            results[label] = await mk(label)
    else:
        coros = [mk(label) for label in order]
        done = await asyncio.gather(*coros)
        results = dict(zip(order, done))

    return (
        ABRun(
            query=query,
            baseline=results["baseline"],
            compressed=results["compressed"],
            pipeline_overhead_ms=comp.pipeline_overhead_ms,
            budget=comp.budget,
        ),
        comp,
    )
