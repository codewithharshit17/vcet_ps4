"""FastAPI backend: one request runs both paths and streams everything.

The dashboard's job is to be an evidence machine, so the wire format carries the
*provenance* of each number, not just the number: every candidate sentence with
its score and drop reason, per-stage timings for both paths, and server-reported
usage. Anything unmeasured is emitted as null and rendered as an em dash.

If the LLM key is missing, the compression stage still streams and the LLM
panels stay empty. A dashboard that invents a TTFT when it has no LLM is worse
than one that shows nothing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import settings
from .metrics import ABRun
from .pipeline import CompressionResult, Compressor

app = FastAPI(title="Token-Diet")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_compressor: Compressor | None = None
_warmup: dict[str, float] = {}
_lock = asyncio.Lock()


async def get_compressor() -> Compressor:
    """Load and warm models once, at startup, never inside a timed request."""
    global _compressor, _warmup
    async with _lock:
        if _compressor is None:
            c = await asyncio.to_thread(Compressor.load)
            _warmup = await asyncio.to_thread(c.warmup)
            _compressor = c
    return _compressor


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def serialize_compression(c: CompressionResult) -> dict:
    return {
        "query": c.query,
        "baseline_context_tokens": c.baseline_context_tokens,
        "compressed_context_tokens": c.compressed_context_tokens,
        "tokens_saved": c.tokens_saved,
        "compression_ratio": c.compression_ratio,
        "budget": c.budget,
        "pipeline_overhead_ms": c.pipeline_overhead_ms,
        "stages": c.timeline.to_dict(),
        "cache_stats": c.cache_stats,
        "strip_tokens_saved": c.strip_tokens_saved,
        "baseline_context": c.baseline_context,
        "compressed_context": c.compressed_context,
        "sentences": [
            {
                "sid": s.sid,
                "doc_id": s.sentence.doc_id,
                "chunk_id": s.sentence.chunk_id,
                "char_offset": s.sentence.char_offset,
                "text": s.sentence.text,
                "tokens": s.token_count,
                "kind": s.sentence.kind.value,
                "relevance": s.relevance,
                "ce_norm": s.ce_norm,
                "bm25_norm": s.bm25_norm,
                "mmr_score": s.mmr_score,
                "similarity": s.similarity,
                "redundant_with": s.redundant_with,
                "selected": s.selected,
                "drop_reason": s.drop_reason,
            }
            # Document order, so the diff view reads like the source document.
            for s in sorted(
                c.sentences, key=lambda x: (x.sentence.doc_id, x.sentence.char_offset)
            )
        ],
    }


def serialize_ab(run: ABRun) -> dict:
    def path(p):
        return {
            "context_tokens": p.context_tokens,
            "prompt_tokens": p.prompt_tokens,
            "output_tokens": p.output_tokens,
            "ttft_ms": p.ttft_ms,
            "total_latency_ms": p.total_latency_ms,
            "stream_ms": p.stream_ms,
            "cost_usd": p.cost(),
            "stages": p.stages,
            "error": p.error,
            "contaminated": p.contaminated,
        }

    return {
        "baseline": path(run.baseline),
        "compressed": path(run.compressed),
        "ttft_delta_ms": run.ttft_delta_ms,
        "cost_saved_usd": run.cost_saved_usd,
        "cost_saved_actual_usd": run.cost_saved_actual_usd,
        "output_token_delta": run.output_token_delta,
        "compression_ratio": run.compression_ratio,
        "tokens_saved": run.tokens_saved,
        "contaminated": run.contaminated,
    }


@app.get("/api/config")
async def config() -> dict:
    return {
        "token_budget": settings.token_budget,
        "mmr_lambda": settings.mmr_lambda,
        "relevance_threshold": settings.relevance_threshold,
        "top_k": settings.top_k,
        "aggressive_strip": settings.aggressive_strip,
        "answer_model": settings.answer_model,
        "embed_model": settings.embed_model,
        "cross_encoder_model": settings.cross_encoder_model,
        "price_in_per_mtok": settings.answer_price_input_per_mtok,
        "price_out_per_mtok": settings.answer_price_output_per_mtok,
        "warmup_ms": _warmup,
        "llm_configured": bool(settings.groq_api_key),
        "budget_sweep": [b for b in settings.eval_budget_sweep],
    }


@app.get("/api/run")
async def run(
    query: str,
    budget: int | None = None,
    mmr_lambda: float | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
    strip: bool = False,
    llm: bool = True,
) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        compressor = await get_compressor()

        from .dualpath import SYSTEM_PROMPT, build_prompt
        from .llm.groq import GroqClient, cache_buster
        from .timing import Timeline, new_query_clock, now_ns

        t0 = new_query_clock()
        tl = Timeline(label="compressed", t0_ns=t0)

        comp = await asyncio.to_thread(
            compressor.run,
            query,
            budget=budget,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
            relevance_threshold=threshold,
            aggressive_strip=strip,
            timeline=tl,
        )
        yield sse("compression", serialize_compression(comp))

        if not llm:
            yield sse("done", {"skipped": "llm disabled"})
            return
        if not settings.groq_api_key:
            # Explicitly say the metrics are unavailable rather than sending 0.
            yield sse("llm_unavailable", {
                "reason": "GROQ_API_KEY not set. LLM metrics unmeasured."
            })
            yield sse("done", {})
            return

        client = GroqClient()
        paths = {
            "baseline": (comp.baseline_context, comp.baseline_context_tokens),
            "compressed": (comp.compressed_context, comp.compressed_context_tokens),
        }
        queue: asyncio.Queue = asyncio.Queue()

        async def drive(label: str) -> None:
            ctx, ntok = paths[label]
            first = False
            try:
                prompt = build_prompt(ctx, query, cache_buster())
                usage = None
                async for ev in client.stream(SYSTEM_PROMPT, prompt):
                    if ev.delta:
                        if not first:
                            first = True
                            await queue.put(("ttft", {"path": label,
                                                      "ttft_ms": (now_ns() - t0) / 1e6}))
                        await queue.put(("delta", {"path": label, "text": ev.delta}))
                    if ev.usage is not None:
                        usage = ev.usage
                await queue.put(("path_done", {
                    "path": label,
                    "total_latency_ms": (now_ns() - t0) / 1e6,
                    "context_tokens": ntok,
                    "prompt_tokens": None if usage is None else usage.prompt_tokens,
                    "output_tokens": None if usage is None else usage.completion_tokens,
                    "cached_prompt_tokens": None if usage is None else usage.cached_prompt_tokens,
                    "contaminated": bool(usage and usage.contaminated),
                }))
            except Exception as exc:  # noqa: BLE001
                await queue.put(("path_error", {"path": label,
                                                "error": f"{type(exc).__name__}: {exc}"}))
            finally:
                await queue.put(("__end__", {"path": label}))

        tasks = [asyncio.create_task(drive(p)) for p in ("baseline", "compressed")]
        finished = 0
        while finished < 2:
            event, payload = await queue.get()
            if event == "__end__":
                finished += 1
                continue
            yield sse(event, payload)

        await asyncio.gather(*tasks, return_exceptions=True)
        yield sse("done", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@app.get("/api/eval")
async def eval_results() -> dict:
    """Serve eval/results.json if the harness has been run."""
    p = Path(__file__).resolve().parent.parent / "eval" / "results.json"
    if not p.exists():
        return {"available": False, "reason": "Run `uv run python eval/run_eval.py` first."}
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}


if _DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")
