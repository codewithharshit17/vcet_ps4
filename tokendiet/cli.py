"""Token-Diet CLI."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from .config import settings

# SEC filings legitimately contain bullets, em-dashes and smart quotes. The
# Windows console defaults to cp1252 and renders them as replacement glyphs,
# which looks like corpus corruption when the corpus is actually fine.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - not all streams support reconfigure
        pass

app = typer.Typer(add_completion=False, help="Token-Diet — context compression with receipts.")
console = Console()


@app.command("fetch-corpus")
def fetch_corpus(
    tickers: str = typer.Option("", help="Comma-separated tickers. Default: a 5-company mix.")
) -> None:
    """Download 10-K filings from SEC EDGAR into data/corpus/."""
    from .corpus.fetch_sec import DEFAULT_TICKERS, fetch

    chosen = tuple(t.strip().upper() for t in tickers.split(",") if t.strip()) or DEFAULT_TICKERS
    paths = fetch(chosen)
    console.print(f"[green]Fetched {len(paths)} filings.[/green]")


@app.command()
def ingest(
    recreate: bool = typer.Option(True, help="Drop and rebuild the collection."),
) -> None:
    """Chunk the corpus and index it into Qdrant."""
    from .ingest import ingest as run_ingest

    chunks = run_ingest(recreate=recreate)
    tokens = sum(c.token_count for c in chunks)
    console.print(
        f"[green]Indexed {len(chunks)} chunks, {tokens:,} tokens "
        f"(mean {tokens / len(chunks):.0f}/chunk).[/green]"
    )


@app.command("calibrate-tokenizer")
def calibrate_tokenizer(n: int = typer.Option(12, help="Number of probe samples.")) -> None:
    """Prove the local tokenizer against the server's own usage.prompt_tokens."""
    from .llm.groq import GroqClient
    from .tokens import calibrate, get_counter

    counter = get_counter()
    console.print(f"Local encoding resolved to [bold]{counter.encoding_name}[/bold]")

    from .store import get_client
    from .store import iter_all_chunks

    chunks = iter_all_chunks(get_client())
    if not chunks:
        raise typer.BadParameter("Index is empty; run `tokendiet ingest` first.")

    # Deliberately spread across lengths so the slope test is meaningful.
    step = max(1, len(chunks) // n)
    samples = [c.text for c in chunks[::step][:n]]
    samples = [s[: 4000 * (i % 4 + 1)] for i, s in enumerate(samples)]

    client = GroqClient()
    server_counts: list[int] = []
    for i, s in enumerate(samples, 1):
        usage = client.probe_prompt_tokens(s)
        server_counts.append(usage)
        console.print(f"  probe {i}/{len(samples)}: server={usage} local={counter.count(s)}")

    result = calibrate(counter, samples, server_counts)
    console.print()
    console.print(result.render())


@app.command()
def retrieve(
    queries: list[str] = typer.Argument(..., help="One or more queries."),
    k: int = typer.Option(None, help=f"Top-K (default {settings.top_k})."),
    show: int = typer.Option(0, help="Print the top N chunk bodies in full."),
) -> None:
    """Run hybrid retrieval and report real token counts."""
    from .retrieve import Retriever
    from .timing import Timeline, new_query_clock
    from .tokens import get_counter

    counter = get_counter()
    r = Retriever.load()
    console.print(f"[dim]Index: {r.describe()}[/dim]")

    warm = r.warmup()
    console.print(
        f"[dim]Warm-up: {warm:.1f} ms — discarded, excluded from every timing below.[/dim]"
    )
    console.print(f"[dim]Tokenizer: {counter.encoding_name}[/dim]")

    for query in queries:
        tl = Timeline(label="retrieval", t0_ns=new_query_clock())
        hits = r.retrieve(query, k=k, timeline=tl)

        total = sum(h.chunk.token_count for h in hits)
        dense_only = sum(1 for h in hits if h.bm25_rank is None)
        bm25_only = sum(1 for h in hits if h.dense_rank is None)
        both = sum(1 for h in hits if h.dense_rank and h.bm25_rank)

        table = Table(title=f"\nTop-{len(hits)}  ·  {query}", header_style="bold", title_justify="left")
        table.add_column("#", justify="right")
        table.add_column("source")
        table.add_column("RRF", justify="right")
        table.add_column("dense", justify="right")
        table.add_column("bm25", justify="right")
        table.add_column("tokens", justify="right")
        for i, h in enumerate(hits, 1):
            table.add_row(
                str(i),
                h.source,
                f"{h.rrf_score:.5f}",
                "-" if h.dense_rank is None else str(h.dense_rank),
                "-" if h.bm25_rank is None else str(h.bm25_rank),
                f"{h.chunk.token_count:,}",
            )
        console.print(table)
        console.print(
            f"  baseline context -> [bold]{total:,} tokens[/bold] in {len(hits)} chunks   "
            f"| fusion: {both} both, {dense_only} dense-only, {bm25_only} bm25-only   "
            f"| retrieval_ms [bold]{tl.duration_ms('retrieval_ms'):.1f}[/bold]"
        )

        for h in hits[:show]:
            console.print(f"\n[bold cyan]--- {h.source} ({h.chunk.token_count} tokens) ---[/bold cyan]")
            console.print(h.chunk.text[:700] + ("…" if len(h.chunk.text) > 700 else ""))


@app.command()
def compress(
    queries: list[str] = typer.Argument(..., help="One or more queries."),
    budget: int = typer.Option(None, help=f"Token budget (default {settings.token_budget})."),
    k: int = typer.Option(None, help=f"Top-K (default {settings.top_k})."),
    lam: float = typer.Option(None, "--lambda", help=f"MMR lambda (default {settings.mmr_lambda})."),
    threshold: float = typer.Option(None, help="Relevance threshold."),
    strip: bool = typer.Option(None, help="Aggressive word stripping (default off)."),
    show: int = typer.Option(0, help="Print the first N lines of each context."),
) -> None:
    """Run stages [2]-[6] and report the compression achieved."""
    from .pipeline import Compressor

    c = Compressor.load()
    warm = c.warmup()
    console.print(
        "[dim]Warm-up (discarded): "
        + ", ".join(f"{k_}={v:.0f}ms" for k_, v in warm.items())
        + "[/dim]"
    )

    summary = Table(header_style="bold", title="\nCompression summary", title_justify="left")
    for col, just in (
        ("query", "left"), ("baseline", "right"), ("compressed", "right"),
        ("saved", "right"), ("ratio", "right"), ("sent kept", "right"),
        ("overhead", "right"),
    ):
        summary.add_column(col, justify=just)

    for q in queries:
        r = c.run(
            q, budget=budget, top_k=k, mmr_lambda=lam,
            relevance_threshold=threshold, aggressive_strip=strip,
        )
        ratio = r.compression_ratio
        summary.add_row(
            q if len(q) < 46 else q[:43] + "...",
            f"{r.baseline_context_tokens:,}",
            f"{r.compressed_context_tokens:,}",
            f"{r.tokens_saved:,}",
            "—" if ratio is None else f"{ratio * 100:.1f}%",
            f"{len(r.kept)}/{len(r.sentences)}",
            f"{r.pipeline_overhead_ms:.0f}ms",
        )

        stage = Table(header_style="bold", title=f"\n  stages · {q[:60]}", title_justify="left")
        stage.add_column("stage")
        stage.add_column("ms", justify="right")
        for name, val in r.timeline.to_dict().items():
            stage.add_row(name, "—" if val is None else f"{val:.1f}")
        console.print(stage)

        reasons: dict[str, int] = {}
        for s in r.dropped:
            raw = s.drop_reason or "?"
            # Group by reason TYPE, not by the specific sentence cited.
            key = "redundant" if raw.startswith("redundant") else raw.split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            console.print(
                "  dropped: "
                + ", ".join(f"[bold]{v}[/bold] {k_}" for k_, v in sorted(reasons.items()))
            )
        cs = r.cache_stats
        console.print(
            f"  cache: cross-encoder {cs.get('ce_hits', 0)}h/{cs.get('ce_misses', 0)}m, "
            f"embeddings {cs.get('emb_hits', 0)}h/{cs.get('emb_misses', 0)}m"
        )
        if r.strip_tokens_saved is not None:
            console.print(f"  aggressive strip saved: {r.strip_tokens_saved} tokens")

        if show:
            for label, body in (
                ("BASELINE (what the LLM would have received)", r.baseline_context),
                ("COMPRESSED (what it actually receives)", r.compressed_context),
            ):
                console.print(f"\n[bold cyan]--- {label} ---[/bold cyan]")
                console.print("\n".join(body.splitlines()[:show]))

    console.print(summary)


if __name__ == "__main__":
    app()
