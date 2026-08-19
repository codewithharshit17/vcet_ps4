# Token-Diet — Dynamic Post-Retrieval Context Compressor

**Problem statement:** Smart Context Compression
**Domain:** Application Data Search (GenAI / RAG Optimization)
**Team:** Qtiyapa — Aneesh Pardikar, Om Mane, Harshit Jaiswar
**College:** VES Institute of Technology, Chembur

Token-Diet sits **between a vector database and an LLM**. It cuts retrieved
context down to the information-dense sentences actually needed to answer a
query, then **proves the win with measured numbers** on a live dashboard.

It is not a compression library. It is an evidence machine: for every query it
runs the uncompressed baseline **and** the compressed path, live, in the same
request, and shows you what changed in tokens, latency, cost, and answer quality.

> **On the numbers in this README.** Every figure below was measured on this
> machine by a command you can re-run. Metrics that have not been measured yet
> are written as `—`, never as a plausible-looking placeholder. See
> [What is measured, and what isn't](#what-is-measured-and-what-isnt).

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Commands](#commands)
- [Running the dashboard](#running-the-dashboard)
- [Configuration](#configuration)
- [Measured results](#measured-results)
- [What is measured, and what isn't](#what-is-measured-and-what-isnt)
- [Where this approach loses](#where-this-approach-loses)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
User query
   │
   ├─► [1] Hybrid retrieval  ────────────────► top-K chunks (K=20, over-fetch on purpose)
   │       dense (Qdrant) + BM25, fused by
   │       Reciprocal Rank Fusion: Σ 1/(60+rank)
   │                        │
   │                        ▼
   │        ┌── Token-Diet compression engine ────────────┐
   │        │ [2] Sentence decomposition                  │
   │        │     tables / code / bullets stay ATOMIC     │
   │        │ [3] Parallel hybrid scoring                 │
   │        │     cross-encoder (semantic) + BM25 (lexical)│
   │        │ [4] MMR redundancy suppression (λ)          │
   │        │ [5] Token-budget knapsack (score-per-token) │
   │        │ [5b] Word stripping — OFF by default        │
   │        │ [6] Reassembly: source order, [...], cites  │
   │        └────────────────────┬────────────────────────┘
   │                             │
   ├─► BASELINE PATH ────────────┼──────────► LLM (streaming)
   │   (raw top-K, uncompressed) │
   └──────────► COMPRESSED PATH ─┴──────────► LLM (streaming)
                                              │
                                              ▼
                                   Metrics collector → dashboard
```

**Both paths run for every query, concurrently, from a single shared
`t_query_submitted` stamp.** That means the compressed path's time-to-first-token
includes all of our own retrieval, scoring, MMR and selection cost. Compression
has to pay for itself.

### Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 · FastAPI · SSE streaming |
| Vector DB | **Qdrant** (embedded by default, server via Docker) |
| Cache | **Redis** (falls back to on-disk SQLite) |
| Models | Hugging Face — `all-MiniLM-L6-v2` (embed), `ms-marco-MiniLM-L-6-v2` (cross-encoder) |
| Lexical | `rank_bm25` |
| Sentence split | `syntok` |
| Tokenizer | `tiktoken` `o200k_harmony` |
| LLM | Groq (OpenAI-compatible), `openai/gpt-oss-120b` |
| Frontend | React · Vite · TypeScript · Tailwind · **Chart.js** |
| Packaging | Docker Compose |

Everything local runs on **CPU**. No GPU required.

---

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | **3.11** | Pinned. `uv` installs it for you if missing. |
| [uv](https://docs.astral.sh/uv/) | latest | Package/venv manager. `pip install uv` or `winget install astral-sh.uv` |
| Node.js | 20+ | Only needed to build the dashboard. |
| Docker | optional | Only for the Qdrant + Redis services. **The app runs fine without it.** |
| `make` | optional | Convenience only. Windows users: use the `uv run` commands, which are listed alongside every target. |

**Disk:** ~3 GB (PyTorch + models + corpus).
**Network:** needed once, to download models and the SEC filings.

---

## Setup

### 1. Install dependencies

```bash
uv sync --python 3.11 --extra api
```

This creates `.venv/` and installs everything. First run downloads PyTorch, so
expect a few minutes.

### 2. Add your API key

The **only** manual step.

```bash
cp .env.example .env
```

Open `.env` and paste in a free Groq key from
**<https://console.groq.com/keys>** (no credit card required):

```ini
GROQ_API_KEY=gsk_your_key_here

# SEC EDGAR rejects requests without a real contact string.
SEC_USER_AGENT=Your Name your.email@example.com
```

> Retrieval and compression work **without** a key. The key is only needed for
> the LLM A/B (`ab`), the tokenizer calibration, and the eval harness. Without
> it the dashboard still runs and simply reports LLM metrics as `—`.

### 3. Download the corpus

```bash
uv run tokendiet fetch-corpus          # make corpus
```

Downloads the latest 10-K annual reports for AAPL, MSFT, JPM, JNJ and KO from
SEC EDGAR into `data/corpus/`. These are public-domain, enormous, and
pathologically repetitive — which is the point: the same risk-factor boilerplate
recurs within a filing *and* across companies, so there is real redundancy to
remove.

Custom tickers:

```bash
uv run tokendiet fetch-corpus --tickers TSLA,NVDA,WMT
```

You can also drop your own `.txt` / `.md` / `.pdf` files into `data/corpus/`.

### 4. Build the index

```bash
uv run tokendiet ingest                # make ingest
```

Chunks at ~400 tokens with 15% overlap, respecting paragraph boundaries and
never splitting a table, then embeds and indexes into Qdrant.

Ingest **fails loudly** if the corpus yields fewer than 200 chunks — a corpus
small enough to make retrieval trivial would flatter every downstream number.

### 5. Build the dashboard (optional)

```bash
cd web && npm install && npm run build   # make web-build
```

### One-shot setup

```bash
make dev        # setup + corpus + ingest + web-build
```

---

## Commands

Every command works as `uv run tokendiet <cmd>`. `make` targets are shown where
they exist.

| Command | What it does |
|---|---|
| `fetch-corpus` | Download 10-K filings from SEC EDGAR |
| `ingest` | Chunk + embed + index into Qdrant |
| `retrieve` | Hybrid retrieval with real token counts |
| `compress` | Run stages [2]–[6], report compression achieved |
| `ab` | **Live A/B** — both paths, median + p95 TTFT |
| `calibrate-tokenizer` | Prove the local tokenizer against the server |

### `retrieve` — see what the LLM *would* have received

```bash
uv run tokendiet retrieve "What are the principal risk factors related to supply chain?"
```

Accepts several queries at once (models load once):

```bash
uv run tokendiet retrieve "query one" "query two" "query three"
```

| Option | Default | Meaning |
|---|---|---|
| `--k` | 20 | Top-K chunks to retrieve |
| `--show N` | 0 | Print the first N chunk bodies in full |

Prints per-chunk RRF score, the dense and BM25 rank that produced it (`-` means
that ranker did not surface the chunk at all), token counts, and `retrieval_ms`.

### `compress` — before/after

```bash
uv run tokendiet compress "What was total net sales for the most recent fiscal year?"
```

| Option | Default | Meaning |
|---|---|---|
| `--budget` | 800 | Token budget for the compressed context |
| `--k` | 20 | Top-K chunks |
| `--lambda` | 0.7 | MMR λ — relevance vs. diversity |
| `--threshold` | 0.0 | Minimum relevance to survive |
| `--strip` | off | Aggressive word-level stripping |
| `--show N` | 0 | Print the first N lines of each context |

Reports per-stage timings, drop reasons, cache hit rates, and the compression
ratio.

### `ab` — the live A/B (**needs `GROQ_API_KEY`**)

```bash
uv run tokendiet ab "What are the principal risk factors?"        # make ab Q="..."
```

| Option | Default | Meaning |
|---|---|---|
| `--runs` | 5 | Runs per query. Headline metrics are a median over N≥5 with p95 alongside |
| `--budget` | 800 | Token budget |
| `--sequential` | off | Run paths one after another instead of concurrently |
| `--show-answers` | off | Print both answers |

Runs both paths **concurrently against the same endpoint in the same request**,
and prints context tokens, server-reported prompt/output tokens, TTFT, total
latency and cost for each path, plus the deltas.

### `calibrate-tokenizer` — prove the token counts

```bash
uv run tokendiet calibrate-tokenizer      # make calibrate
```

Token counts are the foundation of every headline metric, so the local tokenizer
is checked against the server's own `usage.prompt_tokens` across texts of
varying length. If the difference is **constant** with length, the tokenizer
agrees exactly and the constant is fixed chat-template overhead. If it **grows**
with length, the tokenizers genuinely disagree and the slope quantifies by how
much. The verdict is printed either way.

---

## Running the dashboard

```bash
uv run uvicorn tokendiet.api:app --host 127.0.0.1 --port 8000    # make api
```

Open **<http://127.0.0.1:8000>**.

FastAPI serves the built frontend, so there is one origin and no CORS setup.

For frontend development with hot reload, run the API on 8000 and in a second
terminal:

```bash
cd web && npm run dev     # http://localhost:5173, proxies /api to :8000
```

### What's on the dashboard

- **Hero row** — compression ratio, TTFT drop, cost per query, answer quality.
- **Context diff** — every candidate sentence. Kept ones tinted by relevance,
  dropped ones struck through. **Hover any sentence** for the exact reason:
  `low relevance (0.12)`, `redundant with JNJ_10K#29.4 (sim 0.91)`, `over budget`.
- **Latency waterfall** — stacked bars, baseline vs compressed, segmented by
  stage. Makes the overhead trade-off visible instead of hidden.
- **Live controls** — token budget, MMR λ, relevance threshold, top-K, and the
  aggressive-strip toggle. Re-runs without a reload; scores and embeddings are
  cached, so moving a slider is cheap.
- **Both answers side by side**, streaming live.
- **Eval tab** — quality-vs-compression curve and the gold-set table.

### Optional: Qdrant + Redis services

The app defaults to **embedded Qdrant** and an **on-disk cache**, so it runs
with Docker Desktop stopped. To use the deployed topology:

```bash
docker compose up -d          # make services
```

Then uncomment in `.env`:

```ini
TOKENDIET_QDRANT_URL=http://localhost:6333
TOKENDIET_REDIS_URL=redis://localhost:6379/0
```

Re-run `ingest` after switching, since the index lives in the new backend.

---

## Configuration

**One file: [`tokendiet/config.py`](tokendiet/config.py).** Budgets, λ,
thresholds, K, model names and pricing all live there. No other module hardcodes
any of them. Anything can be overridden by an env var with the `TOKENDIET_`
prefix:

```bash
TOKENDIET_TOKEN_BUDGET=400 uv run tokendiet compress "..."
```

| Setting | Default | Meaning |
|---|---|---|
| `top_k` | 20 | Chunks retrieved (deliberate over-fetch) |
| `token_budget` | 800 | Compressed context budget |
| `mmr_lambda` | 0.7 | Relevance vs. diversity |
| `mmr_redundancy_threshold` | 0.85 | Cosine above which a sentence is a duplicate |
| `relevance_threshold` | 0.0 | Minimum relevance to survive |
| `score_weight_cross_encoder` | 0.7 | Weight of the semantic signal |
| `score_weight_bm25` | 0.3 | Weight of the lexical signal — set to `0` to get a cross-encoder-only pipeline |
| `chunk_target_tokens` | 400 | Chunk size at ingest |
| `min_corpus_chunks` | 200 | Ingest refuses to build a trivially small index |
| `aggressive_strip` | `false` | Word-level stripping (see below) |
| `eval_runs_per_cell` | 5 | N for every median/p95 |

**Pricing constants** are inputs, not measurements. They are recorded in
`config.py` with their source URL and the date verified
(`openai/gpt-oss-120b`: $0.15 in / $0.60 out per 1M tokens, verified
2026-08-19).

### On aggressive word stripping

Both diagrams in the project abstract put "prune redundant words" in the main
path. It is implemented here **off by default, with its own toggle and its own
metric**, because removing modifiers is genuinely dangerous — *"non-critical
failure"*, *"unapproved vendor"* and *"third-quarter revenue"* all invert or lose
meaning when the modifier goes. It never touches structured content, never
removes a negation, and never removes a word adjacent to a numeral.

Turn it on and let the eval decide:

```bash
uv run tokendiet compress "..." --strip
```

---

## Measured results

Corpus: 5 SEC 10-K filings → **2,512 chunks / 1,088,876 tokens**.
Settings: K=20, budget=800, λ=0.7. Tokenizer: `o200k_harmony`.

| Query | Baseline | Compressed | Ratio | Sentences kept |
|---|---:|---:|---:|---:|
| Supply-chain / manufacturing risks | 12,869 | 882 | **93.1%** | 18 / 308 |
| Total net sales / revenue | 8,750 | 893 | **89.8%** | 23 / 122 |
| Climate risk × regulatory cost | 11,480 | 889 | **92.3%** | 22 / 268 |

**Pipeline overhead**, same machine, same run:

| | cold cache | warm cache |
|---|---:|---:|
| Supply-chain | 6,652 ms | **271 ms** |
| Net sales | 6,904 ms | **120 ms** |
| Climate | 6,054 ms | **225 ms** |

The 26–58× gap is why the cache is load-bearing. Cross-encoder scores and
sentence embeddings are both cached; an embedding is query-independent, so it is
reused by every query that retrieves the same chunk.

**Why the real tokenizer matters.** One chunk measures **403 tokens** under
`o200k_harmony`. `len(text.split())` says 189 and `chars/4` says 247 — the
shortcuts understate the baseline by 40–53%, which would inflate every
compression number downstream.

Reproduce all of the above:

```bash
uv run tokendiet compress \
  "What are the principal risk factors related to supply chain and manufacturing?" \
  "What was total net sales or revenue for the most recent fiscal year?" \
  "How do climate change risks and regulatory compliance costs interact across these companies?"
```

---

## What is measured, and what isn't

| Metric | Status |
|---|---|
| Compression ratio, tokens saved | ✅ measured |
| Per-stage pipeline timings | ✅ measured |
| Retrieval quality (RRF fusion) | ✅ measured |
| Token counts | ✅ real tokenizer |
| **TTFT / total latency** | ⏳ **needs `GROQ_API_KEY`** — run `make ab` |
| **Cost per query** | ⏳ needs a key (uses server-reported usage) |
| **Answer quality / fact retention** | ⏳ needs the eval harness |

Anything in the ⏳ rows renders as `—` in the UI and CLI. It is never estimated.

**Three deliberate guards against flattering numbers:**

1. **No remembered baselines.** Baseline and compressed run in the *same*
   request, concurrently, against the same endpoint. Launch order is randomised
   so neither path is systematically first.
2. **Prompt caching is defeated.** Groq caches prompt prefixes and bills them at
   half rate, so across N runs the baseline would hit cache and look faster and
   cheaper than it is. Every run prepends a unique nonce **first** in the prompt,
   and any run still reporting cached tokens is **discarded and counted**, not
   averaged in.
3. **Warm-up is explicit.** Models and the LLM connection are warmed before
   timing; those timings are reported separately and excluded from every
   distribution.

Headline metrics are a **median over N≥5 runs with p95 alongside**. The p95 uses
nearest-rank — it does not interpolate a smoother number than the sample
supports.

---

## Where this approach loses

Honest limitations, all found by measuring:

- **Tables are the weak spot.** Structured content survives decomposition intact
  and is never split — but real financial tables score only ~0.21 relevance
  (the cross-encoder is prose-trained and pipe-delimited tables score poorly),
  and MMR then penalises them further because near-identical tables across five
  filings mark each other redundant. **Questions whose answer is a number buried
  in a table are the most likely failure mode.** Not tuned around without
  evidence; the eval harness is what should quantify it.
- **Compression may lose on TTFT for small contexts.** Warm overhead is
  120–271 ms. Against a fast provider and a short prompt, that may exceed the
  time saved. Finding the crossover point is a result, not a failure — the
  latency waterfall is built to show exactly which stage spent the time.
- **The first query is slow.** Cold-cache overhead is ~6 s, almost entirely
  cross-encoder scoring and sentence embedding. Subsequent queries over the same
  chunks are 26–58× faster.
- **93% compression is budget-bound, not quality-bound.** These numbers come
  from asking for 800 tokens out of ~12k. The meaningful question is where
  quality falls off — that is the eval harness's curve, not this number.
- **Word stripping is unproven.** Built, guarded, and off by default on purpose.

---

## Project layout

```
tokendiet/
  config.py        ← THE config file: budgets, λ, thresholds, K, models, pricing
  tokens.py        real tokenizer + calibration against server ground truth
  store.py         Qdrant access (embedded or server)
  ingest.py        corpus → chunks → Qdrant
  retrieve.py      [1] dense + BM25 → RRF
  sentences.py     [2] decomposition; tables/code/lists stay atomic
  score.py         [3] cross-encoder + BM25, cached
  mmr.py           [4] redundancy suppression
  select.py        [5] token-budget knapsack
  strip.py         [5b] word stripping (off by default)
  assemble.py      [6] source-ordered reassembly with citations
  pipeline.py      orchestrates [2]–[6]
  dualpath.py      concurrent baseline vs compressed A/B
  metrics.py       every formula; median/p95; cost
  cache.py         Redis, or on-disk SQLite fallback
  timing.py        perf_counter spans; un-run stages are None, never 0
  api.py           FastAPI + SSE
  cli.py           command line
web/               React + Vite + TS + Tailwind + Chart.js
eval/              gold set + sweep harness
data/              corpus, index, cache (git-ignored)
```

---

## Troubleshooting

**`GROQ_API_KEY is not set`**
Expected without a key. `fetch-corpus`, `ingest`, `retrieve` and `compress` all
work regardless; only `ab`, `calibrate-tokenizer` and the eval need one.

**`No .txt/.md/.pdf files in data/corpus`**
Run `uv run tokendiet fetch-corpus` first.

**`Corpus produced only N chunks; need >= 200`**
Working as intended. Add more documents — a trivially small index would make
retrieval meaningless and inflate the results.

**`recreate=True but collection still holds N points`**
A guard against silently ingesting on top of stale data, which would create
duplicate chunks and flatter every compression number. Delete `data/qdrant/`
and re-run `ingest`.

**SEC returns 403**
EDGAR requires a real contact string. Set `SEC_USER_AGENT` in `.env` to
`Your Name your.email@example.com`.

**Hugging Face download errors / offline**
Models are cached after first use. To force offline: `HF_HUB_OFFLINE=1`.

**Garbled characters in the terminal**
The Windows console defaults to cp1252 and mangles the filings' typography.
The CLI reconfigures stdout to UTF-8; if you call the library directly, do the
same. The corpus itself is clean UTF-8.

**Dashboard shows `—` everywhere**
That is the intended behaviour when a metric has not been measured. Check the
banner: without `GROQ_API_KEY`, LLM metrics are unmeasured by design.

**Port 8000 already in use**
`uv run uvicorn tokendiet.api:app --port 8080`
