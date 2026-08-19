# Token-Diet — PLAN.md

Phase 0 deliverable, reconciled against the team's abstract deck
(`6a7093bda5429_Abstract_PPT.pptx.pdf`).

**Problem Statement:** Smart Context Compression
**Domain:** Application Data Search (GenAI / RAG Optimization)
**Team:** Qtiyapa — Aneesh Pardikar, Om Mane, Harshit Jaiswar
**College:** VES Institute of Technology, Chembur

---

## 1. Stack

Taken from the deck's "Technologies used" slide. Where the build brief and the deck
disagreed, **the deck wins** — it's your own spec.

| Layer | Choice | Source | Note |
|---|---|---|---|
| Backend | **FastAPI** (Python 3.11 via `uv`) | deck + brief | SSE streaming for honest TTFT. |
| Vector DB | **Qdrant** (local, Docker, persisted) | **deck** | Brief suggested Chroma/FAISS — deck says Qdrant, and it pairs with the Docker item. Named vectors + payload filtering keep `doc_id`/`chunk_id`/`char_offset` on the point itself. |
| Cache | **Redis** | **deck** | Not in either deck diagram, so I'm assigning it the job the brief explicitly requires: the cross-encoder score cache keyed on `hash(query + sentence)`. Also caches sentence embeddings. Makes the λ / budget / threshold sliders re-run in milliseconds instead of re-scoring. |
| Models | **Hugging Face** — `all-MiniLM-L6-v2` (embed) + `ms-marco-MiniLM-L-6-v2` (cross-encoder) | deck + brief | CPU only, no GPU required. |
| Lexical | `rank_bm25` | deck ("Semantic + BM25 Scoring") | Used at **both** levels — see §3. |
| Sentence split | `syntok` | brief | Regex on `.` shreds legal/technical prose. |
| Frontend | **React + Vite + TypeScript + Tailwind** | deck + brief | |
| Charts | **Chart.js** | **deck** | Brief said Recharts; deck says Chart.js. Going with Chart.js. |
| Packaging | **Docker** (compose: qdrant + redis + api + web) | **deck** | This also fixes `make dev` on Windows — `docker compose up` is the portable entrypoint. `make` is not installed on this machine. |
| Tokenizer | **`tiktoken` (`o200k_harmony`) locally + server `usage` as ground truth** | resolved | See §4.1. Exact, local, and provable. |
| LLM | **Groq**, OpenAI-compatible, streaming, behind `LLMClient` | resolved | `openai/gpt-oss-120b` for answers, `openai/gpt-oss-20b` for the blinded judge. Free tier, fastest TTFT, `usage` returned on every call. |

---

## 2. Architecture (from the deck's two diagrams, merged with the brief's rigor)

```
User Query
   │
   ├─► Fetch Top-K paragraphs from Qdrant   (K=20, deliberate over-fetch)
   │        dense + BM25 fused by RRF: score = Σ 1/(60 + rank_i)
   │                    │
   │                    ▼
   │        ┌─ Token-Diet Compression Engine ──────────────┐
   │        │ [2] Segment paragraphs into sentences        │
   │        │     (tables / code / lists = atomic, never   │
   │        │      split mid-structure)                    │
   │        │                 │                            │
   │        │                 ▼                            │
   │        │ [3] PARALLEL HYBRID SCORING                  │
   │        │       ├─ BM25        (lexical keyword match) │
   │        │       └─ Cross-Encoder (dense semantic match)│
   │        │     aggregate → normalize [0,1] → threshold  │
   │        │                 │                            │
   │        │                 ▼                            │
   │        │ [4] MMR redundancy suppression (λ=0.7)       │
   │        │                 │                            │
   │        │                 ▼                            │
   │        │ [5] Token-budget knapsack (score/token)      │
   │        │                 │                            │
   │        │                 ▼                            │
   │        │ [5b] Word-level stripping  ── OFF by default │
   │        │                 │                            │
   │        │                 ▼                            │
   │        │ [6] Reassembly: source order, […] elisions,  │
   │        │     citation prefixes                        │
   │        └──────────────────┬───────────────────────────┘
   │                           │
   ├─► BASELINE PATH ──────────┼────────────► LLM (streaming)
   │   (raw top-K, uncompressed)             │
   └───────────────► COMPRESSED PATH ────────┴──► LLM (streaming)
                                                   │
                                                   ▼
                                    Metrics Collector → Analytics Dashboard
                                    (tokens saved, TTFT drop, cost, quality)
```

Both paths run for **every** query, concurrently, so the comparison is always live.

### Where I'm merging deck + brief in stage [3]
The deck specifies *parallel hybrid scoring of sentences* (BM25 ∥ cross-encoder →
aggregate → dynamic threshold). The brief specifies cross-encoder only at sentence
level, with BM25 used at chunk retrieval. These are compatible, so I'm doing both:

```
relevance(s) = w_ce · norm(cross_encoder(q, s)) + w_bm25 · norm(bm25(q, s))
```

`w_ce = 0.7`, `w_bm25 = 0.3` by default, both in `config.py`, both exposed as
dashboard sliders. Setting `w_bm25 = 0` gives exactly the brief's pipeline, so the
eval harness can measure whether the deck's BM25 term actually earns its place.
That's a result worth having.

---

## 3. Module boundaries

```
tokendiet/
  config.py       # THE config file: K, budgets, λ, thresholds, w_ce/w_bm25, models, pricing
  tokens.py       # TokenCounter protocol + impls + calibration harness
  ingest.py       # folder(.md/.txt/.pdf) -> 400-tok chunks, 15% overlap, paragraph-aware -> Qdrant
  retrieve.py     # [1] dense + BM25 -> RRF -> top-K
  sentences.py    # [2] split; structured-block detection (table/code/list atomic)
  score.py        # [3] parallel BM25 + cross-encoder, batched 32, normalized, Redis-cached
  mmr.py          # [4] MMR over sentence embeddings
  select.py       # [5] score-per-token knapsack, guarantees top-1 sentence
  strip.py        # [5b] word-level stripping — own toggle, OFF by default
  assemble.py     # [6] source-ordered reassembly + [...] + citations
  llm/base.py     # LLMClient protocol: stream(msgs) -> AsyncIterator[Token]
  llm/<provider>.py
  metrics.py      # timing spans, token accounting, cost, median/p95
  quality.py      # embedding drift, blinded LLM-judge, fact retention
  warmup.py       # explicit warm-up, timings recorded separately and discarded
  api.py          # FastAPI: POST /query -> SSE both paths + metrics
eval/
  gold.yaml       # >=20 queries: required_facts + difficulty label
  run_eval.py     # budget sweep -> results.json + stdout table
web/              # Vite + React + Tailwind + Chart.js
docker-compose.yml
```

Binding rules:
- `metrics.py` owns every number. No stage computes its own headline metric.
- Nothing imports a literal budget/λ/price. Everything reads `config.py`.
- Every metric field is `Optional`. Unmeasured renders `—`, never a plausible default.

---

## 4. Metric formulas (exactly as implemented)

```
compression_ratio    = 1 - (compressed_context_tokens / baseline_context_tokens)
tokens_saved         = baseline_context_tokens - compressed_context_tokens
pipeline_overhead_ms = t_end_of_reassembly - t_end_of_retrieval
ttft_ms              = t_first_token_received - t_query_submitted    # per path, incl. our overhead
ttft_delta_ms        = baseline_ttft_ms - compressed_ttft_ms         # positive = win
total_latency_ms     = t_last_token_received - t_query_submitted
cost_saved_usd       = tokens_saved * INPUT_PRICE_PER_TOKEN
                     + output_token_delta * OUTPUT_PRICE_PER_TOKEN
```

- `t_query_submitted` is stamped **once**, before either path starts, and shared by
  both. The compressed path's TTFT therefore carries our full retrieval + scoring +
  MMR + knapsack cost. That is the point — rule 4.
- Clock: `time.perf_counter_ns()`.
- Per-stage spans, both paths: `retrieval_ms`, `sentence_split_ms`, `rerank_ms`,
  `mmr_ms`, `select_ms`, `llm_ttft_ms`, `llm_stream_ms`. Baseline's compression spans
  are `null`, not `0` — it didn't run them.
- Headline = **median over N≥5, p95 alongside**. Single runs are labelled `n=1`.
- `output_token_delta` comes from the API's own `usage`, not counted client-side.
- **Prompt caching must be defeated — this is a live threat on Groq.** Groq applies
  automatic prefix caching and bills cached input at half rate (`$0.075` vs `$0.15`
  per MTok on `gpt-oss-120b`). Across N=5 runs the baseline path resends the same
  large context, so runs 2–5 would hit cache and report artificially low latency and
  cost — a remembered baseline wearing a live baseline's clothes, which rule 3 bans.
  Mitigation: every run prepends a unique run-nonce at the **front** of the prompt,
  invalidating the shared prefix so both paths are always cold. I additionally assert
  `usage.prompt_tokens_details.cached_tokens == 0` per run and mark any run that
  fails as contaminated rather than averaging it in. Both facts go in the README.
- Identical `max_tokens` on both paths so output length isn't structurally biased.

### Concurrency
`asyncio.gather` for both paths, with a code comment on contention risk. In M3 I
measure path-order-swapped runs; if TTFT variance exceeds the p95 band, I switch to
sequential-with-randomized-order and say so. Measured, not assumed.

### Warm-up
`warmup.py` loads embedder + cross-encoder and makes one throwaway streaming LLM
call at startup. Recorded as `warmup_ms` and explicitly excluded from all reported
distributions.

---

## 5. Build order

1. **M1** — config, tokens (+calibration), ingest, Qdrant, retrieval, timing scaffold, docker-compose. → *retrieval output + real token counts, 3 queries*
2. **M2** — sentences → score → mmr → select → assemble (+strip off), CLI. → *before/after context + compression ratio, 3 queries*
3. **M3** — LLM client, dual concurrent streaming, TTFT over 5 runs. → *median + p95 table*
4. **M4** — FastAPI SSE + React/Tailwind/Chart.js dashboard. → *screenshot*
5. **M5** — eval harness, budget sweep `[200,400,800,1600,∞]`, quality-vs-compression curve. → *curve + gold table*
6. **M6** — README: measured numbers **and** where compression loses.

Commit at each milestone; message states what was measured.

---

## 6. Deck items I'm flagging rather than silently following

**Word-level stripping.** Both deck diagrams put "Prune Redundant Words & Filler
Sentences" *in the main path*. The build brief says make it optional and off by
default, because dropping modifiers inverts meaning — "non-critical failure",
"unapproved vendor", "third-quarter revenue". I'm building it with its own toggle and
its own metric column so the eval can answer this with data instead of opinion. If
the numbers say it's safe, turning it on is a one-line config change. Say the word if
you want it on-path from the start.

**RAGAS.** Your references cite it for faithfulness / answer relevance. The brief's
quality metrics (embedding drift, blinded LLM-judge, fact retention) cover the same
ground without the extra dependency and extra API calls. I'd add RAGAS in M5 only if
you want it named explicitly in the report — tell me and I'll wire it in.

---

## 7. Assumptions

1. Retrieval + compression in-process; Qdrant and Redis via docker-compose.
2. Corpus ships as a fetch script + ingest, not vendored copyrighted text.
3. `required_facts` are exact-match substrings (case/whitespace-normalized). Judged completeness is a *separate* metric.
4. LLM-judge sees both answers with labels hidden and order randomized.
5. Pricing constants in `config.py` are **inputs, not measurements** — README says so.

---

## 8. Resolved decisions

**Q1 — LLM: Groq.** OpenAI-compatible, free tier, no card. Answer model
`openai/gpt-oss-120b`; judge model `openai/gpt-oss-20b`. Both 131,072-token context,
65,536 max output. Setup is one `GROQ_API_KEY` line in `.env`, satisfying the
"no manual steps beyond an API key" requirement. Streaming SSE gives real TTFT.

Note: Groq's production lineup is now the GPT-OSS family, **not** Llama — verified
against their live model docs rather than assumed.

**Pricing** (verified 2026-08-19 from Groq's model pages, recorded in `config.py`
with source URL and date, and labelled an *input constant, not a measurement*):

| Model | Input $/1M | Output $/1M |
|---|---|---|
| `openai/gpt-oss-120b` | 0.15 | 0.60 |
| `openai/gpt-oss-20b` | 0.075 | 0.30 |

**Q2 — Tokenizer: two-tier, and the error is provable.** Groq's choice makes this
much stronger than it would have been on Claude:

- **Ground truth** for every reported metric is `usage.prompt_tokens` returned by the
  actual API call being measured. Server-side, exact, zero extra requests, zero cost.
- **Local counter** for the knapsack's inner loop is `tiktoken` with the
  `o200k_harmony` encoding used by the GPT-OSS family — local, microseconds, no
  network.
- **Calibration harness** (`tokens.py`) measures local-vs-server disagreement across
  the whole corpus and publishes it in the README. I expect ~0 error, in which case
  the local counter *is* ground truth and rule 2 is satisfied with no approximation
  anywhere. If it isn't 0, the number gets published as measured, not hidden.

The exact encoding name is verified empirically in M1 against real `usage` values —
not asserted from memory.

**Q3 — Corpus: SEC 10-K annual reports** via EDGAR. Public domain, hundreds of pages
of boilerplate, repeated disclaimers, and figures buried in tables. Demos the
"Legal & Financial Document Analysis" use case named on your own feasibility slide.
Ships as a fetch script; ≥200 chunks enforced by the ingest script, which fails
loudly if the corpus is too small to make retrieval non-trivial.

**Q4 — Eval: full sweep, N=5, cheap judge.** `gpt-oss-120b` generates the answers
being measured; `gpt-oss-20b` runs the blinded 1–5 judge calls. At these prices a
full sweep is cents, not dollars, so N=5 distributions everywhere are affordable.
`run_eval.py` prints the **measured** eval cost from summed `usage`, not an estimate.
