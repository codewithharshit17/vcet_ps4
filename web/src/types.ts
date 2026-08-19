export type Sentence = {
  sid: string;
  doc_id: string;
  chunk_id: number;
  char_offset: number;
  text: string;
  tokens: number;
  kind: "prose" | "table" | "code" | "list";
  relevance: number;
  ce_norm: number;
  bm25_norm: number;
  mmr_score: number | null;
  similarity: number | null;
  redundant_with: string | null;
  selected: boolean;
  drop_reason: string | null;
};

export type Stages = Record<string, number | null>;

export type Compression = {
  query: string;
  baseline_context_tokens: number;
  compressed_context_tokens: number;
  tokens_saved: number;
  compression_ratio: number | null;
  budget: number | null;
  pipeline_overhead_ms: number;
  stages: Stages;
  cache_stats: Record<string, number>;
  strip_tokens_saved: number | null;
  baseline_context: string;
  compressed_context: string;
  sentences: Sentence[];
};

/** Every field nullable on purpose: unmeasured must render as an em dash. */
export type PathState = {
  answer: string;
  ttft_ms: number | null;
  total_latency_ms: number | null;
  context_tokens: number | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  cached_prompt_tokens: number | null;
  contaminated: boolean;
  error: string | null;
  done: boolean;
};

export const emptyPath = (): PathState => ({
  answer: "",
  ttft_ms: null,
  total_latency_ms: null,
  context_tokens: null,
  prompt_tokens: null,
  output_tokens: null,
  cached_prompt_tokens: null,
  contaminated: false,
  error: null,
  done: false,
});

export type Config = {
  token_budget: number;
  mmr_lambda: number;
  relevance_threshold: number;
  top_k: number;
  aggressive_strip: boolean;
  answer_model: string;
  embed_model: string;
  cross_encoder_model: string;
  price_in_per_mtok: number;
  price_out_per_mtok: number;
  warmup_ms: Record<string, number>;
  llm_configured: boolean;
  budget_sweep: (number | null)[];
};
