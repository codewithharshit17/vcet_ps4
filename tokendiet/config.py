"""Single source of truth for every tunable in Token-Diet.

Rule: no other module may hardcode a budget, lambda, threshold, K, model name,
or price. If you find yourself typing a number somewhere else, it belongs here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QDRANT_LOCAL_PATH = DATA_DIR / "qdrant"
CACHE_LOCAL_PATH = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_prefix="TOKENDIET_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- LLM
    # Groq is OpenAI-compatible, so we drive it with the `openai` SDK.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Answers being measured.
    answer_model: str = "openai/gpt-oss-120b"
    # Blinded 1-5 quality judge. Deliberately a cheaper model; see PLAN.md Q4.
    judge_model: str = "openai/gpt-oss-20b"

    # Identical on both paths so output length is not structurally biased.
    max_output_tokens: int = 1024

    # ------------------------------------------------------------ PRICING
    # INPUT CONSTANTS, NOT MEASUREMENTS. Verified 2026-08-19 from
    #   https://console.groq.com/docs/model/openai/gpt-oss-120b
    #   https://console.groq.com/docs/model/openai/gpt-oss-20b
    # Units: USD per 1,000,000 tokens.
    answer_price_input_per_mtok: float = 0.15
    answer_price_output_per_mtok: float = 0.60
    judge_price_input_per_mtok: float = 0.075
    judge_price_output_per_mtok: float = 0.30

    # -------------------------------------------------------------- MODELS
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # tiktoken encoding used for the LOCAL token counter. This is a candidate
    # list, not an assertion: tokens.py probes it at runtime and calibrates the
    # winner against the server's own `usage.prompt_tokens`. See PLAN.md Q2.
    tokenizer_encoding_candidates: tuple[str, ...] = ("o200k_harmony", "o200k_base")

    # ------------------------------------------------------------ INGEST
    chunk_target_tokens: int = 400
    chunk_overlap_ratio: float = 0.15
    min_corpus_chunks: int = 200  # ingest fails loudly below this

    # ----------------------------------------------------------- RETRIEVAL
    top_k: int = 20  # deliberate over-fetch; recall is cheap when compression follows
    rrf_k: int = 60  # score = sum(1 / (rrf_k + rank_i))

    # --------------------------------------------------------- COMPRESSION
    token_budget: int = 800
    mmr_lambda: float = 0.7
    relevance_threshold: float = 0.0

    # Cosine similarity above which a sentence is called a duplicate of one
    # already kept. MMR's lambda alone only *demotes* redundancy; an explicit
    # threshold is what lets the dashboard say "redundant with S4 (sim 0.91)".
    mmr_redundancy_threshold: float = 0.85

    # Stage [3] score blend. w_bm25=0 reproduces the brief's cross-encoder-only
    # pipeline, so the eval can measure whether the deck's BM25 term earns its keep.
    score_weight_cross_encoder: float = 0.7
    score_weight_bm25: float = 0.3

    min_sentence_tokens: int = 4  # kept anyway if it has a numeral or proper noun
    cross_encoder_batch_size: int = 32

    # Word-level stripping. OFF by default and measured separately: dropping
    # modifiers inverts meaning ("non-critical failure", "unapproved vendor").
    aggressive_strip: bool = False

    # ------------------------------------------------------------- RUNTIME
    # Unset -> embedded Qdrant at QDRANT_LOCAL_PATH (no Docker required).
    qdrant_url: str | None = None
    qdrant_collection: str = "tokendiet"

    # Unset -> on-disk cache fallback (no Docker required).
    redis_url: str | None = None

    # ---------------------------------------------------------------- EVAL
    eval_budget_sweep: tuple[int | None, ...] = (200, 400, 800, 1600, None)  # None = infinity
    eval_runs_per_cell: int = 5  # every headline metric is a median over N>=5

    # -------------------------------------------------------------- CORPUS
    sec_user_agent: str = Field(
        default="Token-Diet Research contact@example.com", alias="SEC_USER_AGENT"
    )


settings = Settings()
