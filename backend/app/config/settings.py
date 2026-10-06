from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.paths import (
    get_arxiv_cache_dir,
    get_bm25_index_dir,
    get_openalex_cache_dir,
    get_processed_dir,
    get_project_root,
    get_raw_dir,
    get_research_history_dir,
)

_ENV_FILES = (get_project_root() / ".env", Path(".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="karanVerse Research Lab", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_processed_dir: Path = Field(
        default_factory=get_processed_dir,
        alias="DATA_PROCESSED_DIR",
    )
    data_raw_dir: Path = Field(default_factory=get_raw_dir, alias="DATA_RAW_DIR")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    chunk_min_size: int = Field(default=50, alias="CHUNK_MIN_SIZE")
    bm25_index_dir: Path = Field(
        default_factory=get_bm25_index_dir,
        alias="BM25_INDEX_DIR",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="research_chunks", alias="QDRANT_COLLECTION")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    vector_dimension: int = Field(default=384, alias="VECTOR_DIMENSION")
    vector_search_top_k: int = Field(default=10, alias="VECTOR_SEARCH_TOP_K")
    qdrant_indexing_enabled: bool = Field(default=True, alias="QDRANT_INDEXING_ENABLED")
    rrf_k: int = Field(default=60, alias="RRF_K")
    hybrid_bm25_weight: float = Field(default=1.0, alias="HYBRID_BM25_WEIGHT")
    hybrid_vector_weight: float = Field(default=1.0, alias="HYBRID_VECTOR_WEIGHT")
    hybrid_candidate_pool: int = Field(default=50, alias="HYBRID_CANDIDATE_POOL")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )
    rerank_candidate_count: int = Field(default=30, alias="RERANK_CANDIDATE_COUNT")
    rerank_final_top_k: int = Field(default=10, alias="RERANK_FINAL_TOP_K")
    llm_provider: Literal["openrouter", "omnirouter", "gemini", "ollama"] = Field(
        default="openrouter",
        alias="LLM_PROVIDER",
    )
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_retry_backoff_seconds: float = Field(default=1.0, alias="LLM_RETRY_BACKOFF_SECONDS")
    omnirouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OMNIROUTER_API_KEY", "OMNIROUTER_KEY"),
    )
    omnirouter_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OMNIROUTER_BASE_URL", "OMNIROUTER_API_BASE_URL"),
    )
    omnirouter_model: str = Field(
        default="google/gemini-2.0-flash-001",
        alias="OMNIROUTER_MODEL",
    )
    omnirouter_app_name: str | None = Field(default=None, alias="OMNIROUTER_APP_NAME")
    omnirouter_app_url: str | None = Field(default=None, alias="OMNIROUTER_APP_URL")
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "OPENROUTER_API_BASE_URL"),
    )
    openrouter_model: str = Field(
        default="openrouter/free",
        validation_alias=AliasChoices("OPENROUTER_MODEL", "OPENROUTER_DEFAULT_MODEL"),
    )
    openrouter_app_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_APP_NAME", "OPENROUTER_TITLE"),
    )
    openrouter_site_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_SITE_URL", "OPENROUTER_HTTP_REFERER"),
    )
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    openalex_base_url: str = Field(default="https://api.openalex.org", alias="OPENALEX_BASE_URL")
    openalex_mailto: str | None = Field(default=None, alias="OPENALEX_MAILTO")
    openalex_timeout_seconds: float = Field(default=30.0, alias="OPENALEX_TIMEOUT_SECONDS")
    openalex_cache_enabled: bool = Field(default=True, alias="OPENALEX_CACHE_ENABLED")
    openalex_cache_ttl_seconds: int = Field(default=86_400, alias="OPENALEX_CACHE_TTL_SECONDS")
    openalex_cache_dir: Path = Field(
        default_factory=get_openalex_cache_dir,
        alias="OPENALEX_CACHE_DIR",
    )
    arxiv_base_url: str = Field(default="https://export.arxiv.org/api", alias="ARXIV_BASE_URL")
    arxiv_timeout_seconds: float = Field(default=30.0, alias="ARXIV_TIMEOUT_SECONDS")
    arxiv_cache_enabled: bool = Field(default=True, alias="ARXIV_CACHE_ENABLED")
    arxiv_cache_ttl_seconds: int = Field(default=86_400, alias="ARXIV_CACHE_TTL_SECONDS")
    arxiv_cache_dir: Path = Field(
        default_factory=get_arxiv_cache_dir,
        alias="ARXIV_CACHE_DIR",
    )
    research_planner_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        alias="RESEARCH_PLANNER_TEMPERATURE",
    )
    research_agent_papers_per_query: int = Field(default=5, ge=1, le=50, alias="RESEARCH_AGENT_PAPERS_PER_QUERY")
    research_agent_document_pool: int = Field(default=20, ge=1, le=100, alias="RESEARCH_AGENT_DOCUMENT_POOL")
    research_agent_hybrid_top_k: int = Field(default=20, ge=1, le=100, alias="RESEARCH_AGENT_HYBRID_TOP_K")
    research_agent_evidence_top_k: int = Field(default=5, ge=1, le=20, alias="RESEARCH_AGENT_EVIDENCE_TOP_K")
    research_agent_min_evidence_hits: int = Field(default=1, ge=0, le=20, alias="RESEARCH_AGENT_MIN_EVIDENCE_HITS")
    research_agent_max_paper_title_queries: int = Field(
        default=3,
        ge=0,
        le=20,
        alias="RESEARCH_AGENT_MAX_PAPER_TITLE_QUERIES",
    )
    critic_temperature: float = Field(default=0.1, ge=0.0, le=2.0, alias="CRITIC_TEMPERATURE")
    critic_max_excerpt_chars: int = Field(default=500, ge=100, le=4000, alias="CRITIC_MAX_EXCERPT_CHARS")
    critic_min_evidence_count: int = Field(default=2, ge=0, le=50, alias="CRITIC_MIN_EVIDENCE_COUNT")
    critic_min_external_papers: int = Field(default=1, ge=0, le=50, alias="CRITIC_MIN_EXTERNAL_PAPERS")
    critic_low_retrieval_score_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        alias="CRITIC_LOW_RETRIEVAL_SCORE_THRESHOLD",
    )
    critic_duplicate_jaccard_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        alias="CRITIC_DUPLICATE_JACCARD_THRESHOLD",
    )
    synthesizer_temperature: float = Field(default=0.15, ge=0.0, le=2.0, alias="SYNTHESIZER_TEMPERATURE")
    synthesizer_max_excerpt_chars: int = Field(default=600, ge=100, le=4000, alias="SYNTHESIZER_MAX_EXCERPT_CHARS")
    research_max_iterations: int = Field(default=2, ge=1, le=5, alias="RESEARCH_MAX_ITERATIONS")
    neo4j_enabled: bool = Field(default=False, alias="NEO4J_ENABLED")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str | None = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str | None = Field(default=None, alias="NEO4J_DATABASE")
    research_history_dir: Path = Field(
        default_factory=get_research_history_dir,
        alias="RESEARCH_HISTORY_DIR",
    )
    research_history_context_top_k: int = Field(default=3, ge=0, le=10, alias="RESEARCH_HISTORY_CONTEXT_TOP_K")
    research_update_temperature: float = Field(
        default=0.15,
        ge=0.0,
        le=2.0,
        alias="RESEARCH_UPDATE_TEMPERATURE",
    )
    research_update_max_excerpt_chars: int = Field(
        default=500,
        ge=100,
        le=4000,
        alias="RESEARCH_UPDATE_MAX_EXCERPT_CHARS",
    )
    knowledge_gap_temperature: float = Field(
        default=0.15,
        ge=0.0,
        le=2.0,
        alias="KNOWLEDGE_GAP_TEMPERATURE",
    )
    knowledge_gap_max_excerpt_chars: int = Field(
        default=450,
        ge=100,
        le=4000,
        alias="KNOWLEDGE_GAP_MAX_EXCERPT_CHARS",
    )
    knowledge_gap_low_evidence_threshold: int = Field(
        default=3,
        ge=0,
        le=50,
        alias="KNOWLEDGE_GAP_LOW_EVIDENCE_THRESHOLD",
    )
    knowledge_gap_weak_score_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        alias="KNOWLEDGE_GAP_WEAK_SCORE_THRESHOLD",
    )
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    max_upload_bytes: int = Field(default=25_000_000, ge=1, alias="MAX_UPLOAD_BYTES")
    max_ingest_text_chars: int = Field(default=500_000, ge=1, alias="MAX_INGEST_TEXT_CHARS")
    pdf_max_pages: int = Field(default=500, ge=1, alias="PDF_MAX_PAGES")
    research_allow_private_knowledge: bool = Field(
        default=False,
        alias="RESEARCH_ALLOW_PRIVATE_KNOWLEDGE",
        description="Master switch: allow private corpus in research when the client also opts in.",
    )
    external_llm_allow_private_knowledge: bool = Field(
        default=False,
        alias="EXTERNAL_LLM_ALLOW_PRIVATE_KNOWLEDGE",
        description="Allow private/project chunks in prompts sent to non-local LLM providers.",
    )
    api_auth_enabled: bool = Field(default=False, alias="API_AUTH_ENABLED")
    api_key: str | None = Field(default=None, alias="API_KEY")
    cors_allow_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ALLOW_ORIGINS",
    )

    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
