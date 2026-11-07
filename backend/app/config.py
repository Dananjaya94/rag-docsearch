from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set with an environment variable of the same
    name in upper case (e.g. DATABASE_URL, EMBEDDER)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://docsearch:docsearch@localhost:5432/docsearch"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = 25

    # "queue" hands ingestion to the RQ worker; "inline" runs it in the request (tests, demos).
    ingest_mode: Literal["queue", "inline"] = "queue"

    # "sentence-transformers" for real semantic search, "hashing" for an offline, dependency-free
    # lexical embedder (used in CI).
    embedder: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 1024

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
