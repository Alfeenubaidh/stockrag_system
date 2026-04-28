from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sec_filings"

    # Embedder
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 128

    # Retrieval
    retrieval_fetch_k: int = 50
    retrieval_score_threshold: float = 0.0

    # Groq / generator
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b-instruct-q4_0"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Scheduler
    watchlist_tickers: list[str] = Field(default_factory=list)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    api_keys: str = ""  # comma-separated; empty string disables auth (dev only)


settings = Settings()
