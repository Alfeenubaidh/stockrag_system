from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/config/settings.py → project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sec_filings"
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")

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

    # Remote embeddings
    use_remote_embeddings: bool = Field(default=False, env="USE_REMOTE_EMBEDDINGS")
    hf_api_key: str = Field(default="", env="HF_API_KEY")
    hf_api_token: str = Field(default="", env="HF_API_KEY")

    # Paths
    raw_pdfs_dir: Path = _PROJECT_ROOT / "data" / "raw" / "pdfs"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    api_keys: str = ""  # comma-separated; empty string disables auth (dev only)


settings = Settings()
