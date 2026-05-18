from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096

    # ── RAG ──────────────────────────────────────────────────
    # 日本語・多言語対応の軽量 E5 埋め込みモデル
    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    # ── ベクトルストア ──────────────────────────────────────
    vector_store_path: str = "vector_store"

    # ── エージェント ────────────────────────────────────────
    max_iterations: int = 10
    memory_window: int = 10       # 保持する直近ターン数

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
