from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    AI_PROVIDER: Literal["ollama", "openrouter", "gemini"] = "ollama"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen3:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_CHAT_MODEL: str = "meta-llama/llama-3-8b-instruct:free"

    GEMINI_API_KEY: str | None = None
    GEMINI_CHAT_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"


settings = AppSettings()
