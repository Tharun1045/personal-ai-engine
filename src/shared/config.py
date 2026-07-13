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

    MONGO_URI: str = "mongodb://root:rootpassword@localhost:27017"
    MONGO_DB_NAME: str = "personal_ai_engine"

    ZENML_STORE_URL: str = "http://localhost:8080"

    NOTION_SECRET_KEY: str | None = None
    NOTION_DATABASE_ID: str | None = None
    NOTION_API_VERSION: str = "2022-06-28"
    NOTION_PAGE_SIZE: int = 100
    NOTION_TIMEOUT: int = 15
    NOTION_RETRIES: int = 3
    NOTION_BACKOFF: float = 2.0

    CRAWLER_CONCURRENCY: int = 5
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_MAX_RESPONSE_SIZE: int = 10485760  # 10MB
    CRAWLER_ALLOWED_DOMAINS: list[str] = []
    CRAWLER_BLOCKED_DOMAINS: list[str] = []
    CRAWLER_USER_AGENT: str = "PersonalAIEngineBot/1.0"

    QUALITY_SCORE_THRESHOLD: float = 0.5
    MONGO_DB_LOAD_COLLECTION: str = "documents"


settings = AppSettings()
