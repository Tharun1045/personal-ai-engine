from typing import Literal, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    AI_PROVIDER: Literal["ollama", "openrouter", "gemini"] = "ollama"
    EMBEDDING_PROVIDER: Literal["ollama", "gemini"] = "ollama"

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen3:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"

    # OpenRouter Settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_CHAT_MODEL: str = "meta-llama/llama-3-8b-instruct:free"

    # Gemini Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_CHAT_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # MongoDB Settings
    MONGO_URI: str = "mongodb://root:rootpassword@localhost:27017"
    MONGO_DB_NAME: str = "personal_ai_engine"
    MONGO_DB_LOAD_COLLECTION: str = "documents"
    MONGO_CHUNKS_COLLECTION: str = "chunks"
    MONGO_VECTOR_INDEX_NAME: str = "vector_index"

    # ZenML Settings
    ZENML_STORE_URL: str = "http://localhost:8080"

    # Notion Settings
    NOTION_SECRET_KEY: Optional[str] = None
    NOTION_DATABASE_ID: Optional[str] = None
    NOTION_API_VERSION: str = "2022-06-28"
    NOTION_PAGE_SIZE: int = 100
    NOTION_TIMEOUT: int = 15
    NOTION_RETRIES: int = 3
    NOTION_BACKOFF: float = 2.0

    # Crawler Settings
    CRAWLER_CONCURRENCY: int = 5
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_MAX_RESPONSE_SIZE: int = 10485760  # 10MB
    CRAWLER_ALLOWED_DOMAINS: List[str] = []
    CRAWLER_BLOCKED_DOMAINS: List[str] = []
    CRAWLER_USER_AGENT: str = "PersonalAIEngineBot/1.0"

    # Quality Settings
    QUALITY_SCORE_THRESHOLD: float = 0.5

    # RAG & Retrieval Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_STRATEGY: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    VECTOR_SEARCH_BACKEND: Literal["local", "atlas"] = "local"

    # Hugging Face Settings
    HF_TOKEN: Optional[str] = None
    HF_USERNAME: Optional[str] = None

    # Comet ML Settings
    COMET_API_KEY: Optional[str] = None
    COMET_PROJECT_NAME: str = "personal-ai-engine"
    COMET_WORKSPACE: Optional[str] = None

    # Opik Settings
    OPIK_API_KEY: Optional[str] = None
    OPIK_PROJECT_NAME: str = "personal-ai-engine"

    # Gradio Settings
    GRADIO_PORT: int = 7860
    GRADIO_SHARE: bool = False


settings = AppSettings()
