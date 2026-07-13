from src.shared.config import settings
from src.gateway.interfaces import ChatGenerator, TextEmbedder
from src.gateway.adapters.ollama_adapter import OllamaChatGenerator, OllamaTextEmbedder
from src.gateway.adapters.openrouter_adapter import (
    OpenRouterChatGenerator,
    OpenRouterTextEmbedder,
)
from src.gateway.adapters.gemini_adapter import GeminiChatGenerator, GeminiTextEmbedder


class AIGatewayFactory:
    @staticmethod
    def get_chat_generator(
        provider: str | None = None, model: str | None = None
    ) -> ChatGenerator:
        active_provider = provider or settings.AI_PROVIDER
        if active_provider == "ollama":
            active_model = model or settings.OLLAMA_CHAT_MODEL
            return OllamaChatGenerator(
                base_url=settings.OLLAMA_BASE_URL, model=active_model
            )
        elif active_provider == "openrouter":
            active_model = model or settings.OPENROUTER_CHAT_MODEL
            return OpenRouterChatGenerator(
                api_key=settings.OPENROUTER_API_KEY or "",
                model=active_model,
            )
        elif active_provider == "gemini":
            active_model = model or settings.GEMINI_CHAT_MODEL
            return GeminiChatGenerator(
                api_key=settings.GEMINI_API_KEY or "", model=active_model
            )
        else:
            raise ValueError(f"Unsupported AI_PROVIDER: {active_provider}")

    @staticmethod
    def get_text_embedder(
        provider: str | None = None, model: str | None = None
    ) -> TextEmbedder:
        active_provider = provider or settings.EMBEDDING_PROVIDER
        if active_provider == "ollama":
            active_model = model or settings.OLLAMA_EMBEDDING_MODEL
            return OllamaTextEmbedder(
                base_url=settings.OLLAMA_BASE_URL, model=active_model
            )
        elif active_provider == "gemini":
            active_model = model or settings.GEMINI_EMBEDDING_MODEL
            return GeminiTextEmbedder(
                api_key=settings.GEMINI_API_KEY or "",
                model=active_model,
            )
        elif active_provider == "openrouter":
            # openrouter doesn't natively support text embeddings in our app
            active_model = model or settings.OPENROUTER_CHAT_MODEL
            return OpenRouterTextEmbedder(
                api_key=settings.OPENROUTER_API_KEY or "",
                model=active_model,
            )
        else:
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {active_provider}")
