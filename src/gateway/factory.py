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
    def get_chat_generator() -> ChatGenerator:
        provider = settings.AI_PROVIDER
        if provider == "ollama":
            return OllamaChatGenerator(
                base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_CHAT_MODEL
            )
        elif provider == "openrouter":
            return OpenRouterChatGenerator(
                api_key=settings.OPENROUTER_API_KEY or "",
                model=settings.OPENROUTER_CHAT_MODEL,
            )
        elif provider == "gemini":
            return GeminiChatGenerator(
                api_key=settings.GEMINI_API_KEY or "", model=settings.GEMINI_CHAT_MODEL
            )
        else:
            raise ValueError(f"Unsupported AI_PROVIDER: {provider}")

    @staticmethod
    def get_text_embedder() -> TextEmbedder:
        provider = settings.AI_PROVIDER
        if provider == "ollama":
            return OllamaTextEmbedder(
                base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
            )
        elif provider == "openrouter":
            return OpenRouterTextEmbedder(
                api_key=settings.OPENROUTER_API_KEY or "",
                model=settings.OPENROUTER_CHAT_MODEL,  # Not strictly embedding model, but as placeholder
            )
        elif provider == "gemini":
            return GeminiTextEmbedder(
                api_key=settings.GEMINI_API_KEY or "",
                model=settings.GEMINI_EMBEDDING_MODEL,
            )
        else:
            raise ValueError(f"Unsupported AI_PROVIDER: {provider}")
