import pytest
from unittest.mock import patch
from src.gateway.factory import AIGatewayFactory
from src.gateway.adapters.ollama_adapter import OllamaChatGenerator, OllamaTextEmbedder
from src.gateway.adapters.openrouter_adapter import (
    OpenRouterChatGenerator,
    OpenRouterTextEmbedder,
)
from src.gateway.adapters.gemini_adapter import GeminiChatGenerator, GeminiTextEmbedder


@patch("src.gateway.factory.settings")
def test_factory_returns_ollama(mock_settings):
    mock_settings.AI_PROVIDER = "ollama"
    mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
    mock_settings.OLLAMA_CHAT_MODEL = "qwen3:8b"
    mock_settings.OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

    chat = AIGatewayFactory.get_chat_generator()
    assert isinstance(chat, OllamaChatGenerator)

    embedder = AIGatewayFactory.get_text_embedder()
    assert isinstance(embedder, OllamaTextEmbedder)


@patch("src.gateway.factory.settings")
def test_factory_returns_openrouter(mock_settings):
    mock_settings.AI_PROVIDER = "openrouter"
    mock_settings.OPENROUTER_API_KEY = "test_key"
    mock_settings.OPENROUTER_CHAT_MODEL = "meta-llama/llama-3-8b-instruct:free"

    chat = AIGatewayFactory.get_chat_generator()
    assert isinstance(chat, OpenRouterChatGenerator)

    embedder = AIGatewayFactory.get_text_embedder()
    assert isinstance(embedder, OpenRouterTextEmbedder)


@patch("src.gateway.factory.settings")
def test_factory_returns_gemini(mock_settings):
    mock_settings.AI_PROVIDER = "gemini"
    mock_settings.GEMINI_API_KEY = "test_key"
    mock_settings.GEMINI_CHAT_MODEL = "gemini-1.5-flash"
    mock_settings.GEMINI_EMBEDDING_MODEL = "text-embedding-004"

    chat = AIGatewayFactory.get_chat_generator()
    assert isinstance(chat, GeminiChatGenerator)

    embedder = AIGatewayFactory.get_text_embedder()
    assert isinstance(embedder, GeminiTextEmbedder)


@patch("src.gateway.factory.settings")
def test_factory_unsupported_provider(mock_settings):
    mock_settings.AI_PROVIDER = "invalid_provider"
    with pytest.raises(ValueError, match="Unsupported AI_PROVIDER"):
        AIGatewayFactory.get_chat_generator()


def test_openrouter_missing_key():
    with pytest.raises(ValueError, match="OpenRouter API key is missing"):
        OpenRouterChatGenerator(api_key="", model="test")


def test_gemini_missing_key():
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        GeminiChatGenerator(api_key="", model="test")


def test_openrouter_generate_raises():
    adapter = OpenRouterChatGenerator(api_key="test", model="test")
    with pytest.raises(NotImplementedError):
        adapter.generate("test")


def test_gemini_generate_raises():
    adapter = GeminiChatGenerator(api_key="test", model="test")
    with pytest.raises(NotImplementedError):
        adapter.generate("test")
