import pytest
from src.shared.config import AppSettings


def test_default_config():
    settings = AppSettings()
    assert settings.AI_PROVIDER == "ollama"
    assert settings.OLLAMA_CHAT_MODEL == "qwen3:8b"
    assert settings.OLLAMA_EMBEDDING_MODEL == "nomic-embed-text:latest"


def test_config_provider_validation():
    # pydantic literal validation
    with pytest.raises(ValueError):
        AppSettings(AI_PROVIDER="invalid")


def test_openrouter_config():
    settings = AppSettings(AI_PROVIDER="openrouter")
    assert settings.AI_PROVIDER == "openrouter"


def test_gemini_config():
    settings = AppSettings(AI_PROVIDER="gemini")
    assert settings.AI_PROVIDER == "gemini"
