import pytest
import requests
from src.shared.config import settings
from src.gateway.factory import AIGatewayFactory


@pytest.mark.integration
def test_ollama_service_health():
    response = requests.get(settings.OLLAMA_BASE_URL)
    assert response.status_code == 200


@pytest.mark.integration
def test_ollama_models_availability():
    response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
    assert response.status_code == 200
    models = [model["name"] for model in response.json().get("models", [])]

    assert settings.OLLAMA_CHAT_MODEL in models, (
        f"Model {settings.OLLAMA_CHAT_MODEL} is not available in Ollama"
    )
    assert settings.OLLAMA_EMBEDDING_MODEL in models, (
        f"Model {settings.OLLAMA_EMBEDDING_MODEL} is not available in Ollama"
    )


@pytest.mark.integration
def test_ollama_chat_generation():
    generator = AIGatewayFactory.get_chat_generator()
    response = generator.generate("Say hello")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
def test_ollama_text_embedding():
    embedder = AIGatewayFactory.get_text_embedder()
    embedding = embedder.embed("Test embedding")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)
