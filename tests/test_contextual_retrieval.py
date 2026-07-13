from unittest.mock import patch, MagicMock
from src.offline.etl.contextual_retrieval import ContextualRetriever


@patch("src.offline.etl.contextual_retrieval.AIGatewayFactory.get_chat_generator")
def test_contextualizer_success(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.return_value = "This is a document about dogs."
    mock_get_chat.return_value = mock_chat_generator

    retriever = ContextualRetriever(provider="ollama")
    result = retriever.contextualize("document content", "chunk content")

    assert "This is a document about dogs." in result
    assert "chunk content" in result
    mock_chat_generator.generate.assert_called_once()


@patch("src.offline.etl.contextual_retrieval.AIGatewayFactory.get_chat_generator")
def test_contextualizer_fallback(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.side_effect = Exception("LLM connection error")
    mock_get_chat.return_value = mock_chat_generator

    retriever = ContextualRetriever(provider="ollama")
    result = retriever.contextualize("document content", "chunk content")

    # Should fall back to original chunk content without raising
    assert result == "chunk content"
