from unittest.mock import patch, MagicMock
from src.personal_ai_engine.domain.document import Document, DocumentMetadata
from src.offline.training.dataset_generation import QADatasetGenerator


@patch("src.offline.training.dataset_generation.AIGatewayFactory.get_chat_generator")
def test_dataset_generation_success(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.return_value = """
    [
      {
        "instruction": "What is the capital of France?",
        "output": "Paris"
      }
    ]
    """
    mock_get_chat.return_value = mock_chat_generator

    meta = DocumentMetadata(
        id="doc1", url="http://france", title="France Doc", properties={}
    )
    doc = Document(
        id="doc1",
        metadata=meta,
        content="France is a country in Europe. Its capital is Paris.",
    )

    generator = QADatasetGenerator(provider="ollama")
    result = generator.generate_pairs(doc, num_pairs=1)

    assert len(result) == 1
    assert result[0]["instruction"] == "What is the capital of France?"
    assert result[0]["output"] == "Paris"
    assert result[0]["input"] == ""  # Added for Alpaca compliance
    mock_chat_generator.generate.assert_called_once()


@patch("src.offline.training.dataset_generation.AIGatewayFactory.get_chat_generator")
def test_dataset_generation_failure(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.side_effect = Exception("Generation error")
    mock_get_chat.return_value = mock_chat_generator

    meta = DocumentMetadata(
        id="doc1", url="http://france", title="France Doc", properties={}
    )
    doc = Document(
        id="doc1",
        metadata=meta,
        content="France is a country in Europe. Its capital is Paris.",
    )

    generator = QADatasetGenerator(provider="ollama")
    result = generator.generate_pairs(doc, num_pairs=1)

    # Should gracefully return empty list
    assert result == []
