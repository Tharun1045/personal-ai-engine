from unittest.mock import patch, MagicMock
from src.personal_ai_engine.domain.document import DocumentChunk
from src.offline.etl.embedding import EmbeddingGenerator


@patch("src.offline.etl.embedding.AIGatewayFactory.get_text_embedder")
def test_embedding_generator_batch(mock_get_embedder):
    mock_embedder = MagicMock()
    # Mock embed_batch to return dummy embeddings
    mock_embedder.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_get_embedder.return_value = mock_embedder

    chunks = [
        DocumentChunk(parent_id="doc1", content="chunk 1"),
        DocumentChunk(parent_id="doc1", content="chunk 2"),
    ]

    generator = EmbeddingGenerator(provider="ollama")
    result = generator.embed_chunks(chunks, batch_size=2)

    assert len(result) == 2
    assert result[0].embedding == [0.1, 0.2]
    assert result[1].embedding == [0.3, 0.4]
    mock_embedder.embed_batch.assert_called_once_with(["chunk 1", "chunk 2"])


@patch("src.offline.etl.embedding.AIGatewayFactory.get_text_embedder")
def test_embedding_generator_fallback(mock_get_embedder):
    mock_embedder = MagicMock()
    # Let embed_batch fail, fallback to single embed
    mock_embedder.embed_batch.side_effect = Exception("Batch failed")
    mock_embedder.embed.side_effect = [[0.9], [0.8]]
    mock_get_embedder.return_value = mock_embedder

    chunks = [
        DocumentChunk(parent_id="doc1", content="chunk 1"),
        DocumentChunk(parent_id="doc1", content="chunk 2"),
    ]

    generator = EmbeddingGenerator(provider="ollama")
    result = generator.embed_chunks(chunks, batch_size=2)

    assert len(result) == 2
    assert result[0].embedding == [0.9]
    assert result[1].embedding == [0.8]
    assert mock_embedder.embed.call_count == 2
