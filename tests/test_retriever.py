from unittest.mock import patch, MagicMock
from src.personal_ai_engine.domain.document import DocumentChunk
from src.personal_ai_engine.domain.queries import SearchQuery
from src.online.retrieval.retriever import DocumentRetriever


@patch("src.online.retrieval.retriever.AIGatewayFactory.get_text_embedder")
@patch("src.online.retrieval.retriever.MongoDBClient")
def test_retriever_keyword(mock_mongo_client, mock_get_embedder):
    mock_db = mock_mongo_client.return_value.__enter__.return_value
    mock_db.text_search.return_value = [
        (DocumentChunk(parent_id="p1", content="result"), 1.0)
    ]

    retriever = DocumentRetriever(strategy="keyword")
    query = SearchQuery(text="test query", strategy="keyword", top_k=2)
    results = retriever.retrieve(query)

    assert len(results) == 1
    assert results[0][0].content == "result"
    assert results[0][1] == 1.0
    mock_db.text_search.assert_called_once_with("test query", top_k=2, filters={})


@patch("src.online.retrieval.retriever.AIGatewayFactory.get_text_embedder")
@patch("src.online.retrieval.retriever.MongoDBClient")
def test_retriever_semantic(mock_mongo_client, mock_get_embedder):
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2]
    mock_get_embedder.return_value = mock_embedder

    mock_db = mock_mongo_client.return_value.__enter__.return_value
    mock_db.vector_search.return_value = [
        (DocumentChunk(parent_id="p1", content="semantic result"), 0.95)
    ]

    retriever = DocumentRetriever(strategy="semantic")
    query = SearchQuery(text="semantic query", strategy="semantic", top_k=5)
    results = retriever.retrieve(query)

    assert len(results) == 1
    assert results[0][0].content == "semantic result"
    assert results[0][1] == 0.95
    mock_embedder.embed.assert_called_once_with("semantic query")
    mock_db.vector_search.assert_called_once_with([0.1, 0.2], top_k=5, filters={})


@patch("src.online.retrieval.retriever.AIGatewayFactory.get_text_embedder")
@patch("src.online.retrieval.retriever.MongoDBClient")
def test_retriever_hybrid(mock_mongo_client, mock_get_embedder):
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.5]
    mock_get_embedder.return_value = mock_embedder

    mock_db = mock_mongo_client.return_value.__enter__.return_value
    mock_db.hybrid_search.return_value = [
        (DocumentChunk(parent_id="p1", content="hybrid result"), 0.033)
    ]

    retriever = DocumentRetriever(strategy="hybrid")
    query = SearchQuery(text="hybrid query", strategy="hybrid")
    results = retriever.retrieve(query)

    assert len(results) == 1
    assert results[0][0].content == "hybrid result"
    mock_db.hybrid_search.assert_called_once_with(
        "hybrid query", [0.5], top_k=5, filters={}
    )
