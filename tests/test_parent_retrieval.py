from unittest.mock import patch
from src.personal_ai_engine.domain.document import (
    Document,
    DocumentMetadata,
    DocumentChunk,
)
from src.online.retrieval.parent_retrieval import retrieve_parent_documents


@patch("src.online.retrieval.parent_retrieval.MongoDBClient")
def test_retrieve_parent_documents(mock_mongo_client):
    mock_db = mock_mongo_client.return_value.__enter__.return_value

    meta = DocumentMetadata(
        id="parent1", url="http://parent", title="Parent Document", properties={}
    )
    parent_doc = Document(id="parent1", metadata=meta, content="Full text here")
    mock_db.fetch_documents.return_value = [parent_doc]

    chunks = [DocumentChunk(id="c1", parent_id="parent1", content="chunk text")]

    result = retrieve_parent_documents(chunks)

    assert len(result) == 1
    assert result[0].id == "parent1"
    assert result[0].content == "Full text here"
    mock_db.fetch_documents.assert_called_once_with({"id": {"$in": ["parent1"]}})
