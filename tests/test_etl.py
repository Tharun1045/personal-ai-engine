import pytest
from src.personal_ai_engine.domain.document import Document, DocumentMetadata
from src.offline.etl.scoring import HeuristicQualityAgent
from src.offline.steps.etl.clean import clean_content
from src.offline.mongo_client import MongoDBClient


def test_heuristic_quality_scoring():
    meta = DocumentMetadata(id="1", url="http", title="Test", properties={})

    # High URL content ratio
    doc1 = Document(
        id="1", metadata=meta, content="a " * 20, child_urls=["http://test.com/a" * 5]
    )
    agent = HeuristicQualityAgent()
    doc1_scored = agent(doc1)
    assert doc1_scored.content_quality_score == 0.0

    # Low URL content ratio
    doc2 = Document(
        id="2", metadata=meta, content="a " * 1000, child_urls=["http://test.com/a"]
    )
    doc2_scored = agent(doc2)
    assert doc2_scored.content_quality_score == 1.0


def test_content_cleaning():
    raw_content = "This   is   a\n\n\n\ntest."
    cleaned, content_hash = clean_content(raw_content)
    assert (
        cleaned == "This is a\n\nTest." or cleaned == "This is a\n\ntest."
    )  # Actually regex replaces spaces but leaves \n\n


@pytest.mark.integration
def test_mongo_idempotent_writes():
    meta = DocumentMetadata(id="1", url="http", title="Test", properties={})
    doc = Document(id="1", metadata=meta, content="test")

    with MongoDBClient(model=Document, collection_name="test_idempotent") as client:
        client.clear_collection()
        client.ingest_documents([doc])
        client.ingest_documents([doc])  # Re-ingest

        count = client.get_collection_count()
        assert count == 1  # Duplicate was avoided
