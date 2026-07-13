from unittest.mock import patch
import pytest

from src.personal_ai_engine.domain import DocumentMetadata
from src.offline.notion.database import NotionDatabaseClient
from src.offline.notion.document import NotionDocumentClient


@pytest.fixture
def mock_notion_env(monkeypatch):
    monkeypatch.setenv("NOTION_SECRET_KEY", "test_key")


@patch("src.offline.notion.database.requests.post")
def test_query_notion_database(mock_post, mock_notion_env):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "results": [
            {
                "id": "1",
                "url": "https://notion.so/1",
                "properties": {
                    "Name": {
                        "type": "title",
                        "title": [{"plain_text": "Test Page"}],
                    }
                },
            }
        ]
    }

    client = NotionDatabaseClient(api_key="test_key")
    results = client.query_notion_database("db_id")

    assert len(results) == 1
    assert isinstance(results[0], DocumentMetadata)
    assert results[0].id == "1"
    assert results[0].title == "Test Page"


@patch("src.offline.notion.document.requests.get")
def test_extract_document(mock_get, mock_notion_env):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "results": [
            {
                "id": "b1",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Hello world!"}]},
            }
        ]
    }

    client = NotionDocumentClient(api_key="test_key")
    meta = DocumentMetadata(
        id="1", url="https://notion.so/1", title="Test Page", properties={}
    )

    doc = client.extract_document(meta)

    assert doc.id == "1"
    assert "Hello world!" in doc.content
