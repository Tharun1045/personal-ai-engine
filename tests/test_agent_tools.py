from unittest.mock import patch
from src.personal_ai_engine.domain.document import DocumentChunk
from src.online.tools.retrieval_tool import RetrievalTool
from src.online.tools.search_tool import SearchTool


@patch("src.online.tools.retrieval_tool.DocumentRetriever")
def test_retrieval_tool(mock_retriever):
    mock_inst = mock_retriever.return_value
    mock_inst.retrieve.return_value = [
        (DocumentChunk(parent_id="p1", content="mock contents"), 0.99)
    ]

    tool = RetrievalTool(strategy="hybrid", top_k=1)
    res = tool.run("my test query")

    assert "mock contents" in res
    assert "0.9900" in res
    mock_inst.retrieve.assert_called_once()


def test_search_tool():
    tool = SearchTool()
    res = tool.run("hello")
    assert "Search results for 'hello'" in res
