from unittest.mock import patch, MagicMock
from src.personal_ai_engine.domain.document import DocumentChunk
from src.online.ui.gradio_app import (
    check_infrastructure_status,
    answer_question,
    build_app,
)


@patch("src.online.ui.gradio_app.requests.get")
@patch("src.online.ui.gradio_app.MongoDBClient")
def test_check_infrastructure_status(mock_mongo, mock_get):
    # Mock MongoDB Online
    mock_db = mock_mongo.return_value.__enter__.return_value
    mock_db.ping.return_value = True

    # Mock Ollama Online
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    status = check_infrastructure_status()
    assert "🟢 Online" in status["mongodb"]
    assert "🟢 Ollama Online" in status["provider"]


@patch("src.online.ui.gradio_app.AIGatewayFactory.get_chat_generator")
@patch("src.online.ui.gradio_app.DocumentRetriever")
@patch("src.online.ui.gradio_app.check_infrastructure_status")
def test_answer_question_flow(mock_status, mock_retriever, mock_chat_factory):
    # Mock Online status
    mock_status.return_value = {"mongodb": "🟢 Online", "provider": "🟢 Online"}

    # Mock retriever results
    mock_retriever_inst = mock_retriever.return_value
    mock_retriever_inst.retrieve.return_value = [
        (
            DocumentChunk(
                parent_id="doc1",
                content="Paris is the capital.",
                metadata={"title": "Paris", "url": "http://paris"},
            ),
            0.98,
        )
    ]

    # Mock Chat generator
    mock_generator = MagicMock()
    mock_generator.generate_with_system.return_value = "The capital is Paris."
    mock_chat_factory.return_value = mock_generator

    history, cleared_msg, sources_html, metadata_str = answer_question(
        "What is the capital?", "Hybrid", []
    )

    assert len(history) == 1
    assert history[0][0] == "What is the capital?"
    assert history[0][1] == "The capital is Paris."
    assert "Paris" in sources_html
    assert "doc1" in metadata_str


def test_build_app():
    # Make sure app blocks build without raising exception
    with patch("src.online.ui.gradio_app.check_infrastructure_status") as mock_status:
        mock_status.return_value = {"mongodb": "🟢 Online", "provider": "🟢 Online"}
        app = build_app()
        assert app is not None
        assert app.title == "Personal AI Engine"
