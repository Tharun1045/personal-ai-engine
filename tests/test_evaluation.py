from unittest.mock import patch, MagicMock
from src.online.evaluation.evaluation import RAGEvaluator


@patch("src.online.evaluation.evaluation.AIGatewayFactory.get_chat_generator")
def test_evaluation_faithfulness_success(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.return_value = (
        '{"faithful": true, "reason": "all grounded"}'
    )
    mock_get_chat.return_value = mock_chat_generator

    evaluator = RAGEvaluator(provider="ollama")
    score = evaluator.evaluate_faithfulness("context text", "faithful answer")

    assert score == 1.0
    mock_chat_generator.generate.assert_called_once()


@patch("src.online.evaluation.evaluation.AIGatewayFactory.get_chat_generator")
def test_evaluation_relevance_success(mock_get_chat):
    mock_chat_generator = MagicMock()
    mock_chat_generator.generate.return_value = (
        '{"score": 0.85, "reason": "highly relevant"}'
    )
    mock_get_chat.return_value = mock_chat_generator

    evaluator = RAGEvaluator(provider="ollama")
    score = evaluator.evaluate_relevance("question", "answer")

    assert score == 0.85
    mock_chat_generator.generate.assert_called_once()


def test_evaluation_mock_mode():
    evaluator = RAGEvaluator(mock=True)
    assert evaluator.evaluate_faithfulness("a", "b") == 1.0
    assert evaluator.evaluate_relevance("a", "b") == 0.9
    assert evaluator.evaluate_context_precision("a", "b") == 1.0
