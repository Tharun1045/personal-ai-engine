import json
from loguru import logger
from src.gateway.factory import AIGatewayFactory


class RAGEvaluator:
    def __init__(
        self, provider: str | None = None, model: str | None = None, mock: bool = False
    ):
        self.mock = mock
        self.generator = AIGatewayFactory.get_chat_generator(
            provider=provider, model=model
        )

    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """Evaluate if the generated answer is faithful to (grounded in) the retrieved context.

        Returns:
            float: 1.0 if faithful, 0.0 if not.
        """
        if self.mock:
            return 1.0

        prompt = (
            "You are an evaluator. Given the CONTEXT and the ANSWER below, determine if the answer "
            "contains any statements that are NOT supported by the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Return only a valid JSON response:\n"
            '{"faithful": true, "reason": "why"}'
        )
        try:
            res = self.generator.generate(prompt)
            # clean json wrapping
            cleaned = res.strip().strip("`").strip("json").strip()
            data = json.loads(cleaned)
            return 1.0 if data.get("faithful") is True else 0.0
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5

    def evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate if the generated answer is relevant to the question.

        Returns:
            float: Score between 0.0 and 1.0.
        """
        if self.mock:
            return 0.9

        prompt = (
            "You are an evaluator. Given the QUESTION and the ANSWER below, score the relevance "
            "of the answer to the question on a scale from 0.0 (completely irrelevant) to 1.0 (completely relevant).\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Return only a valid JSON response:\n"
            '{"score": 0.9, "reason": "why"}'
        )
        try:
            res = self.generator.generate(prompt)
            cleaned = res.strip().strip("`").strip("json").strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.5

    def evaluate_context_precision(self, question: str, context: str) -> float:
        """Evaluate if the retrieved context contains relevant information for the question.

        Returns:
            float: Score between 0.0 and 1.0.
        """
        if self.mock:
            return 1.0

        prompt = (
            "You are an evaluator. Given the QUESTION and the CONTEXT below, score the precision "
            "of the context (i.e. whether the context contains the necessary info to answer the question) "
            "on a scale from 0.0 (contains no useful info) to 1.0 (contains all needed info).\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Return only a valid JSON response:\n"
            '{"score": 1.0, "reason": "why"}'
        )
        try:
            res = self.generator.generate(prompt)
            cleaned = res.strip().strip("`").strip("json").strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except Exception as e:
            logger.warning(f"Context precision evaluation failed: {e}")
            return 0.5
