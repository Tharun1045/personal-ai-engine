import json
from loguru import logger
from src.personal_ai_engine.domain import Document
from src.gateway.factory import AIGatewayFactory


class HeuristicQualityAgent:
    def __init__(self):
        self.version = "heuristic-v1"

    def __call__(
        self, documents: Document | list[Document]
    ) -> Document | list[Document]:
        is_single = isinstance(documents, Document)
        docs = [documents] if is_single else documents
        scored = [self.__score_document(d) for d in docs]
        return scored[0] if is_single else scored

    def __score_document(self, document: Document) -> Document:
        url_based_content = sum(len(url) for url in document.child_urls)
        word_count = len(document.content.split())
        url_content_ratio = url_based_content / max(len(document.content), 1)

        reasons = []
        if url_content_ratio >= 0.7:
            score = 0.0
            reasons.append("Too many URLs relative to text")
        elif url_content_ratio >= 0.5:
            score = 0.2
            reasons.append("High URL ratio")
        elif word_count < 10:
            score = 0.0
            reasons.append("Content too short")
        else:
            score = 1.0

        document.quality_assessment = {
            "version": self.version,
            "total_score": score,
            "signal_scores": {
                "url_ratio": 1.0 - min(url_content_ratio, 1.0),
                "length": min(word_count / 100, 1.0),
            },
            "classification": "relevant" if score >= 0.5 else "irrelevant",
            "rejection_reasons": reasons,
        }
        return document.add_quality_score(score=score)


class QualityScoreAgent:
    SYSTEM_PROMPT_TEMPLATE = """You are an expert judge tasked with evaluating the quality of a given DOCUMENT.

Guidelines:
1. Evaluate the DOCUMENT based on generally accepted facts and reliable information.
2. Evaluate that the DOCUMENT contains relevant information and not only links or error messages.
3. Check that the DOCUMENT doesn't oversimplify or generalize information in a way that changes its meaning or accuracy.

Analyze the text thoroughly and assign a quality score between 0 and 1, where:
- **0.0**: completely irrelevant containing only noise
- **0.1 - 0.7**: partially relevant
- **0.8 - 1.0**: entirely relevant

It is crucial that you return only the score in the following JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "reasons": ["reason 1", "reason 2"]
}}

DOCUMENT:
{document}
"""

    def __init__(self, mock: bool = False, max_concurrent_requests: int = 1):
        self.mock = mock
        self.version = "ollama-qwen3:8b-v1"
        self.chat_generator = AIGatewayFactory.get_chat_generator(provider="ollama")

    def __call__(
        self, documents: Document | list[Document]
    ) -> Document | list[Document]:
        is_single = isinstance(documents, Document)
        docs = [documents] if is_single else documents

        scored = []
        for doc in docs:
            scored.append(self.__score_document(doc))

        return scored[0] if is_single else scored

    def __score_document(self, document: Document) -> Document:
        if self.mock:
            score = 0.8
            reasons = []
        else:
            prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                document=document.content[:8192]
            )
            try:
                response_text = self.chat_generator.generate(prompt)
                dict_content = json.loads(response_text)
                score = float(dict_content.get("score", 0.5))
                reasons = dict_content.get("reasons", [])
            except Exception as e:
                logger.warning(f"Failed to score document {document.id}: {str(e)}")
                score = 0.5
                reasons = ["LLM error"]

        document.quality_assessment = {
            "version": self.version,
            "total_score": score,
            "signal_scores": {
                "llm_relevance": score,
            },
            "classification": "relevant" if score >= 0.5 else "irrelevant",
            "rejection_reasons": reasons,
        }
        return document.add_quality_score(score=score)
