import json
from typing import List, Union
from loguru import logger
from src.personal_ai_engine.domain import Document
from src.gateway.factory import AIGatewayFactory


class HeuristicQualityAgent:
    def __call__(
        self, documents: Union[Document, List[Document]]
    ) -> Union[Document, List[Document]]:
        if isinstance(documents, Document):
            docs = [documents]
            is_single = True
        else:
            docs = documents
            is_single = False

        scored = [self.__score_document(d) for d in docs]
        return scored[0] if is_single else scored

    def __score_document(self, document: Document) -> Document:
        if not document.content:
            return document.add_quality_score(score=0.0)

        # Simple heuristic based on URL to text ratio
        url_based_content = sum(len(url) for url in document.child_urls)
        url_content_ratio = url_based_content / max(len(document.content), 1)

        if url_content_ratio >= 0.7:
            return document.add_quality_score(score=0.0)
        elif url_content_ratio >= 0.5:
            return document.add_quality_score(score=0.2)

        return document


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
    "score": <your score between 0.0 and 1.0>
}}

DOCUMENT:
{document}
"""

    def __init__(
        self,
        model_id: str = "qwen3:8b",
        mock: bool = False,
        max_concurrent_requests: int = 10,
    ):
        self.mock = mock
        self.model_id = model_id
        # Enforce Ollama provider as per requirements
        self.chat_generator = AIGatewayFactory.get_chat_generator(
            provider="ollama", model=model_id
        )

    def __call__(
        self, documents: Union[Document, List[Document]]
    ) -> Union[Document, List[Document]]:
        if isinstance(documents, Document):
            docs = [documents]
            is_single = True
        else:
            docs = documents
            is_single = False

        scored = []
        for doc in docs:
            scored.append(self.__score_document(doc))

        return scored[0] if is_single else scored

    def __score_document(self, document: Document) -> Document:
        if self.mock:
            return document.add_quality_score(score=0.8)

        prompt = self.SYSTEM_PROMPT_TEMPLATE.format(document=document.content[:8192])
        try:
            # We use the generic chat generator instead of direct LiteLLM
            response_text = self.chat_generator.generate(prompt)
            dict_content = json.loads(response_text)
            score = float(dict_content.get("score", 0.5))
            return document.add_quality_score(score=score)
        except Exception as e:
            logger.warning(f"Failed to score document {document.id}: {str(e)}")
            # Fallback to a default score
            return document.add_quality_score(score=0.5)
