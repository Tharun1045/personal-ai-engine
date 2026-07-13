from loguru import logger
from src.gateway.factory import AIGatewayFactory
from src.personal_ai_engine.domain.prompts import CONTEXTUAL_CHUNK_PROMPT


class ContextualRetriever:
    def __init__(self, provider: str | None = None, model: str | None = None):
        """Initialize the contextualizer.

        Args:
            provider: LLM provider override.
            model: LLM model override.
        """
        self.generator = AIGatewayFactory.get_chat_generator(
            provider=provider, model=model
        )

    def contextualize(self, doc_content: str, chunk_content: str) -> str:
        """Use LLM to generate 1-2 sentence context explaining how the chunk fits in the document.

        Args:
            doc_content: The full content of the parent document.
            chunk_content: The content of the chunk.

        Returns:
            str: Contextualized chunk content (context prepended to chunk content).
        """
        prompt = CONTEXTUAL_CHUNK_PROMPT.format(
            document_content=doc_content[
                :16384
            ],  # limit document length to avoid context overflow
            chunk_content=chunk_content,
        )
        try:
            context = self.generator.generate(prompt).strip()
            # Prepend context to original chunk content
            return f"{context}\n\n{chunk_content}"
        except Exception as e:
            logger.warning(
                f"Contextual chunking failed: {e}. Falling back to original chunk content."
            )
            return chunk_content
