import json
from typing import List, Dict
from loguru import logger
from src.gateway.factory import AIGatewayFactory
from src.personal_ai_engine.domain.document import Document
from src.personal_ai_engine.domain.prompts import QA_GENERATION_PROMPT


class QADatasetGenerator:
    def __init__(self, provider: str | None = None, model: str | None = None):
        """Initialize the QA dataset generator.

        Args:
            provider: AI provider override.
            model: Model ID override.
        """
        self.generator = AIGatewayFactory.get_chat_generator(
            provider=provider, model=model
        )

    def generate_pairs(
        self, document: Document, num_pairs: int = 3
    ) -> List[Dict[str, str]]:
        """Generate QA instruction-response pairs from a document.

        Args:
            document: The input document.
            num_pairs: Number of pairs to generate.

        Returns:
            List[Dict[str, str]]: List of Alpaca-format QA pairs.
        """
        if not document.content or len(document.content.strip()) < 50:
            logger.warning(
                f"Document {document.id} content too short. Skipping QA generation."
            )
            return []

        prompt = QA_GENERATION_PROMPT.format(
            num_pairs=num_pairs,
            document=document.content[:8192],  # limit document content
        )

        try:
            response_text = self.generator.generate(prompt)

            # Clean possible markdown wrapping
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            pairs = json.loads(cleaned_text)

            # Add input field to fit standard Alpaca format if not present
            for pair in pairs:
                if "input" not in pair:
                    pair["input"] = ""

            return pairs
        except Exception as e:
            logger.error(f"QA dataset generation failed for doc {document.id}: {e}")
            return []
