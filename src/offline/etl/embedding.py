from typing import List
from loguru import logger
from src.gateway.factory import AIGatewayFactory
from src.personal_ai_engine.domain.document import DocumentChunk


class EmbeddingGenerator:
    def __init__(self, provider: str | None = None, model: str | None = None):
        """Initialize the embedding generator.

        Args:
            provider: Independent embedding provider override.
            model: Independent embedding model override.
        """
        self.embedder = AIGatewayFactory.get_text_embedder(
            provider=provider, model=model
        )

    def embed_chunks(
        self, chunks: List[DocumentChunk], batch_size: int = 16
    ) -> List[DocumentChunk]:
        """Embed a list of DocumentChunks in batches.

        Args:
            chunks: List of document chunks to embed.
            batch_size: Number of chunks to embed in one batch.

        Returns:
            List[DocumentChunk]: The chunks with their embedding field populated.
        """
        if not chunks:
            return []

        logger.info(
            f"Generating embeddings for {len(chunks)} chunks in batches of {batch_size}"
        )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            try:
                embeddings = self.embedder.embed_batch(texts)
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                logger.error(
                    f"Failed to generate embeddings for batch starting at {i}: {e}"
                )
                # Fallback to single embeds in case batching failed
                for chunk in batch:
                    try:
                        chunk.embedding = self.embedder.embed(chunk.content)
                    except Exception as inner_e:
                        logger.error(
                            f"Failed to generate embedding for chunk {chunk.id}: {inner_e}"
                        )
                        chunk.embedding = None

        return chunks
