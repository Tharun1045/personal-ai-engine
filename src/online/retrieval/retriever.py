from typing import List, Tuple, Optional
from loguru import logger
from src.shared.config import settings
from src.personal_ai_engine.domain.document import DocumentChunk
from src.personal_ai_engine.domain.queries import SearchQuery
from src.offline.mongo_client import MongoDBClient
from src.gateway.factory import AIGatewayFactory


class DocumentRetriever:
    def __init__(
        self,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the document retriever.

        Args:
            strategy: Search strategy override ('semantic', 'keyword', 'hybrid').
            top_k: Number of search results to return.
            embedding_provider: Embedding model provider override.
            embedding_model: Embedding model name override.
        """
        self.strategy = strategy or settings.RETRIEVAL_STRATEGY
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.embedder = AIGatewayFactory.get_text_embedder(
            provider=embedding_provider, model=embedding_model
        )

    def retrieve(self, query: SearchQuery) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve chunks based on the query configuration.

        Args:
            query: The search query configuration.

        Returns:
            List[Tuple[DocumentChunk, float]]: Chunks and similarity/relevance scores.
        """
        strategy = query.strategy or self.strategy
        top_k = query.top_k or self.top_k
        filters = query.filters

        logger.info(
            f"Retrieving using strategy '{strategy}' and top_k {top_k} for query: {query.text}"
        )

        with MongoDBClient(
            model=DocumentChunk, collection_name=settings.MONGO_CHUNKS_COLLECTION
        ) as client:
            if strategy == "keyword":
                return client.text_search(query.text, top_k=top_k, filters=filters)

            # Embed the query for semantic / hybrid
            query_embedding = self.embedder.embed(query.text)

            if strategy == "semantic":
                return client.vector_search(
                    query_embedding, top_k=top_k, filters=filters
                )
            elif strategy == "hybrid":
                return client.hybrid_search(
                    query.text, query_embedding, top_k=top_k, filters=filters
                )
            else:
                raise ValueError(f"Unknown search strategy: {strategy}")
