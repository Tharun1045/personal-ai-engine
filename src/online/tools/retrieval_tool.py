from loguru import logger
from src.personal_ai_engine.domain.queries import SearchQuery
from src.online.retrieval.retriever import DocumentRetriever


class RetrievalTool:
    def __init__(self, strategy: str = "hybrid", top_k: int = 5):
        self.retriever = DocumentRetriever(strategy=strategy, top_k=top_k)

    def run(self, query_text: str) -> str:
        """Run the retrieval tool to fetch relevant context from the user's second brain.

        Args:
            query_text: The search query text.

        Returns:
            str: Formatted context string with source citations.
        """
        logger.info(f"RetrievalTool executing query: {query_text}")
        query = SearchQuery(text=query_text)

        try:
            results = self.retriever.retrieve(query)
            if not results:
                return "No relevant context found in stored documents."

            context_blocks = []
            for rank, (chunk, score) in enumerate(results):
                title = chunk.metadata.get("title", "Untitled")
                url = chunk.metadata.get("url", "N/A")
                block = (
                    f"[{rank + 1}] Source: {title} ({url})\n"
                    f"Relevance Score: {score:.4f}\n"
                    f"Content:\n{chunk.content}\n"
                )
                context_blocks.append(block)

            return "\n---\n".join(context_blocks)
        except Exception as e:
            logger.error(f"RetrievalTool execution failed: {e}")
            return f"Error executing retrieval: {e}"
