from loguru import logger


class SearchTool:
    """Mock web search tool for LLM agent."""

    def run(self, query_text: str) -> str:
        logger.info(f"SearchTool executing query: {query_text}")
        # Return mock search results
        return (
            f"Search results for '{query_text}':\n"
            "1. Mock result 1: Decoding AI Course covers fine-tuning and evaluation.\n"
            "2. Mock result 2: ZenML is a extensible pipeline orchestrator.\n"
            "3. Mock result 3: MongoDB Atlas Vector Search supports cosine distance metrics."
        )
