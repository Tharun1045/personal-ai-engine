from typing import List
from loguru import logger
from src.shared.config import settings
from src.personal_ai_engine.domain.document import Document, DocumentChunk
from src.offline.mongo_client import MongoDBClient


def retrieve_parent_documents(chunks: List[DocumentChunk]) -> List[Document]:
    """Retrieve full parent documents for a list of retrieved chunks.

    Args:
        chunks: List of retrieved DocumentChunks.

    Returns:
        List[Document]: The parent documents associated with the chunks.
    """
    if not chunks:
        return []

    parent_ids = list({c.parent_id for c in chunks if c.parent_id})
    if not parent_ids:
        return []

    logger.info(
        f"Retrieving {len(parent_ids)} parent documents for the retrieved chunks."
    )

    with MongoDBClient(
        model=Document, collection_name=settings.MONGO_DB_LOAD_COLLECTION
    ) as client:
        parent_docs = client.fetch_documents({"id": {"$in": parent_ids}})

    return parent_docs
