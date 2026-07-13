from loguru import logger
from src.shared.config import settings
from src.offline.mongo_client import MongoDBClient


def create_search_indexes() -> None:
    """Create all required database indexes, including unique id constraints,
    keyword text search indexes, and vector search placeholders.
    """
    logger.info("Initializing search indexes in MongoDB...")

    # 1. Unique index on parent documents
    with MongoDBClient(collection_name=settings.MONGO_DB_LOAD_COLLECTION) as client:
        client.collection.create_index("id", unique=True)
        logger.info(
            f"Unique index created on 'id' for collection '{settings.MONGO_DB_LOAD_COLLECTION}'"
        )

    # 2. Text and unique indexes on chunks
    with MongoDBClient(collection_name=settings.MONGO_CHUNKS_COLLECTION) as client:
        client.collection.create_index("id", unique=True)
        client.collection.create_index("parent_id")
        client.create_text_index()
        logger.info(
            f"Unique, parent, and text indexes created for collection '{settings.MONGO_CHUNKS_COLLECTION}'"
        )

        # Vector search index creation is only supported via Atlas API or UI for MongoDB Atlas.
        # For local cosine similarity, no index needs to be created on MongoDB side.
        backend = settings.VECTOR_SEARCH_BACKEND
        if backend == "atlas":
            logger.info(
                f"Vector search backend is set to 'atlas'. Ensure that index '{settings.MONGO_VECTOR_INDEX_NAME}' "
                "is created in MongoDB Atlas with mapping: { 'fields': [ { 'type': 'vector', 'path': 'embedding', 'numDimensions': 1536, 'similarity': 'cosine' } ] }"
            )
        else:
            logger.info(
                "Vector search backend is set to 'local'. Local cosine similarity similarity will be computed in memory."
            )
