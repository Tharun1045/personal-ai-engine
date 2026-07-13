from loguru import logger
from pydantic import BaseModel
from typing_extensions import Annotated
from zenml.steps import get_step_context, step

from src.offline.mongo_client import MongoDBClient


@step
def ingest_to_mongodb(
    models: list[BaseModel], collection_name: str, clear_collection: bool = False
) -> Annotated[int, "output"]:
    """ZenML step to ingest documents into MongoDB."""

    if not models:
        raise ValueError("No documents provided for ingestion")

    model_type = type(models[0])
    logger.info(
        f"Ingesting {len(models)} documents of type '{model_type.__name__}' into MongoDB collection '{collection_name}'"
    )
    with MongoDBClient(model=model_type, collection_name=collection_name) as service:
        if clear_collection:
            logger.warning(
                f"'clear_collection' is set to True. Clearing MongoDB collection '{collection_name}' before ingestion."
            )
            service.clear_collection()

        metrics = service.ingest_documents(models)

        count = service.get_collection_count()
        logger.info(
            f"Successfully ingested {count} documents into MongoDB collection '{collection_name}'. "
            f"Metrics: {metrics}"
        )

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="output",
        metadata={
            "count": count,
            "inserted": metrics["inserted"],
            "updated": metrics["updated"],
            "unchanged": metrics["unchanged"],
        },
    )

    return count
