from typing_extensions import Annotated
from zenml.steps import get_step_context, step

from src.personal_ai_engine.domain import Document
from src.offline.mongo_client import MongoDBClient


@step
def fetch_from_mongodb(
    collection_name: str,
    limit: int,
) -> Annotated[list[dict], "documents"]:
    with MongoDBClient(model=Document, collection_name=collection_name) as service:
        documents = service.fetch_documents(query={}, limit=limit)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="documents",
        metadata={
            "count": len(documents),
        },
    )

    return documents
