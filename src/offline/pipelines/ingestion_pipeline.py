from pathlib import Path

from loguru import logger
from zenml import pipeline

from src.offline.steps.collect_notion_data import (
    extract_notion_documents,
    extract_notion_documents_metadata,
)

from src.offline.steps.infrastructure import save_documents_to_disk


@pipeline
def collect_notion_data(database_ids: list[str], data_dir: Path) -> None:
    notion_data_dir = data_dir / "notion"
    notion_data_dir.mkdir(parents=True, exist_ok=True)

    for index, database_id in enumerate(database_ids):
        logger.info(f"Collecting pages from database '{database_id}'")
        documents_metadata = extract_notion_documents_metadata(database_id=database_id)
        documents_data = extract_notion_documents(documents_metadata=documents_metadata)

        save_documents_to_disk(
            documents=documents_data,
            output_dir=notion_data_dir / f"database_{index}",
        )
