import pytest
from src.offline.pipelines.etl_pipeline import etl
from src.offline.pipelines.ingestion_pipeline import collect_notion_data
from src.offline.pipelines.rag_index_pipeline import compute_rag_index


@pytest.mark.integration
def test_etl_pipeline_definition():
    # Just asserting the pipelines build properly.
    assert etl.name == "etl"
    assert collect_notion_data.name == "collect_notion_data"
    assert compute_rag_index.name == "compute_rag_index"
