import pytest
from src.offline.pipelines.etl_pipeline import etl
from src.offline.pipelines.ingestion_pipeline import collect_notion_data


@pytest.mark.integration
def test_etl_pipeline_definition():
    # Just asserting the pipeline builds properly.
    assert etl.name == "etl"
    assert collect_notion_data.name == "collect_notion_data"
