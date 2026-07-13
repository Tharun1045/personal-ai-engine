import pytest
from src.shared.config import AppSettings
from src.offline.zenml_setup import minimal_smoke_pipeline, verify_zenml_server


def test_zenml_config_validation():
    settings = AppSettings(ZENML_STORE_URL="http://localhost:8080")
    assert settings.ZENML_STORE_URL == "http://localhost:8080"


@pytest.mark.integration
def test_zenml_smoke():
    # Verify server reachable
    is_reachable = verify_zenml_server()
    assert is_reachable is True, "ZenML server is not reachable"

    # Run minimal pipeline
    try:
        minimal_smoke_pipeline()
    except Exception as e:
        pytest.fail(f"ZenML pipeline failed to execute: {e}")
