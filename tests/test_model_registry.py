import sys
from unittest.mock import MagicMock

# Mock datasets and huggingface_hub modules before imports
mock_datasets = MagicMock()
mock_huggingface = MagicMock()
sys.modules["datasets"] = mock_datasets
sys.modules["huggingface_hub"] = mock_huggingface

from src.offline.training.model_registry import HuggingFaceModelRegistry  # noqa: E402
from src.offline.training.dataset_registry import HuggingFaceDatasetRegistry  # noqa: E402


def test_registry_unconfigured():
    # If not configured, should return False (mock mode) and not raise
    m_registry = HuggingFaceModelRegistry(token=None, username=None)
    res = m_registry.push_model("test_model", "dummy_path")
    assert res is False

    d_registry = HuggingFaceDatasetRegistry(token=None, username=None)
    res2 = d_registry.push_dataset("test_dataset", [])
    assert res2 is False


def test_dataset_registry_push_success():
    # Setup mock behavior
    mock_dataset_class = MagicMock()
    mock_datasets.Dataset = mock_dataset_class
    mock_dataset_instance = MagicMock()
    mock_dataset_class.from_list.return_value = mock_dataset_instance

    mock_hf_api_class = MagicMock()
    mock_huggingface.HfApi = mock_hf_api_class
    mock_api_instance = MagicMock()
    mock_hf_api_class.return_value = mock_api_instance

    d_registry = HuggingFaceDatasetRegistry(token="hf_tok", username="hf_user")
    res = d_registry.push_dataset("test_dataset", [{"instruction": "q", "output": "a"}])

    assert res is True
    mock_dataset_class.from_list.assert_called_once_with(
        [{"instruction": "q", "output": "a"}]
    )
    mock_dataset_instance.push_to_hub.assert_called_once_with(
        "hf_user/test_dataset", token="hf_tok", private=True
    )
