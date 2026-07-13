from typing_extensions import Annotated
from zenml import step, get_step_context
from src.offline.training.dataset_registry import HuggingFaceDatasetRegistry


@step
def push_dataset_to_hf(
    data: list[dict],
    dataset_name: str,
    private: bool = True,
) -> Annotated[bool, "push_success"]:
    """ZenML step to push the generated dataset to the Hugging Face hub."""
    registry = HuggingFaceDatasetRegistry()
    success = registry.push_dataset(dataset_name, data, private=private)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="push_success",
        metadata={
            "success": success,
            "dataset_name": dataset_name,
            "private": private,
        },
    )

    return success
