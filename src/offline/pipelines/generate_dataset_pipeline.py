from zenml import pipeline
from src.offline.steps.infrastructure import fetch_from_mongodb
from src.offline.steps.dataset import generate_qa_dataset, push_dataset_to_hf


@pipeline
def generate_dataset_pipeline(
    load_collection_name: str,
    dataset_name: str,
    limit: int = 0,
    num_pairs_per_doc: int = 3,
    provider: str | None = None,
    model: str | None = None,
    private: bool = True,
) -> None:
    """ZenML pipeline to generate fine-tuning dataset and push to Hugging Face Hub."""
    # 1. Fetch documents from MongoDB
    documents = fetch_from_mongodb(collection_name=load_collection_name, limit=limit)

    # 2. Generate instruction-response pairs
    dataset = generate_qa_dataset(
        documents=documents,
        num_pairs_per_doc=num_pairs_per_doc,
        provider=provider,
        model=model,
    )

    # 3. Push to HuggingFace Hub
    push_dataset_to_hf(
        data=dataset,
        dataset_name=dataset_name,
        private=private,
    )
