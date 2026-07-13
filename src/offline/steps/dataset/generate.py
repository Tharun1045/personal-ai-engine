from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain.document import Document
from src.offline.training.dataset_generation import QADatasetGenerator


@step
def generate_qa_dataset(
    documents: list[Document],
    num_pairs_per_doc: int = 3,
    provider: str | None = None,
    model: str | None = None,
) -> Annotated[list[dict], "dataset"]:
    """ZenML step to generate QA pairs from a list of documents."""
    generator = QADatasetGenerator(provider=provider, model=model)

    all_pairs = []
    for doc in documents:
        pairs = generator.generate_pairs(doc, num_pairs=num_pairs_per_doc)
        all_pairs.extend(pairs)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="dataset",
        metadata={
            "documents_evaluated": len(documents),
            "generated_pairs_count": len(all_pairs),
        },
    )

    return all_pairs
