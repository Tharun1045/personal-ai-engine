from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain import Document
from src.offline.etl.cleaning import clean_content


@step
def clean_documents(
    documents: list[Document],
) -> Annotated[list[Document], "cleaned_documents"]:
    cleaned = []

    # Track metrics
    empty_docs = 0
    too_small = 0
    too_large = 0

    for doc in documents:
        cleaned_text, content_hash = clean_content(doc.content)

        # Minimum/Maximum content checks
        if not cleaned_text:
            empty_docs += 1
            continue

        word_count = len(cleaned_text.split())
        if word_count < 10:
            too_small += 1
            continue

        if word_count > 100000:
            too_large += 1
            continue

        doc.content = cleaned_text
        doc.content_hash = content_hash
        cleaned.append(doc)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents",
        metadata={
            "initial_count": len(documents),
            "final_count": len(cleaned),
            "rejected_empty": empty_docs,
            "rejected_too_small": too_small,
            "rejected_too_large": too_large,
        },
    )
    return cleaned
