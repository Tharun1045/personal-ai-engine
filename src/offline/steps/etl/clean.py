import re
from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain import Document


def clean_content(content: str) -> str:
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content)
    return content.strip()


@step
def clean_documents(
    documents: list[Document],
) -> Annotated[list[Document], "cleaned_documents"]:
    cleaned = []
    for doc in documents:
        doc.content = clean_content(doc.content)
        cleaned.append(doc)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents",
        metadata={
            "count": len(cleaned),
        },
    )
    return cleaned
