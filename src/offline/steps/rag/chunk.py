from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain.document import Document, DocumentChunk
from src.offline.etl.chunking import chunk_document


@step
def chunk_documents(
    documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 50
) -> Annotated[list[DocumentChunk], "chunks"]:
    """ZenML step to split documents into chunks."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="chunks",
        metadata={
            "documents_count": len(documents),
            "chunks_count": len(all_chunks),
        },
    )

    return all_chunks
