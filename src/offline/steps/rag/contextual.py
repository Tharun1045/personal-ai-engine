from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain.document import Document, DocumentChunk
from src.offline.etl.contextual_retrieval import ContextualRetriever


@step
def contextualize_chunks(
    documents: list[Document],
    chunks: list[DocumentChunk],
    use_context: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> Annotated[list[DocumentChunk], "contextualized_chunks"]:
    """ZenML step to optionally add document-level context to each chunk."""
    if not use_context:
        return chunks

    doc_map = {doc.id: doc.content for doc in documents}
    contextualizer = ContextualRetriever(provider=provider, model=model)

    for chunk in chunks:
        parent_content = doc_map.get(chunk.parent_id)
        if parent_content:
            chunk.content = contextualizer.contextualize(parent_content, chunk.content)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="contextualized_chunks",
        metadata={
            "chunks_count": len(chunks),
            "contextualized_count": len(chunks),
        },
    )

    return chunks
