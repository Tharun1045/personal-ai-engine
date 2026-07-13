from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain.document import DocumentChunk
from src.offline.etl.embedding import EmbeddingGenerator


@step
def embed_chunks(
    chunks: list[DocumentChunk],
    provider: str | None = None,
    model: str | None = None,
    batch_size: int = 16,
) -> Annotated[list[DocumentChunk], "embedded_chunks"]:
    """ZenML step to generate embeddings for chunks."""
    generator = EmbeddingGenerator(provider=provider, model=model)
    embedded = generator.embed_chunks(chunks, batch_size=batch_size)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_chunks",
        metadata={
            "chunks_count": len(chunks),
            "embedded_count": len([c for c in embedded if c.embedding is not None]),
        },
    )

    return embedded
