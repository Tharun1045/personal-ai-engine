from zenml import pipeline
from src.offline.steps.infrastructure import fetch_from_mongodb, ingest_to_mongodb
from src.offline.steps.rag import chunk_documents, contextualize_chunks, embed_chunks


@pipeline
def compute_rag_index(
    load_collection_name: str,
    chunks_collection_name: str,
    limit: int = 0,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    use_context: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    batch_size: int = 16,
) -> None:
    """ZenML pipeline to load documents from MongoDB, chunk them, contextualize, embed, and save chunks back to MongoDB."""
    # 1. Fetch parent documents
    documents = fetch_from_mongodb(collection_name=load_collection_name, limit=limit)

    # 2. Chunk documents
    chunks = chunk_documents(
        documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # 3. Contextualize chunks (optional)
    context_chunks = contextualize_chunks(
        documents=documents,
        chunks=chunks,
        use_context=use_context,
        provider=llm_provider,
        model=llm_model,
    )

    # 4. Embed chunks
    embedded_chunks = embed_chunks(
        chunks=context_chunks,
        provider=embedding_provider,
        model=embedding_model,
        batch_size=batch_size,
    )

    # 5. Ingest chunks back to MongoDB chunks collection
    ingest_to_mongodb(
        models=embedded_chunks,
        collection_name=chunks_collection_name,
        clear_collection=False,
    )
