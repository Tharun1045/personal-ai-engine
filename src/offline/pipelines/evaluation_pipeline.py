from zenml import pipeline
from src.offline.steps.infrastructure import fetch_from_mongodb
from src.offline.steps.evaluation import (
    generate_eval_queries,
    evaluate_rag_pipeline_step,
)


@pipeline
def evaluation_pipeline(
    load_collection_name: str,
    num_queries: int = 5,
    strategy: str = "hybrid",
    mock: bool = True,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """ZenML pipeline to fetch documents, generate evaluation queries, run RAG, and compute quality metrics."""
    # 1. Fetch documents
    documents = fetch_from_mongodb(
        collection_name=load_collection_name, limit=num_queries
    )

    # 2. Generate evaluation queries
    queries = generate_eval_queries(documents=documents, num_queries=num_queries)

    # 3. Evaluate RAG system performance
    evaluate_rag_pipeline_step(
        queries=queries,
        strategy=strategy,
        mock=mock,
        provider=provider,
        model=model,
    )
