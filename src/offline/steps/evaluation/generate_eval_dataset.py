from typing_extensions import Annotated
from zenml import step, get_step_context
from src.personal_ai_engine.domain.document import Document


@step
def generate_eval_queries(
    documents: list[Document],
    num_queries: int = 5,
) -> Annotated[list[dict], "eval_queries"]:
    """ZenML step to generate a list of evaluation queries (mocked based on documents)."""
    queries = []

    # Generate simple queries from first few documents
    for idx, doc in enumerate(documents[:num_queries]):
        title = doc.metadata.title
        snippet = doc.content[:100]
        queries.append(
            {
                "query": f"What is discussed in the document titled '{title}'?",
                "ground_truth": snippet,
            }
        )

    # If not enough documents, add standard test queries
    while len(queries) < num_queries:
        queries.append(
            {
                "query": "What is the capital of France?",
                "ground_truth": "Paris is the capital of France.",
            }
        )

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="eval_queries",
        metadata={
            "queries_count": len(queries),
        },
    )

    return queries
