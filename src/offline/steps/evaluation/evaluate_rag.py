from typing_extensions import Annotated
from loguru import logger
from zenml import step, get_step_context
from src.personal_ai_engine.domain.queries import SearchQuery
from src.personal_ai_engine.domain.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.online.retrieval.retriever import DocumentRetriever
from src.online.evaluation.evaluation import RAGEvaluator
from src.gateway.factory import AIGatewayFactory


@step
def evaluate_rag_pipeline_step(
    queries: list[dict],
    strategy: str = "hybrid",
    mock: bool = True,
    provider: str | None = None,
    model: str | None = None,
) -> Annotated[dict, "evaluation_results"]:
    """ZenML step to run evaluation on RAG pipeline using RAGEvaluator."""
    retriever = DocumentRetriever(strategy=strategy)
    evaluator = RAGEvaluator(provider=provider, model=model, mock=mock)

    total_faithfulness = 0.0
    total_relevance = 0.0
    total_context_precision = 0.0
    count = 0

    for item in queries:
        q_text = item["query"]
        try:
            # 1. Retrieve context
            results = retriever.retrieve(SearchQuery(text=q_text, top_k=3))
            context_blocks = [chunk.content for chunk, _ in results]
            context_str = "\n\n".join(context_blocks)

            # 2. Generate response
            chat_generator = AIGatewayFactory.get_chat_generator()
            system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)
            user_prompt = RAG_USER_PROMPT.format(question=q_text)
            answer = chat_generator.generate_with_system(
                system=system_prompt, prompt=user_prompt
            )

            # 3. Compute metrics
            f_score = evaluator.evaluate_faithfulness(context_str, answer)
            r_score = evaluator.evaluate_relevance(q_text, answer)
            cp_score = evaluator.evaluate_context_precision(q_text, context_str)

            total_faithfulness += f_score
            total_relevance += r_score
            total_context_precision += cp_score
            count += 1
        except Exception as e:
            logger.error(f"Failed to evaluate query '{q_text}': {e}")

    results = {
        "queries_count": count,
        "mean_faithfulness": total_faithfulness / max(count, 1),
        "mean_relevance": total_relevance / max(count, 1),
        "mean_context_precision": total_context_precision / max(count, 1),
    }

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="evaluation_results", metadata=results)

    return results
