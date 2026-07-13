from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from src.offline.etl.scoring import (
    HeuristicQualityAgent,
    QualityScoreAgent,
)
from src.personal_ai_engine.domain import Document


@step
def add_quality_score(
    documents: list[Document],
    use_llm: bool = False,
    model_id: str = "qwen3:8b",
    mock: bool = False,
    max_workers: int = 10,
) -> Annotated[list[Document], "scored_documents"]:
    """Adds quality scores to documents using heuristic and model-based scoring agents."""
    heuristic_quality_agent = HeuristicQualityAgent()
    scored_documents: list[Document] = heuristic_quality_agent(documents)

    if use_llm:
        # Evaluate documents that were not confidently rejected by heuristics
        documents_to_eval = [
            d
            for d in scored_documents
            if d.quality_assessment and d.quality_assessment.get("total_score", 0) > 0.0
        ]
        documents_rejected = [
            d
            for d in scored_documents
            if d.quality_assessment
            and d.quality_assessment.get("total_score", 0) == 0.0
        ]

        quality_agent = QualityScoreAgent(
            mock=mock, max_concurrent_requests=max_workers
        )
        scored_documents_with_agents: list[Document] = quality_agent(documents_to_eval)

        scored_documents = documents_rejected + scored_documents_with_agents

    len_documents = len(documents)
    len_documents_with_scores = len(
        [doc for doc in scored_documents if doc.content_quality_score is not None]
    )
    logger.info(f"Total documents: {len_documents}")
    logger.info(f"Total documents that were scored: {len_documents_with_scores}")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="scored_documents",
        metadata={
            "len_documents": len_documents,
            "len_documents_with_scores": len_documents_with_scores,
        },
    )

    return scored_documents
