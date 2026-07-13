from typing import Literal
from pydantic import BaseModel, Field
from .document import DocumentChunk


class SearchQuery(BaseModel):
    text: str
    strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    top_k: int = 5
    filters: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
