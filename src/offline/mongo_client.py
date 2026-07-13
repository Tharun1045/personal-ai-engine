import math
from typing import Generic, Type, TypeVar, List, Tuple, Dict, Any, Optional
from bson import ObjectId
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.database import Database
from src.shared.config import settings

T = TypeVar("T", bound=BaseModel)


class MongoDBClient(Generic[T]):
    def __init__(
        self, model: Type[T] | None = None, collection_name: str | None = None
    ):
        self.model = model
        self.collection_name = collection_name
        self.client: MongoClient = MongoClient(
            settings.MONGO_URI, serverSelectionTimeoutMS=2000
        )
        self.db: Database = self.client[settings.MONGO_DB_NAME]

        if collection_name:
            self.collection = self.db[collection_name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def clear_collection(self) -> None:
        if not self.collection_name:
            raise ValueError("Collection name not set")
        self.collection.delete_many({})

    def create_text_index(self) -> None:
        """Create a text index on the content field for keyword search."""
        if not self.collection_name:
            raise ValueError("Collection name not set")
        self.collection.create_index([("content", "text")])

    def ingest_documents(self, documents: list[T]) -> dict[str, int]:
        if not documents:
            return {"inserted": 0, "updated": 0, "unchanged": 0}
        if not self.collection_name:
            raise ValueError("Collection name not set")

        # Ensure unique index on id
        self.collection.create_index("id", unique=True)

        metrics = {"inserted": 0, "updated": 0, "unchanged": 0}

        for doc in documents:
            doc_dict = doc.model_dump()
            result = self.collection.update_one(
                {"id": doc_dict.get("id")}, {"$set": doc_dict}, upsert=True
            )
            if result.upserted_id:
                metrics["inserted"] += 1
            elif result.modified_count > 0:
                metrics["updated"] += 1
            else:
                metrics["unchanged"] += 1

        return metrics

    def fetch_documents(self, query: dict, limit: int = 0) -> list[T]:
        if not self.collection_name or not self.model:
            raise ValueError("Collection name or model not set")

        cursor = self.collection.find(query)
        if limit > 0:
            cursor = cursor.limit(limit)

        documents = list(cursor)
        return self.__parse_documents(documents)

    def __parse_documents(self, documents: list[dict]) -> list[T]:
        if not self.model:
            return []

        parsed_documents = []
        for doc in documents:
            doc_dict = doc.copy()
            for key, value in doc_dict.items():
                if isinstance(value, ObjectId):
                    doc_dict[key] = str(value)

            _id = doc_dict.pop("_id", None)
            if "id" not in doc_dict and _id:
                doc_dict["id"] = _id

            parsed_doc = self.model.model_validate(doc_dict)
            parsed_documents.append(parsed_doc)

        return parsed_documents

    def get_collection_count(self) -> int:
        if not self.collection_name:
            raise ValueError("Collection name not set")
        return self.collection.count_documents({})

    def close(self) -> None:
        self.client.close()

    # Search methods for retrieval backend
    def text_search(
        self, query_text: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> List[Tuple[T, float]]:
        """Perform text/keyword search using the text index.

        Args:
            query_text: The search query string.
            top_k: The number of top results to return.
            filters: Optional filters.
        """
        if not self.collection_name or not self.model:
            raise ValueError("Collection name or model not set")

        # Combine text search query with optional filters
        query = {"$text": {"$search": query_text}}
        if filters:
            query.update(filters)

        cursor = (
            self.collection.find(query, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(top_k)
        )

        results = list(cursor)
        parsed = self.__parse_documents(results)

        return [
            (doc, float(raw.get("score", 1.0))) for doc, raw in zip(parsed, results)
        ]

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[Tuple[T, float]]:
        """Perform semantic vector search using either local cosine similarity or MongoDB Atlas Vector Search.

        Args:
            query_embedding: Query embedding vector.
            top_k: The number of top results to return.
            filters: Optional metadata filters.
        """
        if not self.collection_name or not self.model:
            raise ValueError("Collection name or model not set")

        backend = settings.VECTOR_SEARCH_BACKEND
        if backend == "local":
            return self._local_vector_search(query_embedding, top_k, filters)
        elif backend == "atlas":
            return self._atlas_vector_search(query_embedding, top_k, filters)
        else:
            raise ValueError(f"Unknown VECTOR_SEARCH_BACKEND: {backend}")

    def _local_vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[Tuple[T, float]]:
        """Local vector search fallback using cosine similarity in Python.

        NOTE: This is a development/testing implementation and is not equivalent in scalability to MongoDB Atlas.
        """
        query = filters or {}
        cursor = self.collection.find(query)

        scored_docs = []
        for raw in cursor:
            embedding = raw.get("embedding")
            if not embedding:
                continue

            similarity = self._cosine_similarity(query_embedding, embedding)
            scored_docs.append((raw, similarity))

        # Sort by similarity desc
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_docs[:top_k]

        parsed = self.__parse_documents([r[0] for r in top_results])
        return [(doc, score) for doc, (_, score) in zip(parsed, top_results)]

    def _atlas_vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[Tuple[T, float]]:
        """Atlas Vector Search implementation."""
        # Atlas aggregation pipeline
        vector_search_stage: Dict[str, Any] = {
            "index": settings.MONGO_VECTOR_INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": top_k * 10,
            "limit": top_k,
        }
        if filters:
            vector_search_stage["filter"] = filters

        pipeline = [
            {"$vectorSearch": vector_search_stage},
            {
                "$project": {
                    "_id": 1,
                    "id": 1,
                    "parent_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            cursor = self.collection.aggregate(pipeline)
            results = list(cursor)
            parsed = self.__parse_documents(results)
            return [
                (doc, float(raw.get("score", 1.0))) for doc, raw in zip(parsed, results)
            ]
        except Exception as e:
            raise RuntimeError(f"Atlas Vector Search aggregation failed: {e}")

    def _cosine_similarity(self, u: List[float], v: List[float]) -> float:
        if not u or not v:
            return 0.0
        dot_product = sum(a * b for a, b in zip(u, v))
        norm_u = math.sqrt(sum(a * a for a in u))
        norm_v = math.sqrt(sum(b * b for b in v))
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return dot_product / (norm_u * norm_v)

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[Tuple[T, float]]:
        """Perform hybrid search combining keyword and vector search via Reciprocal Rank Fusion (RRF)."""
        # Get semantic results
        semantic_results = self.vector_search(
            query_embedding, top_k=top_k * 2, filters=filters
        )
        # Get keyword results
        keyword_results = self.text_search(query_text, top_k=top_k * 2, filters=filters)

        # Apply Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1.0 / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, T] = {}

        for rank, (doc, _) in enumerate(semantic_results):
            # mypy generic/BaseModel has id attribute or doc can be assumed to have 'id'
            doc_id = getattr(doc, "id", None)
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)
                chunks_map[doc_id] = doc

        for rank, (doc, _) in enumerate(keyword_results):
            doc_id = getattr(doc, "id", None)
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)
                chunks_map[doc_id] = doc

        # Sort by RRF score desc
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        return [(chunks_map[doc_id], score) for doc_id, score in sorted_rrf]
