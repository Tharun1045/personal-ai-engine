from typing import Generic, Type, TypeVar

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
        self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
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

    def ingest_documents(self, documents: list[T]) -> None:
        if not documents:
            return
        if not self.collection_name:
            raise ValueError("Collection name not set")

        for doc in documents:
            doc_dict = doc.model_dump()
            # Ensure idempotent writes by using 'id' to update or insert
            # We can use update_one with upsert=True instead of insert_many to handle duplicates
            self.collection.update_one(
                {"id": doc_dict.get("id")}, {"$set": doc_dict}, upsert=True
            )

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
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    doc[key] = str(value)

            _id = doc.pop("_id", None)
            if "id" not in doc and _id:
                doc["id"] = _id

            parsed_doc = self.model.model_validate(doc)
            parsed_documents.append(parsed_doc)

        return parsed_documents

    def get_collection_count(self) -> int:
        if not self.collection_name:
            raise ValueError("Collection name not set")
        return self.collection.count_documents({})

    def close(self) -> None:
        self.client.close()
