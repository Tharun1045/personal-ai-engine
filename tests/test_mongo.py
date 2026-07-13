import pytest
from src.shared.config import AppSettings
from src.offline.mongo_client import MongoDBClient


def test_mongo_config_validation():
    settings = AppSettings(MONGO_URI="mongodb://localhost:27017", MONGO_DB_NAME="test")
    assert settings.MONGO_URI == "mongodb://localhost:27017"
    assert settings.MONGO_DB_NAME == "test"


def test_local_cosine_similarity():
    client = MongoDBClient()
    sim = client._cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 1e-6
    sim2 = client._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim2 - 0.0) < 1e-6

    # Orthogonal vectors
    sim3 = client._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim3 - 0.0) < 1e-6


@pytest.mark.integration
def test_mongo_smoke():
    # Will fail if docker is not running or mongodb is unavailable
    client = MongoDBClient()
    assert client.ping() is True

    # temporary CRUD
    collection = client.db["test_collection"]
    result = collection.insert_one({"test": "value"})
    assert result.inserted_id is not None

    doc = collection.find_one({"_id": result.inserted_id})
    assert doc is not None
    assert doc["test"] == "value"

    # Cleanup
    collection.delete_one({"_id": result.inserted_id})
