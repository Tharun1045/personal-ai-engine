import pytest
from src.shared.config import AppSettings
from src.offline.mongo_client import MongoDBClient


def test_mongo_config_validation():
    settings = AppSettings(MONGO_URI="mongodb://localhost:27017", MONGO_DB_NAME="test")
    assert settings.MONGO_URI == "mongodb://localhost:27017"
    assert settings.MONGO_DB_NAME == "test"


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
