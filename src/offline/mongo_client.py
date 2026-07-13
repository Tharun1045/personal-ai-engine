from pymongo import MongoClient
from pymongo.database import Database
from src.shared.config import settings


class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
        self.db: Database = self.client[settings.MONGO_DB_NAME]

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False
