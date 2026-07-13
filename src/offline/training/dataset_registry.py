from typing import List, Dict, Optional
from loguru import logger
from src.shared.config import settings


class HuggingFaceDatasetRegistry:
    def __init__(self, token: Optional[str] = None, username: Optional[str] = None):
        self.token = token or settings.HF_TOKEN
        self.username = username or settings.HF_USERNAME

    def push_dataset(
        self, dataset_name: str, data: List[Dict[str, str]], private: bool = True
    ) -> bool:
        """Push a dataset list of dicts to HuggingFace Hub.

        If HF credentials are not configured, log a warning and skip.
        """
        if not self.token or not self.username:
            logger.warning(
                "Hugging Face credentials (HF_TOKEN, HF_USERNAME) are not configured. "
                "Skipping real dataset push. (Using mock mode)"
            )
            return False

        try:
            from datasets import Dataset  # type: ignore

            # Convert list of dicts to datasets Dataset
            hf_dataset = Dataset.from_list(data)
            repo_id = f"{self.username}/{dataset_name}"

            logger.info(f"Pushing dataset to HuggingFace Hub repository: {repo_id}")

            hf_dataset.push_to_hub(repo_id, token=self.token, private=private)

            logger.info(f"Dataset successfully pushed to HuggingFace Hub: {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to push dataset to Hugging Face: {e}")
            raise e
