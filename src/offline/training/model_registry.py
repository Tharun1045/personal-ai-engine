from typing import Optional
from loguru import logger
from src.shared.config import settings


class HuggingFaceModelRegistry:
    def __init__(self, token: Optional[str] = None, username: Optional[str] = None):
        self.token = token or settings.HF_TOKEN
        self.username = username or settings.HF_USERNAME

    def push_model(self, model_id: str, local_dir: str, private: bool = True) -> bool:
        """Push a model folder to HuggingFace Hub.

        If HF credentials are not configured, log a warning and skip.
        """
        if not self.token or not self.username:
            logger.warning(
                "Hugging Face credentials (HF_TOKEN, HF_USERNAME) are not configured. "
                "Skipping real model registry push. (Using mock mode)"
            )
            return False

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            repo_id = f"{self.username}/{model_id}"

            logger.info(
                f"Pushing model from local directory {local_dir} to HuggingFace Hub repository: {repo_id}"
            )

            api.create_repo(
                repo_id=repo_id, token=self.token, private=private, exist_ok=True
            )
            api.upload_folder(folder_path=local_dir, repo_id=repo_id, token=self.token)

            logger.info(f"Model successfully pushed to HuggingFace Hub: {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to push model to Hugging Face: {e}")
            raise e
