from typing import List
from src.gateway.interfaces import ChatGenerator, TextEmbedder


class OpenRouterChatGenerator(ChatGenerator):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is missing")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        # Placeholder for actual OpenRouter API call
        raise NotImplementedError(
            "OpenRouter chat generation not yet fully implemented"
        )


class OpenRouterTextEmbedder(TextEmbedder):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is missing")
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> List[float]:
        # Placeholder for actual OpenRouter API call
        raise NotImplementedError("OpenRouter text embedding not yet fully implemented")
