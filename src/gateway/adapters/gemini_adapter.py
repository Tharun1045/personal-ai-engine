from typing import List
from src.gateway.interfaces import ChatGenerator, TextEmbedder


class GeminiChatGenerator(ChatGenerator):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        # Placeholder for actual Gemini API call
        raise NotImplementedError("Gemini chat generation not yet fully implemented")


class GeminiTextEmbedder(TextEmbedder):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> List[float]:
        # Placeholder for actual Gemini API call
        raise NotImplementedError("Gemini text embedding not yet fully implemented")
