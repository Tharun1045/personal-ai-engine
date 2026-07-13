import requests
from typing import Any, List
from src.gateway.interfaces import ChatGenerator, TextEmbedder


class OpenRouterChatGenerator(ChatGenerator):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is missing")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Tharun1045/personal-ai-engine",
            "X-Title": "Personal AI Engine",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter chat generation failed: {e}")

    def generate_with_system(self, system: str, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Tharun1045/personal-ai-engine",
            "X-Title": "Personal AI Engine",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter chat generation with system failed: {e}")


class OpenRouterTextEmbedder(TextEmbedder):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenRouter API key is missing")
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError(
            "OpenRouter does not support text embeddings natively."
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "OpenRouter does not support text embeddings natively."
        )
