import requests
from typing import Any, List
from src.gateway.interfaces import ChatGenerator, TextEmbedder


class GeminiChatGenerator(ChatGenerator):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Extract content from Gemini response structure
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Gemini chat generation failed: {e}")

    def generate_with_system(self, system: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Gemini chat generation with system failed: {e}")


class GeminiTextEmbedder(TextEmbedder):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> List[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"content": {"parts": [{"text": text}]}}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]
        except Exception as e:
            raise RuntimeError(f"Gemini embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Gemini batch embed API or fallback to loop
        embeddings = []
        for t in texts:
            embeddings.append(self.embed(t))
        return embeddings
