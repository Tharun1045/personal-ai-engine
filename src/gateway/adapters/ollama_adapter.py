from typing import List
import ollama
from src.gateway.interfaces import ChatGenerator, TextEmbedder


class OllamaChatGenerator(ChatGenerator):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=self.base_url)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama chat generation failed: {e}")

    def generate_with_system(self, system: str, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama chat generation failed: {e}")


class OllamaTextEmbedder(TextEmbedder):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=self.base_url)

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # In Ollama we do sequential calls or batch if backend supports it. Sequential is safe.
        embeddings = []
        for t in texts:
            embeddings.append(self.embed(t))
        return embeddings
