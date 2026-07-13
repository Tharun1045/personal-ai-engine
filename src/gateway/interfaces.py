from abc import ABC, abstractmethod
from typing import List


class ChatGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class TextEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass
