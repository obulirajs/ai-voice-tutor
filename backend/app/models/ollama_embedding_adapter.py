from __future__ import annotations

import ollama

from .embedding_base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embedding adapter backed by the Ollama Python client."""

    def __init__(self, model: str, client: ollama.Client | None = None) -> None:
        self._model = model
        self._client = client or ollama.Client()

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(model=self._model, input=texts)
        return [list(vector) for vector in result["embeddings"]]
