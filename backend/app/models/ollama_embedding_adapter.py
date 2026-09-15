from __future__ import annotations

import ollama

from .base import Usage
from .embedding_base import EmbeddingProvider, EmbeddingResponse


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

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        result = self._client.embed(model=self._model, input=texts)
        return EmbeddingResponse(
            embeddings=[list(vector) for vector in result["embeddings"]],
            # Embeddings have no output tokens -- prompt_eval_count is the
            # total input tokens processed for this whole batch call.
            usage=Usage(input_tokens=result["prompt_eval_count"], output_tokens=None),
        )
