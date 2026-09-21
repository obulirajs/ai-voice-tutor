from __future__ import annotations

import time

import ollama

from .base import Usage
from .embedding_base import EmbeddingProvider, EmbeddingResponse

# A large ingestion batch (one call embedding every chunk of a whole
# textbook) runs long enough that Ollama's model runner has occasionally
# dropped out from under it in practice -- a "connection refused"
# ResponseError to what should be a local, already-loaded model. That's a
# transient local hiccup, not a real failure, but hitting it here throws
# away an entire ingestion run's worth of paid vision-transcription calls
# for a downstream step that is itself free and local. Worth a few retries
# before giving up for real.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0


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
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                result = self._client.embed(model=self._model, input=texts)
                return EmbeddingResponse(
                    embeddings=[list(vector) for vector in result["embeddings"]],
                    # Embeddings have no output tokens -- prompt_eval_count is the
                    # total input tokens processed for this whole batch call.
                    usage=Usage(input_tokens=result["prompt_eval_count"], output_tokens=None),
                )
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module note
                last_error = exc
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_DELAY_SECONDS)
        assert last_error is not None
        raise last_error
