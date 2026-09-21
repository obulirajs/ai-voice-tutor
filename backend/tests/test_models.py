from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models import Message, get_provider, get_vision_provider
from app.models.anthropic_adapter import AnthropicProvider
from app.models.anthropic_vision_adapter import AnthropicVisionProvider
from app.models.ollama_adapter import OllamaProvider
from app.models.ollama_embedding_adapter import OllamaEmbeddingProvider
from app.models.tesseract_vision_adapter import TesseractVisionProvider

# Mirrors tesseract_vision_adapter's own PATH/common-install-location lookup
# -- see test_tesseract_vision.py for the fuller version of this check.
_TESSERACT_AVAILABLE = bool(shutil.which("tesseract")) or Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file()
skip_without_tesseract = pytest.mark.skipif(not _TESSERACT_AVAILABLE, reason="Tesseract not installed")


def test_ollama_provider_generate_parses_response() -> None:
    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {"content": "Bonjour !", "tool_calls": []},
        "prompt_eval_count": 12,
        "eval_count": 4,
    }
    provider = OllamaProvider(model="llama3.2:1b", client=fake_client)
    messages: list[Message] = [{"role": "user", "content": "Dis bonjour"}]

    response = provider.generate(messages)

    fake_client.chat.assert_called_once_with(model="llama3.2:1b", messages=messages)
    assert response.content == "Bonjour !"
    assert response.tool_calls == []
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 4


def test_ollama_provider_exposes_name_and_model() -> None:
    provider = OllamaProvider(model="llama3.2:1b", client=MagicMock())

    assert provider.provider_name == "ollama"
    assert provider.model_name == "llama3.2:1b"


def test_anthropic_provider_exposes_name_and_model() -> None:
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=MagicMock())

    assert provider.provider_name == "anthropic"
    assert provider.model_name == "claude-sonnet-5"


def test_ollama_provider_passes_tools_through() -> None:
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "ok"}}
    provider = OllamaProvider(model="llama3.2:1b", client=fake_client)
    tools = [{"type": "function", "function": {"name": "switch_subject"}}]

    provider.generate([{"role": "user", "content": "hi"}], tools=tools)

    _, kwargs = fake_client.chat.call_args
    assert kwargs["tools"] == tools


def test_ollama_provider_passes_temperature_through_as_option() -> None:
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "ok"}}
    provider = OllamaProvider(model="llama3.2:1b", client=fake_client)

    provider.generate([{"role": "user", "content": "hi"}], temperature=0.0)

    _, kwargs = fake_client.chat.call_args
    assert kwargs["options"] == {"temperature": 0.0}


def test_ollama_provider_omits_options_when_temperature_not_given() -> None:
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "ok"}}
    provider = OllamaProvider(model="llama3.2:1b", client=fake_client)

    provider.generate([{"role": "user", "content": "hi"}])

    _, kwargs = fake_client.chat.call_args
    assert "options" not in kwargs


def test_anthropic_provider_generate_parses_text_and_usage() -> None:
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Bonjour !")],
        usage=SimpleNamespace(input_tokens=20, output_tokens=6),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=fake_client)

    response = provider.generate([{"role": "user", "content": "Dis bonjour"}])

    assert response.content == "Bonjour !"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 6


def test_anthropic_provider_splits_system_message() -> None:
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=fake_client)

    provider.generate(
        [
            {"role": "system", "content": "You are a French tutor."},
            {"role": "user", "content": "Bonjour"},
        ]
    )

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["system"] == "You are a French tutor."
    assert kwargs["messages"] == [{"role": "user", "content": "Bonjour"}]


def test_anthropic_provider_passes_temperature_through() -> None:
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=fake_client)

    provider.generate([{"role": "user", "content": "hi"}], temperature=0.0)

    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["temperature"] == 0.0


def test_anthropic_provider_omits_temperature_when_not_given() -> None:
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=fake_client)

    provider.generate([{"role": "user", "content": "hi"}])

    _, kwargs = fake_client.messages.create.call_args
    assert "temperature" not in kwargs


def test_anthropic_provider_parses_tool_use_blocks() -> None:
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="tool_1", name="switch_subject", input={"subject": "french"}),
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    provider = AnthropicProvider(model="claude-sonnet-5", api_key="test-key", client=fake_client)

    response = provider.generate([{"role": "user", "content": "switch to french"}])

    assert response.content == ""
    assert response.tool_calls == [{"id": "tool_1", "name": "switch_subject", "input": {"subject": "french"}}]


def test_get_provider_defaults_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = get_provider()

    assert isinstance(provider, AnthropicProvider)


def test_get_provider_returns_ollama_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")

    provider = get_provider()

    assert isinstance(provider, OllamaProvider)


def test_get_provider_requires_api_key_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_provider()


def test_get_provider_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "bogus")

    with pytest.raises(ValueError):
        get_provider()


@skip_without_tesseract
def test_get_vision_provider_defaults_to_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISION_PROVIDER", raising=False)
    if not shutil.which("tesseract"):
        monkeypatch.setenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    provider = get_vision_provider()

    assert isinstance(provider, TesseractVisionProvider)


def test_get_vision_provider_returns_anthropic_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = get_vision_provider()

    assert isinstance(provider, AnthropicVisionProvider)


def test_get_vision_provider_requires_api_key_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_vision_provider()


def test_get_vision_provider_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "bogus")

    with pytest.raises(ValueError):
        get_vision_provider()


def test_ollama_embedding_provider_parses_response() -> None:
    fake_client = MagicMock()
    fake_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]], "prompt_eval_count": 5}
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", client=fake_client)

    response = provider.embed(["hello"])

    fake_client.embed.assert_called_once_with(model="nomic-embed-text", input=["hello"])
    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens is None


def test_ollama_embedding_provider_retries_on_transient_failure() -> None:
    """Regression coverage for a real incident: a large ingestion batch hit
    a transient "connection refused" from Ollama's model runner partway
    through, which without a retry would throw away an entire paid
    vision-transcription pass for a downstream, free, local embedding step.
    """
    fake_client = MagicMock()
    fake_client.embed.side_effect = [
        ConnectionError("connection refused"),
        {"embeddings": [[0.1, 0.2]], "prompt_eval_count": 3},
    ]
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", client=fake_client)

    with patch("app.models.ollama_embedding_adapter.time.sleep") as fake_sleep:
        response = provider.embed(["hello"])

    assert response.embeddings == [[0.1, 0.2]]
    assert fake_client.embed.call_count == 2
    fake_sleep.assert_called_once()


def test_ollama_embedding_provider_raises_after_exhausting_retries() -> None:
    fake_client = MagicMock()
    fake_client.embed.side_effect = ConnectionError("connection refused")
    provider = OllamaEmbeddingProvider(model="nomic-embed-text", client=fake_client)

    with patch("app.models.ollama_embedding_adapter.time.sleep"), pytest.raises(ConnectionError):
        provider.embed(["hello"])

    assert fake_client.embed.call_count == 3
