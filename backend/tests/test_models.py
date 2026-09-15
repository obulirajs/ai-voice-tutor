from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models import Message, get_provider
from app.models.anthropic_adapter import AnthropicProvider
from app.models.ollama_adapter import OllamaProvider


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
