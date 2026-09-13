from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.main import app
from app.models import Message, ModelProvider, ModelResponse, Usage, get_provider


class FakeProvider(ModelProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.received: list[Message] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.received = messages
        return ModelResponse(content=self._reply, usage=Usage(input_tokens=3, output_tokens=5))


def test_chat_returns_provider_reply() -> None:
    fake = FakeProvider(reply="Bonjour ! Comment puis-je t'aider ?")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/chat", json={"message": "Bonjour"})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 200
    assert response.json() == {"reply": "Bonjour ! Comment puis-je t'aider ?"}
    assert fake.received == [{"role": "user", "content": "Bonjour"}]


def test_chat_requires_message_field() -> None:
    app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
    client = TestClient(app)

    try:
        response = client.post("/chat", json={})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 422
