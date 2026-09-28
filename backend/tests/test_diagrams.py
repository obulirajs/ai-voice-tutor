from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import DiagramDefaults
from app.models import Message, ModelProvider, ModelResponse, Usage
from app.orchestration import diagrams
from app.orchestration.diagrams import (
    build_diagram_prompt,
    generate_diagram,
    sanitize_mermaid,
)


class FakeProvider(ModelProvider):
    def __init__(self, reply: str, *, delay: float = 0.0) -> None:
        self._reply = reply
        self._delay = delay
        self.received: list[Message] = []
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "qwen2.5-coder:3b"

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        self.received = messages
        if self._delay:
            # A genuine await, not time.sleep() -- this simulates a real
            # async provider's network wait, which is what actually lets
            # asyncio.wait_for's timeout fire in generate_diagram().
            await asyncio.sleep(self._delay)
        return ModelResponse(content=self._reply, usage=Usage(input_tokens=50, output_tokens=10))


# --- sanitize_mermaid ---


def test_sanitize_strips_code_fences() -> None:
    raw = "```mermaid\nflowchart TD\nA[Start] --> B[End]\n```"

    assert sanitize_mermaid(raw) == "flowchart TD\nA[Start] --> B[End]"


def test_sanitize_strips_prose_before_the_header() -> None:
    raw = "Sure! Here's a diagram:\nflowchart TD\nA[Start] --> B[End]"

    assert sanitize_mermaid(raw) == "flowchart TD\nA[Start] --> B[End]"


def test_sanitize_none_response_returns_none() -> None:
    assert sanitize_mermaid("NONE") is None
    assert sanitize_mermaid("NONE\n") is None


def test_sanitize_rejects_diagrams_over_the_node_cap() -> None:
    letters = "ABCDEFGHIJKLM"  # 13 distinct nodes, one over the default cap of 12
    body = "\n".join(f"{letters[i]} --> {letters[i + 1]}" for i in range(len(letters) - 1))
    raw = f"flowchart TD\n{body}"

    assert sanitize_mermaid(raw) is None


def test_sanitize_allows_diagrams_at_the_node_cap() -> None:
    letters = "ABCDEFGHIJKL"  # exactly 12 nodes
    body = "\n".join(f"{letters[i]} --> {letters[i + 1]}" for i in range(len(letters) - 1))
    raw = f"flowchart TD\n{body}"

    result = sanitize_mermaid(raw)
    assert result is not None
    assert result.startswith("flowchart TD")


def test_sanitize_strips_click_and_style_lines() -> None:
    raw = (
        "flowchart TD\n"
        "A[Start] --> B[End]\n"
        'click A "http://example.com"\n'
        "style A fill:#f9f\n"
    )

    result = sanitize_mermaid(raw)
    assert result == "flowchart TD\nA[Start] --> B[End]"


def test_sanitize_quotes_a_label_containing_parentheses() -> None:
    raw = "flowchart TD\nA[Cellular respiration (aerobic)] --> B[ATP]"

    result = sanitize_mermaid(raw)
    assert result is not None
    assert 'A["Cellular respiration (aerobic)"]' in result


def test_sanitize_rejects_output_with_no_flowchart_header() -> None:
    assert sanitize_mermaid("This answer doesn't describe a process.") is None


def test_sanitize_preserves_french_accents() -> None:
    raw = "flowchart TD\nA[Évaporation] --> B[Précipitation]"

    result = sanitize_mermaid(raw)
    assert result is not None
    assert "Évaporation" in result
    assert "Précipitation" in result


# --- build_diagram_prompt ---


def test_build_diagram_prompt_includes_node_and_label_limits() -> None:
    messages = build_diagram_prompt("Water evaporates, then condenses.")

    system = messages[0]["content"]
    assert "flowchart TD" in system
    assert str(diagrams.DIAGRAM_DEFAULTS.max_nodes) in system
    assert str(diagrams.DIAGRAM_DEFAULTS.max_label_words) in system
    assert messages[1] == {"role": "user", "content": "Water evaporates, then condenses."}


def test_build_diagram_prompt_appends_previous_error() -> None:
    messages = build_diagram_prompt("Water evaporates.", previous_error="Parse error at line 2")

    assert "Parse error at line 2" in messages[0]["content"]


def test_build_diagram_prompt_includes_passages_in_user_message() -> None:
    messages = build_diagram_prompt("Water evaporates.", passages="Chapter 3: the water cycle.")

    assert "Chapter 3: the water cycle." in messages[1]["content"]


# --- generate_diagram ---


def test_generate_diagram_returns_disabled_without_calling_the_provider() -> None:
    result = asyncio.run(generate_diagram("Some reply.", "", None))

    assert result.mermaid is None
    assert result.reason == "disabled"


def test_generate_diagram_returns_sanitized_mermaid() -> None:
    provider = FakeProvider("flowchart TD\nA[Start] --> B[End]")

    result = asyncio.run(generate_diagram("The process starts, then ends.", "", provider))

    assert result.mermaid == "flowchart TD\nA[Start] --> B[End]"
    assert result.reason is None
    assert result.usage.input_tokens == 50
    assert provider.call_count == 1


def test_generate_diagram_reports_not_applicable_for_none() -> None:
    provider = FakeProvider("NONE")

    result = asyncio.run(generate_diagram("Vocabulary: le chat means cat.", "", provider))

    assert result.mermaid is None
    assert result.reason == "not_applicable"


def test_generate_diagram_reports_invalid_for_unparseable_output() -> None:
    provider = FakeProvider("I can't draw that.")

    result = asyncio.run(generate_diagram("Some reply.", "", provider))

    assert result.mermaid is None
    assert result.reason == "invalid"


def test_generate_diagram_passes_previous_error_through_to_the_prompt() -> None:
    provider = FakeProvider("flowchart TD\nA[Start] --> B[End]")

    asyncio.run(generate_diagram("Some reply.", "", provider, previous_error="Unexpected token"))

    assert "Unexpected token" in provider.received[0]["content"]


def test_generate_diagram_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagrams, "DIAGRAM_DEFAULTS", DiagramDefaults(timeout_s=0.05))
    provider = FakeProvider("flowchart TD\nA[Start] --> B[End]", delay=0.3)

    result = asyncio.run(generate_diagram("Some reply.", "", provider))

    assert result.mermaid is None
    assert result.reason == "timeout"
