"""On-demand diagram generation -- the Visual Companion's "Draw diagram"
button. Turns the tutor's already-grounded reply into a Mermaid flowchart
using a small model (see app.config.DIAGRAM_DEFAULTS and
app.models.get_diagram_model_provider()), kept deliberately separate from
the main tutoring Model Provider call: this is a post-hoc rendering step,
not part of the grounded chat loop, and reuses the existing ModelProvider
adapters rather than a new interface (technical-design.md's dependency-
inversion convention, same pattern as everything else in app.models).

Two halves:
  - build_diagram_prompt(): the short system prompt a 3B model can reliably
    follow, plus one worked example.
  - sanitize_mermaid(): defends against the small model's output not being
    exactly what was asked for -- strips fences/prose, drops disallowed
    directives, enforces the node cap, and quotes labels containing
    characters Mermaid's bracket syntax chokes on.
generate_diagram() ties them together with the timeout and usage/error
bookkeeping the API endpoint needs.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from app.config import DIAGRAM_DEFAULTS
from app.models import Message, ModelProvider, Usage

__all__ = ["DiagramResult", "build_diagram_prompt", "generate_diagram", "sanitize_mermaid"]

_SYSTEM_PROMPT = f"""You turn a tutor's answer into a simple diagram for a student.
Output ONLY Mermaid code. No prose, no code fences, no explanation.
The first line must be exactly `flowchart TD` or `flowchart LR`.
Use at most {DIAGRAM_DEFAULTS.max_nodes} nodes. Each node label must be \
{DIAGRAM_DEFAULTS.max_label_words} words or fewer.
Node IDs are single letters: A, B, C, D, ... Labels go in square brackets, e.g. A[Evaporation].
Arrows are `-->` or `-->|label|`. Do NOT use styling, classDef, linkStyle, subgraph, click, or HTML.
Use ONLY facts stated in the text below. If the text does not describe a process, sequence, \
cycle, or relationship, output exactly: NONE

Example:
Text: "Water evaporates from the ocean, forms clouds through condensation, falls as \
precipitation, then flows back to the ocean as runoff."
Output:
flowchart TD
A[Evaporation] --> B[Condensation]
B --> C[Precipitation]
C --> D[Runoff]
D --> A
"""


def build_diagram_prompt(reply_text: str, passages: str = "", previous_error: str | None = None) -> list[Message]:
    """Assembles the message list for the diagram model. `passages` is
    optional supporting textbook context (kept short -- see
    DIAGRAM_DEFAULTS.max_context_chars); `reply_text` alone is already
    grounded, so an empty `passages` is fine. `previous_error` is set on a
    retry after sanitize_mermaid()/mermaid.parse() rejected the first
    attempt -- appended to the system prompt so the model sees what to fix.
    """
    system = _SYSTEM_PROMPT
    if previous_error:
        system += (
            f"\nYour previous diagram failed to parse: {previous_error}. "
            "Fix it and output only corrected Mermaid.\n"
        )

    user_content = reply_text.strip()
    if passages.strip():
        user_content += f"\n\nSupporting textbook context:\n{passages.strip()}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*$")
_FLOWCHART_HEADER = re.compile(r"^flowchart\s+(TD|LR|TB|RL)\s*$", re.IGNORECASE)
_FORBIDDEN_LINE = re.compile(r"\b(click|style|classDef|linkStyle)\b|%%\{|<script|https?://", re.IGNORECASE)

# Node references: an ID immediately followed by a label bracket (a
# definition, e.g. "A[Evaporation]") or an ID adjacent to an arrow (a bare
# reference, e.g. "A --> B"). Together these cover every way a node can
# appear in the simple subset of Mermaid the prompt asks for.
_NODE_WITH_LABEL = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?=[\[\(\{])")
_ARROW_ENDPOINTS = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\b(?:\s*(?:\[[^\]\n]*\]|\([^)\n]*\)|\{[^}\n]*\}))?"
    r"\s*-{2,3}>?\s*(?:\|[^|\n]*\|)?\s*"
    r"\b([A-Za-z][A-Za-z0-9_]*)\b"
)

# Wraps a bracket label containing a quote or parenthesis in quotes (escaping
# any inner quotes) -- Mermaid's plain "A[label]" syntax breaks on those
# characters otherwise. Left alone if the label is already quoted.
_LABEL_NEEDS_QUOTING = re.compile(r"\[(?!\s*\")([^\]\n]*[()\"][^\]\n]*)\]")


def _quote_label(match: re.Match[str]) -> str:
    escaped = match.group(1).replace('"', '\\"')
    return f'["{escaped}"]'


def sanitize_mermaid(raw: str) -> str | None:
    """Cleans up the diagram model's raw output, or returns None if it
    can't be turned into a safe diagram (the model said "NONE", nothing
    resembling a flowchart was found, or it exceeds the node cap).
    """
    if raw is None:
        return None

    text = raw.strip()
    if text == "NONE":
        return None

    lines = text.splitlines()

    header_index: int | None = None
    for index, line in enumerate(lines):
        candidate = line.strip()
        if not candidate or _CODE_FENCE.match(candidate):
            continue
        if _FLOWCHART_HEADER.match(candidate):
            header_index = index
            break
        # Any other non-blank, non-fence line before a header means this is
        # prose ("Sure, here's a diagram:") -- keep scanning past it, the
        # header search above already skips over such lines naturally.
        continue

    if header_index is None:
        return None

    body_lines: list[str] = []
    for line in lines[header_index:]:
        stripped = line.strip()
        if not stripped or _CODE_FENCE.match(stripped):
            continue
        if _FORBIDDEN_LINE.search(stripped):
            continue
        body_lines.append(stripped)

    if len(body_lines) < 2:  # header only, nothing to draw
        return None

    header = body_lines[0]
    content_lines = body_lines[1:]

    node_ids: set[str] = set()
    sanitized_content: list[str] = []
    for line in content_lines:
        for match in _ARROW_ENDPOINTS.finditer(line):
            node_ids.add(match.group(1))
            node_ids.add(match.group(2))
        for match in _NODE_WITH_LABEL.finditer(line):
            node_ids.add(match.group(1))
        sanitized_content.append(_LABEL_NEEDS_QUOTING.sub(_quote_label, line))

    if not node_ids or len(node_ids) > DIAGRAM_DEFAULTS.max_nodes:
        return None

    return "\n".join([header, *sanitized_content])


@dataclass(frozen=True)
class DiagramResult:
    mermaid: str | None
    reason: str | None = None  # "not_applicable" | "invalid" | "timeout" | "disabled"
    usage: Usage = field(default_factory=Usage)
    provider_name: str | None = None
    model_name: str | None = None
    latency_ms: float | None = None


async def generate_diagram(
    reply_text: str,
    passages: str,
    provider: ModelProvider | None,
    previous_error: str | None = None,
) -> DiagramResult:
    """Runs one diagram-model call (with a timeout) and sanitizes its output.

    Returns reason="disabled" without calling anything when `provider` is
    None (DIAGRAM_PROVIDER=off). A timeout or an unusable response never
    raises -- both come back as a DiagramResult the caller can log/respond
    with directly.
    """
    if provider is None:
        return DiagramResult(mermaid=None, reason="disabled")

    messages = build_diagram_prompt(reply_text, passages, previous_error)

    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            provider.generate(messages, None, DIAGRAM_DEFAULTS.temperature),
            timeout=DIAGRAM_DEFAULTS.timeout_s,
        )
    except TimeoutError:
        return DiagramResult(
            mermaid=None,
            reason="timeout",
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    latency_ms = (time.perf_counter() - started) * 1000

    mermaid = sanitize_mermaid(response.content)
    reason = None if mermaid is not None else ("not_applicable" if response.content.strip() == "NONE" else "invalid")

    return DiagramResult(
        mermaid=mermaid,
        reason=reason,
        usage=response.usage,
        provider_name=provider.provider_name,
        model_name=provider.model_name,
        latency_ms=latency_ms,
    )
