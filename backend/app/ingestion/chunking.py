from __future__ import annotations

import re

_DEFAULT_MAX_CHARS = 900


def chunk_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """Paragraph-aware chunking.

    Splits on blank lines, then greedily packs paragraphs into chunks up to
    max_chars, so a paragraph — and any formula/diagram transcription inline
    within it — stays attached to its surrounding explanation rather than
    being split by an arbitrary fixed-size cut.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks
