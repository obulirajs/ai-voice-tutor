"""Visual directive extraction — the Visual Companion's render-on-directive
contract (architecture.md: it's "a dumb renderer... driven entirely by
directives from Orchestration"). extract_directives() post-processes an
LLM reply into a list of VisualDirective objects the companion can render
prominently, without altering the reply text itself: the typed-chat path
still renders the reply as-is (KaTeX handles inline LaTeX there), and the
voice path's spoken-form preprocessor (app.voice.spoken_form) separately
turns notation into speech. Kept deliberately simple for v1 -- regex/line
scanning over the patterns prompt_builder.py's system prompt now nudges the
model toward (LaTeX math, bold key terms, [Passage N — page X] citations,
GitHub-style markdown tables for comparisons), not a full markdown/LaTeX
parser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.config import ORCHESTRATION_DEFAULTS

__all__ = ["VisualDirective", "extract_directives"]

_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# A dollar sign not immediately followed by a digit opens inline math --
# excludes amounts like "$5.00" or "$100" (which always start with a digit)
# without needing to special-case currency formatting.
_INLINE_MATH = re.compile(r"\$(?!\d)([^$\n]+?)\$")
_CITATION = re.compile(r"\[\s*Passage\s+(\d+)\s*[-—]\s*page\s+(\S+?)\s*\]", re.IGNORECASE)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
# Teacher mode's supplemented-example marker (prompt_builder.py's
# TEACHING_MODES["teacher"]) -- flags a real-world example the model added
# from general knowledge rather than the retrieved textbook passages.
_SUPPLEMENTED = re.compile(r"\*\*💡\s*Beyond the textbook:\*\*")

# GitHub-style markdown table detection (prompt_builder.py's formatting
# instructions nudge the model toward "header | separator | rows" for
# comparisons). A separator cell is one or more dashes with optional
# leading/trailing colons (alignment markers) -- nothing else, which is
# what distinguishes a real separator row from an ordinary text row and
# keeps this from firing on prose that merely contains a "|".
_TABLE_SEPARATOR_CELL = re.compile(r"^:?-+:?$")
_HEADING_LINE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_FULL_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*[:：]?\s*$")
_CELL_BOLD = re.compile(r"\*\*(.+?)\*\*")


@dataclass(frozen=True)
class VisualDirective:
    directive_type: str  # "formula", "highlight", "image_ref", "text_block", "supplemented", "table"
    content: str
    label: str | None = None


def _split_table_row(line: str) -> list[str]:
    """Splits a "| a | b |" (or "a | b", no outer pipes) row into trimmed
    cells, with any **bold** markers inside a cell stripped -- the table
    renders each cell as its own element, so bold syntax would otherwise
    show up as literal asterisks.
    """
    stripped = line.strip()
    stripped = stripped.removeprefix("|")
    stripped = stripped.removesuffix("|")
    return [_CELL_BOLD.sub(r"\1", cell).strip() for cell in stripped.split("|")]


def _is_separator_row(line: str) -> bool:
    cells = line.strip()
    cells = cells.removeprefix("|")
    cells = cells.removesuffix("|")
    parts = [cell.strip() for cell in cells.split("|")]
    return len(parts) > 0 and all(_TABLE_SEPARATOR_CELL.match(part) for part in parts)


def _preceding_label(lines: list[str], header_index: int) -> str | None:
    """The nearest non-blank line above the table, if it's a heading or an
    entirely-bold line (e.g. "**French Verb Conjugations**") -- anything
    else (ordinary prose) means the table has no label.
    """
    index = header_index - 1
    while index >= 0 and lines[index].strip() == "":
        index -= 1
    if index < 0:
        return None
    line = lines[index].strip()
    heading_match = _HEADING_LINE.match(line)
    if heading_match:
        return heading_match.group(1).strip()
    bold_match = _FULL_BOLD_LINE.match(line)
    if bold_match:
        return bold_match.group(1).strip().rstrip(":：").strip()
    return None


def _extract_tables(reply: str) -> list[VisualDirective]:
    lines = reply.splitlines()
    directives: list[VisualDirective] = []
    index = 0
    total = len(lines)

    while index < total - 1 and len(directives) < ORCHESTRATION_DEFAULTS.max_table_directives:
        header_line = lines[index]
        if "|" not in header_line or not header_line.strip() or not _is_separator_row(lines[index + 1]):
            index += 1
            continue

        header_cells = _split_table_row(header_line)
        separator_cells = _split_table_row(lines[index + 1])
        if not header_cells or len(separator_cells) != len(header_cells):
            # A malformed separator (column-count mismatch, or no separator
            # at all -- caught by _is_separator_row above) means this isn't
            # a real table; skip it rather than emitting a garbled directive.
            index += 1
            continue

        rows: list[list[str]] = []
        row_index = index + 2
        while row_index < total and "|" in lines[row_index] and lines[row_index].strip():
            row_cells = _split_table_row(lines[row_index])
            if len(row_cells) < len(header_cells):
                row_cells = row_cells + [""] * (len(header_cells) - len(row_cells))
            elif len(row_cells) > len(header_cells):
                row_cells = row_cells[: len(header_cells)]
            rows.append(row_cells)
            row_index += 1

        if not rows:
            index += 1
            continue

        directives.append(
            VisualDirective(
                directive_type="table",
                content=json.dumps({"headers": header_cells, "rows": rows}),
                label=_preceding_label(lines, index),
            )
        )
        index = row_index

    return directives


def extract_directives(reply: str) -> tuple[str, list[VisualDirective]]:
    """Extract formula/citation/key-term directives from `reply`.

    Returns (reply, directives): the reply text is returned unchanged --
    nothing needs stripping for v1, since the typed-chat path wants the
    LaTeX/markdown left in place and the voice path's own spoken-form step
    handles notation separately. The signature stays (text, directives) so
    a future pass that does need to rewrite the text can do so without
    changing every caller.
    """
    directives: list[VisualDirective] = []

    for match in _DISPLAY_MATH.finditer(reply):
        directives.append(VisualDirective(directive_type="formula", content=match.group(1).strip()))

    # Inline math is scanned over the reply with display-math spans blanked
    # out first, so a $$...$$ block can't also get picked up (or throw off
    # delimiter pairing) for the inline pattern.
    without_display_math = _DISPLAY_MATH.sub(" ", reply)
    for match in _INLINE_MATH.finditer(without_display_math):
        directives.append(VisualDirective(directive_type="formula", content=match.group(1).strip()))

    for match in _CITATION.finditer(reply):
        directives.append(
            VisualDirective(directive_type="text_block", content=match.group(0), label=f"page {match.group(2)}")
        )

    bold_matches = list(_BOLD.finditer(reply))[: ORCHESTRATION_DEFAULTS.max_highlight_directives]
    for match in bold_matches:
        directives.append(VisualDirective(directive_type="highlight", content=match.group(1).strip()))

    for _ in _SUPPLEMENTED.finditer(reply):
        directives.append(
            VisualDirective(
                directive_type="supplemented",
                content="Example from outside the textbook",
                label="model knowledge",
            )
        )

    directives.extend(_extract_tables(reply))

    return reply, directives
