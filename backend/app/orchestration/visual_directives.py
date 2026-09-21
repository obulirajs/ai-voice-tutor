"""Visual directive extraction — the Visual Companion's render-on-directive
contract (architecture.md: it's "a dumb renderer... driven entirely by
directives from Orchestration"). extract_directives() post-processes an
LLM reply into a list of VisualDirective objects the companion can render
prominently, without altering the reply text itself: the typed-chat path
still renders the reply as-is (KaTeX handles inline LaTeX there), and the
voice path's spoken-form preprocessor (app.voice.spoken_form) separately
turns notation into speech. Kept deliberately simple for v1 -- regex over
the patterns prompt_builder.py's system prompt now nudges the model toward
(LaTeX math, bold key terms, [Passage N — page X] citations), not a full
markdown/LaTeX parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["VisualDirective", "extract_directives"]

# Bold terms are noisy in a long reply (emphasis, not just definitions) --
# cap how many become highlight directives.
_MAX_HIGHLIGHTS = 5

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


@dataclass(frozen=True)
class VisualDirective:
    directive_type: str  # "formula", "highlight", "image_ref", "text_block", "supplemented"
    content: str
    label: str | None = None


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

    bold_matches = list(_BOLD.finditer(reply))[:_MAX_HIGHLIGHTS]
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

    return reply, directives
