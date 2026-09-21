"""Command router — rule-based intent classifier ahead of tutoring content.

architecture.md: "Parses incoming text for control commands ... vs. tutoring
content, before treating input as a question." Deliberately simple
(regex/keyword matching, no LLM classifier) per technical-design.md's AI
design patterns note: "start rule-based; only upgrade ... if rules prove too
brittle in practice." Patterns are anchored to whole-utterance phrasings so
an ambiguous question ("What is French?") is never misread as a command --
see route_command()'s docstring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["CommandResult", "route_command"]


@dataclass(frozen=True)
class CommandResult:
    command_type: str
    args: dict[str, str] = field(default_factory=dict)
    response_text: str = ""


def _pattern(*verbs: str) -> re.Pattern[str]:
    """One whole-utterance pattern matching any of the given verb phrases,
    tolerant of trailing punctuation ASR/typing might add."""
    alternation = "|".join(verbs)
    return re.compile(rf"^\s*(?:{alternation})\s*[.!?]*\s*$", re.IGNORECASE)


_SWITCH_SUBJECT_PATTERN = re.compile(
    r"^\s*(?:switch\s+to|change\s+subject\s+to|let'?s\s+study|open)\s+(?P<subject>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_LIST_SUBJECTS_PATTERN = _pattern(
    r"what\s+subjects?\s+do\s+i\s+have",
    r"list\s+(?:my\s+)?subjects?",
    r"show\s+(?:my\s+)?subjects?",
    r"what\s+can\s+i\s+study",
)

_HELP_PATTERN = _pattern(
    "help",
    r"what\s+can\s+you\s+do",
    r"what\s+commands?\s+are\s+there",
)

_HELP_RESPONSE = "You can ask me questions about your textbooks, or say 'switch to [subject]' to change subjects."


def route_command(text: str, available_subjects: list[str]) -> CommandResult | None:
    """Classify one piece of transcribed/typed text as a control command, or
    None if it's ordinary tutoring content.

    Patterns match the *whole* utterance (allowing only trailing
    punctuation), not a substring -- so a real question that happens to
    contain a command-ish word ("What is French?", "Can you help me with
    verb conjugation?") is never misclassified. Only clear, isolated command
    phrasings match; anything else falls through as a question, per
    technical-design.md's "don't try to be too clever" guidance.
    """
    stripped = text.strip()
    if not stripped:
        return None

    switch_match = _SWITCH_SUBJECT_PATTERN.match(stripped)
    if switch_match:
        return _resolve_switch_subject(switch_match.group("subject"), available_subjects)

    if _LIST_SUBJECTS_PATTERN.match(stripped):
        return _list_subjects(available_subjects)

    if _HELP_PATTERN.match(stripped):
        return CommandResult(command_type="help", response_text=_HELP_RESPONSE)

    return None


def _resolve_switch_subject(requested: str, available_subjects: list[str]) -> CommandResult:
    requested_clean = requested.strip().rstrip(".!?").strip()
    requested_lower = requested_clean.lower()

    for subject in available_subjects:
        subject_lower = subject.lower()
        if subject_lower == requested_lower or subject_lower in requested_lower or requested_lower in subject_lower:
            return CommandResult(
                command_type="switch_subject",
                args={"subject": subject},
                response_text=f"Switching to {subject}.",
            )

    available_list = ", ".join(available_subjects) if available_subjects else "none yet"
    return CommandResult(
        command_type="switch_subject",
        args={"subject": requested_clean},
        response_text=(
            f"I don't have a subject called {requested_clean}. Available subjects are: {available_list}."
        ),
    )


def _list_subjects(available_subjects: list[str]) -> CommandResult:
    if not available_subjects:
        response_text = "You don't have any subjects yet."
    else:
        response_text = f"You have {', '.join(available_subjects)}."
    return CommandResult(command_type="list_subjects", response_text=response_text)
