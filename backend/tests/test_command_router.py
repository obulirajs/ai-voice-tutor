from __future__ import annotations

import pytest

from app.orchestration.command_router import CommandResult, route_command

_SUBJECTS = ["French", "Maths"]


# --- switch_subject ---


@pytest.mark.parametrize(
    "text",
    [
        "switch to French",
        "Switch to french",
        "switch to French.",
        "switch to French!",
        "change subject to French",
        "let's study French",
        "lets study French",
        "open French",
    ],
)
def test_switch_subject_recognizes_phrasings(text: str) -> None:
    result = route_command(text, _SUBJECTS)

    assert result is not None
    assert result.command_type == "switch_subject"
    assert result.args["subject"] == "French"
    assert result.response_text == "Switching to French."


def test_switch_subject_is_case_insensitive_on_subject_name() -> None:
    result = route_command("switch to FRENCH", _SUBJECTS)

    assert result is not None
    assert result.args["subject"] == "French"


def test_switch_subject_fuzzy_matches_partial_name() -> None:
    result = route_command("switch to math", ["Maths"])

    assert result is not None
    assert result.command_type == "switch_subject"
    assert result.args["subject"] == "Maths"


def test_switch_subject_unknown_subject_lists_available() -> None:
    result = route_command("switch to Physics", _SUBJECTS)

    assert result is not None
    assert result.command_type == "switch_subject"
    assert result.args["subject"] == "Physics"
    assert "Physics" in result.response_text
    assert "French" in result.response_text
    assert "Maths" in result.response_text


def test_switch_subject_unknown_subject_with_no_subjects_available() -> None:
    result = route_command("switch to Physics", [])

    assert result is not None
    assert "none yet" in result.response_text


# --- list_subjects ---


@pytest.mark.parametrize(
    "text",
    [
        "what subjects do I have",
        "What subjects do I have?",
        "list subjects",
        "list my subjects",
        "show subjects",
        "show my subjects",
        "what can I study",
    ],
)
def test_list_subjects_recognizes_phrasings(text: str) -> None:
    result = route_command(text, _SUBJECTS)

    assert result is not None
    assert result.command_type == "list_subjects"
    assert result.response_text == "You have French, Maths."


def test_list_subjects_with_none_available() -> None:
    result = route_command("list subjects", [])

    assert result is not None
    assert result.response_text == "You don't have any subjects yet."


# --- help ---


@pytest.mark.parametrize(
    "text",
    [
        "help",
        "Help!",
        "what can you do",
        "What can you do?",
        "what commands are there",
        "What commands are there?",
    ],
)
def test_help_recognizes_phrasings(text: str) -> None:
    result = route_command(text, _SUBJECTS)

    assert result is not None
    assert result.command_type == "help"
    assert "switch to" in result.response_text


# --- not a command ---


@pytest.mark.parametrize(
    "text",
    [
        "What is French?",
        "Can you help me with verb conjugation?",
        "What subjects are covered in chapter 3?",
        "Explain the passe compose.",
        "Comment allez-vous?",
        "",
        "   ",
        "I want to open a book and read about France.",
    ],
)
def test_ordinary_questions_are_not_commands(text: str) -> None:
    assert route_command(text, _SUBJECTS) is None


def test_command_result_is_a_frozen_dataclass_with_defaults() -> None:
    result = CommandResult(command_type="help")

    assert result.args == {}
    assert result.response_text == ""
