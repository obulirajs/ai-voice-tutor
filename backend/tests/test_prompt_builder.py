from __future__ import annotations

from app.knowledge.retrieval import RetrievedChunk
from app.orchestration.prompt_builder import build_grounded_prompt


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Photosynthesis converts light energy into chemical energy.",
        page_number=12,
        score=0.9,
        document_id=1,
    )


def test_teacher_mode_system_prompt_allows_supplemented_examples() -> None:
    messages = build_grounded_prompt("Explain photosynthesis", [_chunk()], "Science", [], mode="teacher")

    system_content = messages[0]["content"]
    assert "**💡 Beyond the textbook:**" in system_content


def test_textbook_mode_system_prompt_has_no_supplemented_example_instruction() -> None:
    messages = build_grounded_prompt("Explain photosynthesis", [_chunk()], "Science", [], mode="textbook")

    system_content = messages[0]["content"]
    assert "Beyond the textbook" not in system_content
