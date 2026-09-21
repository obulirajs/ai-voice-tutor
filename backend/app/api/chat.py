from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge import VectorStore, get_vector_store
from app.models import (
    EmbeddingProvider,
    ModelProvider,
    get_embedding_provider,
    get_provider,
)
from app.orchestration import DEFAULT_SUBJECT, handle_turn
from app.storage import get_db, get_subject

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    # Only used to create a new session (session_id is None) -- an existing
    # session already has a subject, so this is ignored when session_id is set.
    subject: str | None = None
    # See prompt_builder.TEACHING_MODES -- changes only how the model frames
    # its answer, never what gets retrieved or the grounding guardrail.
    mode: Literal["textbook", "teacher"] = "textbook"


class SourceResponse(BaseModel):
    page_number: int | None
    score: float


class VisualDirectiveResponse(BaseModel):
    directive_type: str
    content: str
    label: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: list[SourceResponse] = []
    visual_directives: list[VisualDirectiveResponse] = []


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: ModelProvider = Depends(get_provider),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> ChatResponse:
    try:
        result = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=request.session_id,
            user_message=request.message,
            # Only matters for creating a new session -- handle_turn ignores
            # it once session_id resolves to an existing one.
            subject=request.subject if request.subject is not None else DEFAULT_SUBJECT,
            mode=request.mode,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.reply,
        session_id=result.session_id,
        sources=[
            SourceResponse(page_number=s.page_number, score=s.score)
            for s in result.sources
        ],
        visual_directives=[
            VisualDirectiveResponse(directive_type=d.directive_type, content=d.content, label=d.label)
            for d in result.visual_directives
        ],
    )


@dataclass(frozen=True)
class _QACheck:
    question: str
    # Substrings (any one is enough) expected in a grounded reply, or None to
    # expect the grounding guardrail's refusal instead (an empty sources list).
    expected_substrings: list[str] | None


# Hardcoded for the French textbook (see development-plan.md Phase 7) -- a
# lightweight end-to-end sanity check, not a general per-subject eval suite.
_QA_CHECKS = [
    _QACheck(
        question="Quels sont les festivals mentionnés dans le livre?",
        expected_substrings=["interculturel", "Journées Interculturelles", "Nantes"],
    ),
    _QACheck(question="What is quantum physics?", expected_substrings=None),
]


class QACheckItemResponse(BaseModel):
    question: str
    passed: bool
    response_preview: str
    sources: list[SourceResponse]


class QACheckResponse(BaseModel):
    subject_id: int
    checks: list[QACheckItemResponse]
    all_passed: bool


def _qa_check_passed(
    reply: str, sources: list[SourceResponse], expected_substrings: list[str] | None
) -> bool:
    if expected_substrings is None:
        return not sources  # the grounding guardrail refused, as expected
    reply_lower = reply.lower()
    return any(substring.lower() in reply_lower for substring in expected_substrings)


@router.get("/subjects/{subject_id}/qa-check", response_model=QACheckResponse)
async def qa_check(
    subject_id: int,
    provider: ModelProvider = Depends(get_provider),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> QACheckResponse:
    """Runs a small, hardcoded set of known-answer questions through the full
    grounded chat loop (retrieval + Model Provider, real calls -- this is not
    free) and reports pass/fail per question. Extends the retrieval-only
    consistency/golden-QA checks (app.knowledge.golden_qa,
    /subjects/{subject}/documents/{document_id}/sanity-check) one layer up:
    this confirms the *generated* reply, not just what got retrieved.

    A "pass" for a covered question means the reply contains at least one
    of its expected_substrings. A "pass" for the uncovered question means
    the grounding guardrail refused (no sources) -- see
    scripts/test_grounded_chat.py for eyeballing full reply quality.

    Calls handle_turn with temperature=0 -- an eval loop needs the same
    question against the same retrieved chunks to produce the same
    pass/fail every run, unlike normal chat, which keeps the provider's
    default temperature.
    """
    subject_row = await get_subject(db, subject_id)
    if subject_row is None:
        raise HTTPException(status_code=404, detail=f"No subject with id {subject_id}")

    checks: list[QACheckItemResponse] = []
    for qa_check_item in _QA_CHECKS:
        result = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message=qa_check_item.question,
            subject=subject_row.name,
            temperature=0.0,
        )
        sources = [
            SourceResponse(page_number=s.page_number, score=s.score)
            for s in result.sources
        ]
        checks.append(
            QACheckItemResponse(
                question=qa_check_item.question,
                passed=_qa_check_passed(
                    result.reply, sources, qa_check_item.expected_substrings
                ),
                response_preview=result.reply[:200],
                sources=sources,
            )
        )

    return QACheckResponse(
        subject_id=subject_id, checks=checks, all_passed=all(c.passed for c in checks)
    )
