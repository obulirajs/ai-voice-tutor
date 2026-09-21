"""Orchestration module — the core harness.

Public surface: handle_turn(), the retrieval-grounded session-state loop
that loads-or-creates a session, retrieves chunks for the session's
subject, enforces the grounding guardrail (refuse if nothing's been
ingested, refuse if nothing retrieved clears RETRIEVAL_SCORE_THRESHOLD),
builds the grounded prompt (prompt_builder.build_grounded_prompt(), also
exported here), calls the Model Provider, extracts visual directives from
the reply (visual_directives.extract_directives(), also exported here --
architecture.md's Visual Companion is "driven entirely by directives from
Orchestration"), and logs usage_events for the retrieval, embedding, and
generation calls; and SourceCitation/TurnResult, its result types. Also
handle_voice_turn(), the voice-loop wrapper that
checks for a control command (command_router.route_command()) before
falling through to handle_turn() -- see architecture.md's "control commands
vs. tutoring content" split. Depends on app.models', app.knowledge's,
app.ingestion's, and app.storage's interfaces only, never a concrete
adapter — see technical-design.md's dependency-inversion convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion import collection_for_subject
from app.knowledge import (
    RetrievalResult,
    RetrievalService,
    VectorStore,
    get_retrieval_score_threshold,
    get_retrieval_top_k,
)
from app.models import EmbeddingProvider, ModelProvider, estimate_cost_usd
from app.storage import (
    ChatSession,
    append_message,
    create_session,
    get_history,
    get_or_create_subject,
    get_session,
    get_subject,
    get_subject_by_name,
    list_documents,
    list_subjects,
    log_usage_event,
)

from .command_router import route_command
from .prompt_builder import build_grounded_prompt
from .visual_directives import VisualDirective, extract_directives

__all__ = [
    "DEFAULT_SUBJECT",
    "SourceCitation",
    "TurnResult",
    "VisualDirective",
    "VoiceTurnResult",
    "build_grounded_prompt",
    "extract_directives",
    "handle_turn",
    "handle_voice_turn",
]

DEFAULT_SUBJECT = "general"

_NO_DOCUMENTS_REPLY = "No textbook has been uploaded for this subject yet. Please upload your study material first."
_NOT_COVERED_REPLY = "This topic isn't covered in your textbook. Try asking about something from your syllabus."


@dataclass(frozen=True)
class SourceCitation:
    page_number: int | None
    score: float


@dataclass(frozen=True)
class TurnResult:
    reply: str
    session_id: str
    sources: list[SourceCitation] = field(default_factory=list)
    visual_directives: list[VisualDirective] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceTurnResult:
    reply: str
    session_id: str
    transcribed_text: str
    sources: list[SourceCitation] = field(default_factory=list)
    is_command: bool = False
    command_type: str | None = None
    # Set only when a switch_subject command actually resolved to a real
    # subject -- lets the voice WS endpoint know the active subject changed.
    new_subject: str | None = None
    visual_directives: list[VisualDirective] = field(default_factory=list)


async def handle_turn(
    db: AsyncSession,
    provider: ModelProvider,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    session_id: str | None,
    user_message: str,
    subject: str = DEFAULT_SUBJECT,
    temperature: float | None = None,
    mode: str = "textbook",
) -> TurnResult:
    """Run one conversational turn, grounded in the subject's ingested textbook.

    The grounding guardrail (technical-design.md: answer only from retrieved
    context, say so explicitly otherwise) is enforced here, before the Model
    Provider is ever called:
      - no documents ingested for the subject -> refuse, canned reply.
      - documents exist but nothing retrieved clears RETRIEVAL_SCORE_THRESHOLD
        -> refuse, canned "not covered" reply.
      - otherwise -> build the grounded prompt from the retrieved chunks and
        call the Model Provider.

    temperature is passed straight through to the Model Provider (None ->
    provider default). Normal chat leaves it unset; the qa-check endpoint
    passes 0 for deterministic eval output.

    mode ("textbook" default, or "teacher") selects the system prompt's
    framing -- see prompt_builder.TEACHING_MODES. Retrieval and the
    guardrail above are identical either way; mode only changes how the
    Model Provider is told to present the same retrieved passages.
    """
    session = await _load_or_create_session(db, session_id, subject, mode=mode)
    session.mode = mode

    if not await list_documents(db, session.subject_id):
        return await _finish_turn(db, session, user_message, _NO_DOCUMENTS_REPLY)

    subject_row = await get_subject(db, session.subject_id)
    assert subject_row is not None  # a session's subject_id always references a real row

    prior_messages = await get_history(db, session.id)

    retrieval_result = await RetrievalService(embedding_provider, vector_store).retrieve(
        user_message, collection_for_subject(session.subject_id), top_k=get_retrieval_top_k()
    )
    guardrail_triggered = retrieval_result.best_score < get_retrieval_score_threshold()
    await _log_retrieval_usage(db, session, embedding_provider, retrieval_result, guardrail_triggered)

    if guardrail_triggered:
        return await _finish_turn(db, session, user_message, _NOT_COVERED_REPLY)

    grounded_messages = build_grounded_prompt(
        user_message,
        retrieval_result.chunks,
        subject_row.name,
        [{"role": m.role, "content": m.content} for m in prior_messages],
        mode=mode,
    )

    await append_message(db, session.id, role="user", content=user_message)

    started = time.perf_counter()
    response = provider.generate(grounded_messages, temperature=temperature)
    latency_ms = (time.perf_counter() - started) * 1000

    await append_message(db, session.id, role="assistant", content=response.content)

    cost_usd = estimate_cost_usd(provider.model_name, response.usage.input_tokens, response.usage.output_tokens)
    await log_usage_event(
        db,
        session_id=session.id,
        subject_id=session.subject_id,
        provider=provider.provider_name,
        model=provider.model_name,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        event_type="generation",
    )

    sources = [SourceCitation(page_number=c.page_number, score=c.score) for c in retrieval_result.chunks]
    _, directives = extract_directives(response.content)
    return TurnResult(reply=response.content, session_id=session.id, sources=sources, visual_directives=directives)


async def handle_voice_turn(
    db: AsyncSession,
    provider: ModelProvider,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    session_id: str | None,
    transcribed_text: str,
    subject: str = DEFAULT_SUBJECT,
    temperature: float | None = None,
    mode: str = "textbook",
) -> VoiceTurnResult:
    """Voice-loop wrapper around handle_turn(): checks transcribed_text for a
    control command first (command_router.route_command()) -- a recognized
    command never reaches retrieval or the Model Provider, matching
    architecture.md's "control-command routing ahead of tutoring content".
    Anything not recognized as a command falls through to the normal
    retrieval-grounded handle_turn() flow, unchanged.

    Control-command turns aren't appended to message history: they're a
    separate control channel, not tutoring conversation, so they shouldn't
    pollute the grounded prompt's history context for later questions.
    """
    session = await _load_or_create_session(db, session_id, subject)

    available_subjects = [subject_row.name for subject_row, _ in await list_subjects(db)]
    command = route_command(transcribed_text, available_subjects)

    if command is not None:
        new_subject: str | None = None
        if command.command_type == "switch_subject":
            matched_subject = await get_subject_by_name(db, command.args.get("subject", ""))
            if matched_subject is not None:
                session = await create_session(db, matched_subject.id)
                new_subject = matched_subject.name

        return VoiceTurnResult(
            reply=command.response_text,
            session_id=session.id,
            transcribed_text=transcribed_text,
            is_command=True,
            command_type=command.command_type,
            new_subject=new_subject,
        )

    turn_result = await handle_turn(
        db,
        provider,
        embedding_provider,
        vector_store,
        session_id=session.id,
        user_message=transcribed_text,
        subject=subject,
        temperature=temperature,
        mode=mode,
    )
    return VoiceTurnResult(
        reply=turn_result.reply,
        session_id=turn_result.session_id,
        transcribed_text=transcribed_text,
        sources=turn_result.sources,
        visual_directives=turn_result.visual_directives,
    )


async def _load_or_create_session(
    db: AsyncSession, session_id: str | None, subject: str, mode: str = "textbook"
) -> ChatSession:
    if session_id is not None:
        session = await get_session(db, session_id)
        if session is None:
            raise LookupError(f"Unknown session_id: {session_id!r}")
        return session

    subject_row = await get_or_create_subject(db, subject)
    return await create_session(db, subject_row.id, mode=mode)


async def _finish_turn(db: AsyncSession, session: ChatSession, user_message: str, reply: str) -> TurnResult:
    """Appends the turn (guardrail-refused turns are still saved to history) and returns it, unsourced."""
    await append_message(db, session.id, role="user", content=user_message)
    await append_message(db, session.id, role="assistant", content=reply)
    return TurnResult(reply=reply, session_id=session.id, sources=[])


async def _log_retrieval_usage(
    db: AsyncSession,
    session: ChatSession,
    embedding_provider: EmbeddingProvider,
    retrieval_result: RetrievalResult,
    guardrail_triggered: bool,
) -> None:
    """Logs two usage_events rows per retrieve() call: a "retrieval" row
    (metrics only -- best_score/chunk_count/guardrail_triggered, no token
    cost of its own) and an "embedding" row (the query embedding call's
    actual tokens/cost), matching how ingestion logs its embedding calls.
    """
    await log_usage_event(
        db,
        session_id=session.id,
        subject_id=session.subject_id,
        provider=embedding_provider.provider_name,
        model=embedding_provider.model_name,
        input_tokens=None,
        output_tokens=None,
        cost_usd=0.0,
        event_type="retrieval",
        retrieval_best_score=retrieval_result.best_score,
        retrieval_chunk_count=len(retrieval_result.chunks),
        guardrail_triggered=guardrail_triggered,
    )

    embedding_usage = retrieval_result.query_embedding_usage.usage
    embedding_cost = estimate_cost_usd(
        embedding_provider.model_name, embedding_usage.input_tokens, embedding_usage.output_tokens
    )
    await log_usage_event(
        db,
        session_id=session.id,
        subject_id=session.subject_id,
        provider=embedding_provider.provider_name,
        model=embedding_provider.model_name,
        input_tokens=embedding_usage.input_tokens,
        output_tokens=embedding_usage.output_tokens,
        cost_usd=embedding_cost,
        event_type="embedding",
    )
