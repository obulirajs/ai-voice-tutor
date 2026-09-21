from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.knowledge import Chunk, SearchResult, VectorStore
from app.knowledge.retrieval import RetrievedChunk
from app.models import (
    EmbeddingProvider,
    EmbeddingResponse,
    Message,
    ModelProvider,
    ModelResponse,
    Usage,
)
from app.orchestration import build_grounded_prompt, handle_turn
from app.storage import create_document, get_history, get_or_create_subject
from app.storage.models import Base, UsageEvent

Scenario = Callable[[AsyncSession], Coroutine[Any, Any, None]]


def run(scenario: Scenario) -> None:
    """Run one test scenario against a fresh in-memory SQLite engine, then dispose it."""

    async def wrapper() -> None:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            await scenario(db)

        await engine.dispose()

    asyncio.run(wrapper())


async def _seed_document(db: AsyncSession, subject_name: str) -> int:
    """Creates a subject with one ingested document -- satisfies handle_turn's
    "has this subject got a textbook uploaded?" guardrail check. Returns the
    subject_id."""
    subject = await get_or_create_subject(db, subject_name)
    await create_document(
        db,
        subject_id=subject.id,
        filename="chapter1.pdf",
        collection=f"subject_{subject.id}",
        page_count=5,
    )
    return subject.id


class FakeProvider(ModelProvider):
    def __init__(
        self,
        reply: str,
        *,
        provider: str = "anthropic",
        model: str = "claude-sonnet-5",
        input_tokens: int | None = 1000,
        output_tokens: int | None = 500,
    ) -> None:
        self._reply = reply
        self._provider = provider
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.received: list[Message] = []
        self.received_temperature: float | None = None
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        self.received = messages
        self.received_temperature = temperature
        return ModelResponse(
            content=self._reply,
            usage=Usage(input_tokens=self._input_tokens, output_tokens=self._output_tokens),
        )


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, input_tokens: int | None = 50, output_tokens: int | None = None) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.embed_call_count = 0

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "nomic-embed-text"

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        self.embed_call_count += 1
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2] for _ in texts],
            usage=Usage(input_tokens=self._input_tokens, output_tokens=self._output_tokens),
        )


class FakeVectorStore(VectorStore):
    """Returns the same fixed search results regardless of query embedding,
    so retrieval scores are precisely controllable from the test."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits
        self.search_call_count = 0

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        self.search_call_count += 1
        return self._hits[:top_k]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


def test_handle_turn_creates_session_when_none_given() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="Bonjour !")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=1, distance=0.05)])

        result = await handle_turn(
            db, provider, embedding_provider, vector_store, session_id=None, user_message="Bonjour"
        )

        assert result.reply == "Bonjour !"
        assert result.session_id

    run(scenario)


def test_handle_turn_defaults_new_sessions_to_general_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="ok")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=1, distance=0.05)])

        await handle_turn(db, provider, embedding_provider, vector_store, session_id=None, user_message="hi")

        subject = await get_or_create_subject(db, "general")
        assert subject.name == "general"

    run(scenario)


def test_handle_turn_appends_user_and_assistant_messages() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="Bonjour ! Ca va ?")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=1, distance=0.05)])

        result = await handle_turn(
            db, provider, embedding_provider, vector_store, session_id=None, user_message="Bonjour"
        )

        history = await get_history(db, result.session_id)
        assert [(m.role, m.content) for m in history] == [
            ("user", "Bonjour"),
            ("assistant", "Bonjour ! Ca va ?"),
        ]

    run(scenario)


def test_handle_turn_continues_existing_session_with_prior_history() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=5, distance=0.05)])

        provider = FakeProvider(reply="reply 1")
        first = await handle_turn(
            db, provider, embedding_provider, vector_store, session_id=None, user_message="turn 1"
        )

        provider2 = FakeProvider(reply="reply 2")
        await handle_turn(
            db,
            provider2,
            embedding_provider,
            vector_store,
            session_id=first.session_id,
            user_message="turn 2",
        )

        assert provider2.received[0]["role"] == "system"
        assert provider2.received[1:] == [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
        ]

    run(scenario)


def test_handle_turn_raises_lookup_error_for_unknown_session() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="unused")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([])

        with pytest.raises(LookupError):
            await handle_turn(
                db, provider, embedding_provider, vector_store, session_id="does-not-exist", user_message="hi"
            )

    run(scenario)


def test_handle_turn_logs_usage_event_with_estimated_cost() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(
            reply="ok",
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=1, distance=0.05)])

        result = await handle_turn(db, provider, embedding_provider, vector_store, session_id=None, user_message="hi")

        rows = (
            (
                await db.execute(
                    select(UsageEvent).where(UsageEvent.session_id == result.session_id, UsageEvent.event_type == "generation")
                )
            )
            .scalars()
            .all()
        )

        assert len(rows) == 1
        event = rows[0]
        assert event.provider == "anthropic"
        assert event.model == "claude-sonnet-5"
        assert event.input_tokens == 1_000_000
        assert event.output_tokens == 1_000_000
        # $2/MTok in + $10/MTok out, at 1M tokens each => $2 + $10 = $12
        assert event.cost_usd == pytest.approx(12.0)

    run(scenario)


def test_handle_turn_costs_zero_for_ollama() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(
            reply="ok",
            provider="ollama",
            model="llama3.2:1b",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([SearchResult(text="chunk", document_id=1, page_number=1, distance=0.05)])

        result = await handle_turn(db, provider, embedding_provider, vector_store, session_id=None, user_message="hi")

        rows = (
            (
                await db.execute(
                    select(UsageEvent).where(UsageEvent.session_id == result.session_id, UsageEvent.event_type == "generation")
                )
            )
            .scalars()
            .all()
        )

        assert rows[0].cost_usd == 0.0

    run(scenario)


# --- handle_turn: retrieval-grounded loop + guardrail (Phase 3, prompt 2) ---


def test_handle_turn_grounds_reply_in_retrieved_chunks_and_reports_sources() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="Le present tense answer.")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore(
            [
                SearchResult(text="Le present tense chunk.", document_id=1, page_number=7, distance=0.1),
                SearchResult(text="Another chunk.", document_id=1, page_number=8, distance=0.3),
            ]
        )

        result = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message="How does the present tense work?",
        )

        assert vector_store.search_call_count == 1
        assert embedding_provider.embed_call_count == 1
        assert provider.call_count == 1
        assert provider.received[0]["role"] == "system"
        assert "[Passage 1 — page 7]" in provider.received[0]["content"]
        assert "Le present tense chunk." in provider.received[0]["content"]
        assert provider.received[-1] == {"role": "user", "content": "How does the present tense work?"}

        assert result.reply == "Le present tense answer."
        assert [(s.page_number, s.score) for s in result.sources] == [(7, pytest.approx(0.9)), (8, pytest.approx(0.7))]
        assert provider.received_temperature is None

    run(scenario)


def test_handle_turn_defaults_to_textbook_mode() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="unused")

        await handle_turn(
            db,
            provider,
            FakeEmbeddingProvider(),
            FakeVectorStore([SearchResult(text="Chunk.", document_id=1, page_number=1, distance=0.1)]),
            session_id=None,
            user_message="q",
        )

        system_content = provider.received[0]["content"]
        assert "MUST answer strictly from the passages" in system_content
        assert "experienced, friendly classroom teacher" not in system_content

    run(scenario)


def test_handle_turn_teacher_mode_changes_system_prompt_sent_to_provider() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="unused")

        await handle_turn(
            db,
            provider,
            FakeEmbeddingProvider(),
            FakeVectorStore([SearchResult(text="Chunk.", document_id=1, page_number=1, distance=0.1)]),
            session_id=None,
            user_message="q",
            mode="teacher",
        )

        system_content = provider.received[0]["content"]
        assert "experienced, friendly classroom teacher" in system_content
        assert "**💡 Beyond the textbook:**" in system_content

    run(scenario)


def test_handle_turn_passes_temperature_through_to_provider() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="unused")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=7, distance=0.1)]
        )

        await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message="How does the present tense work?",
            temperature=0.0,
        )

        assert provider.received_temperature == 0.0

    run(scenario)


def test_handle_turn_guardrail_refuses_when_best_score_below_threshold() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="unused")
        embedding_provider = FakeEmbeddingProvider()
        # distance 0.95 -> score 0.05, well below the 0.55 default RETRIEVAL_SCORE_THRESHOLD
        vector_store = FakeVectorStore([SearchResult(text="unrelated", document_id=1, page_number=1, distance=0.95)])

        result = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message="What about photosynthesis?",
        )

        assert provider.call_count == 0
        assert result.reply == "This topic isn't covered in your textbook. Try asking about something from your syllabus."
        assert result.sources == []

        history = await get_history(db, result.session_id)
        assert [(m.role, m.content) for m in history] == [
            ("user", "What about photosynthesis?"),
            ("assistant", "This topic isn't covered in your textbook. Try asking about something from your syllabus."),
        ]

        rows = (await db.execute(select(UsageEvent).where(UsageEvent.session_id == result.session_id))).scalars().all()
        assert not any(r.event_type == "generation" for r in rows)
        retrieval_events = [r for r in rows if r.event_type == "retrieval"]
        assert len(retrieval_events) == 1
        assert retrieval_events[0].guardrail_triggered is True

    run(scenario)


def test_handle_turn_refuses_when_subject_has_no_documents() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="unused")
        embedding_provider = FakeEmbeddingProvider()
        vector_store = FakeVectorStore([])

        result = await handle_turn(db, provider, embedding_provider, vector_store, session_id=None, user_message="Bonjour")

        assert result.reply == (
            "No textbook has been uploaded for this subject yet. Please upload your study material first."
        )
        assert result.sources == []
        assert provider.call_count == 0
        assert embedding_provider.embed_call_count == 0
        assert vector_store.search_call_count == 0

        history = await get_history(db, result.session_id)
        assert [m.content for m in history] == [
            "Bonjour",
            "No textbook has been uploaded for this subject yet. Please upload your study material first.",
        ]

    run(scenario)


def test_handle_turn_logs_retrieval_and_embedding_usage_events() -> None:
    async def scenario(db: AsyncSession) -> None:
        await _seed_document(db, "general")
        provider = FakeProvider(reply="ok")
        embedding_provider = FakeEmbeddingProvider(input_tokens=42)
        vector_store = FakeVectorStore(
            [
                SearchResult(text="a", document_id=1, page_number=1, distance=0.1),
                SearchResult(text="b", document_id=1, page_number=2, distance=0.2),
            ]
        )

        result = await handle_turn(db, provider, embedding_provider, vector_store, session_id=None, user_message="hi")

        rows = (await db.execute(select(UsageEvent).where(UsageEvent.session_id == result.session_id))).scalars().all()
        by_type = {row.event_type: row for row in rows}
        assert set(by_type) == {"retrieval", "embedding", "generation"}

        retrieval_event = by_type["retrieval"]
        assert retrieval_event.retrieval_best_score == pytest.approx(0.9)
        assert retrieval_event.retrieval_chunk_count == 2
        assert retrieval_event.guardrail_triggered is False

        embedding_event = by_type["embedding"]
        assert embedding_event.provider == "ollama"
        assert embedding_event.model == "nomic-embed-text"
        assert embedding_event.input_tokens == 42
        assert embedding_event.cost_usd == 0.0  # ollama has no listed pricing

    run(scenario)


# --- build_grounded_prompt (Phase 3: grounding guardrail) ---


def test_build_grounded_prompt_system_message_includes_passages_with_page_numbers() -> None:
    chunks = [
        RetrievedChunk(text="Le passe compose is formed with avoir or etre.", page_number=42, score=0.9, document_id=1),
        RetrievedChunk(text="Irregular verbs conjugate differently.", page_number=43, score=0.8, document_id=1),
    ]

    messages = build_grounded_prompt("How is the passe compose formed?", chunks, "French", [])

    system_message = messages[0]
    assert system_message["role"] == "system"
    assert "[Passage 1 — page 42]" in system_message["content"]
    assert "Le passe compose is formed with avoir or etre." in system_message["content"]
    assert "[Passage 2 — page 43]" in system_message["content"]
    assert "Irregular verbs conjugate differently." in system_message["content"]


def test_build_grounded_prompt_system_message_instructs_grounding_and_citation() -> None:
    messages = build_grounded_prompt("q", [], "French", [])

    system_content = messages[0]["content"]
    assert "CRITICAL" in system_content
    assert "MUST answer strictly from the passages" in system_content
    assert "Do NOT use your own knowledge" in system_content
    assert "(page X)" in system_content
    assert "French" in system_content


def test_build_grounded_prompt_empty_chunks_still_instructs_not_covered_reply() -> None:
    messages = build_grounded_prompt("What happened in 1789?", [], "French", [])

    system_content = messages[0]["content"]
    assert "This topic isn't covered in your textbook. Try asking about something from your syllabus." in (
        system_content
    )
    assert "No textbook passages were found for this question." in system_content


def test_build_grounded_prompt_includes_conversation_history_before_question() -> None:
    history = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour ! Ca va ?"},
    ]

    messages = build_grounded_prompt("What's next?", [], "French", history)

    assert messages[1] == {"role": "user", "content": "Bonjour"}
    assert messages[2] == {"role": "assistant", "content": "Bonjour ! Ca va ?"}
    assert messages[-1] == {"role": "user", "content": "What's next?"}


def test_build_grounded_prompt_question_is_final_user_message() -> None:
    messages = build_grounded_prompt("Final question", [], "French", [])

    assert messages[-1] == {"role": "user", "content": "Final question"}


def test_build_grounded_prompt_truncates_history_to_last_n_messages() -> None:
    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]

    messages = build_grounded_prompt("q", [], "French", history, max_history_messages=4)

    # 1 system + 4 history + 1 question
    assert len(messages) == 6
    assert messages[1] == {"role": "user", "content": "turn 16"}
    assert messages[-2] == {"role": "user", "content": "turn 19"}


# --- build_grounded_prompt teaching modes ---


def test_build_grounded_prompt_default_mode_is_textbook() -> None:
    messages = build_grounded_prompt("q", [], "French", [])

    system_content = messages[0]["content"]
    assert "MUST answer strictly from the passages" in system_content
    assert "experienced, friendly classroom teacher" not in system_content


def test_build_grounded_prompt_textbook_mode_matches_default() -> None:
    default_messages = build_grounded_prompt("q", [], "French", [])
    explicit_messages = build_grounded_prompt("q", [], "French", [], mode="textbook")

    assert default_messages[0]["content"] == explicit_messages[0]["content"]


def test_build_grounded_prompt_teacher_mode_produces_teacher_system_prompt() -> None:
    messages = build_grounded_prompt("q", [], "French", [], mode="teacher")

    system_content = messages[0]["content"]
    assert "experienced, friendly classroom teacher" in system_content
    assert "**💡 Beyond the textbook:**" in system_content
    # The guardrail is identical in both modes -- still enforced in teacher mode.
    assert "do NOT introduce new factual claims" in system_content
    assert "This topic isn't covered in your textbook" in system_content


def test_build_grounded_prompt_unknown_mode_falls_back_to_textbook() -> None:
    messages = build_grounded_prompt("q", [], "French", [], mode="invalid")

    system_content = messages[0]["content"]
    assert "MUST answer strictly from the passages" in system_content


def test_build_grounded_prompt_formula_instruction_does_not_force_latex_everywhere() -> None:
    """Regression coverage: the old blanket "When presenting a formula..."
    wording read as "always append LaTeX"; both modes now say "If your
    answer involves..." plus an explicit "Do NOT append" instruction."""
    for mode in ("textbook", "teacher"):
        system_content = build_grounded_prompt("q", [], "French", [], mode=mode)[0]["content"]
        assert "When presenting a formula" not in system_content
        assert "If your answer involves a formula" in system_content
        assert "Do NOT append formulas" in system_content
