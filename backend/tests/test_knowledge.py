from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.knowledge import (
    Chunk,
    GoldenQuestion,
    SearchResult,
    VectorStore,
    check_ingestion_consistency,
    get_vector_store,
    parse_expected_page,
    run_golden_qa_check,
    summarize_golden_qa_results,
)
from app.knowledge import store as vec_store_sql
from app.knowledge.sqlite_vec_store import SqliteVecStore
from app.models import EmbeddingProvider


@pytest.fixture
def vector_store(tmp_path: Path) -> SqliteVecStore:
    return SqliteVecStore(db_path=str(tmp_path / "vector_store.db"))


def test_get_vector_store_returns_sqlite_vec_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)

    instance = get_vector_store()

    assert isinstance(instance, SqliteVecStore)


def test_get_vector_store_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "bogus")

    with pytest.raises(ValueError):
        get_vector_store()


def test_search_finds_nearest_chunk_by_embedding(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        chunks = [
            Chunk(text="Le present tense chunk", embedding=[1.0, 0.0, 0.0, 0.0], document_id=1, page_number=1),
            Chunk(text="Le passe compose chunk", embedding=[0.0, 1.0, 0.0, 0.0], document_id=1, page_number=2),
            Chunk(text="Vocabulaire chunk", embedding=[0.0, 0.0, 1.0, 0.0], document_id=1, page_number=3),
        ]
        await vector_store.upsert_chunks("subject_1", chunks)

        results = await vector_store.search("subject_1", [0.9, 0.1, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].text == "Le present tense chunk"
        assert results[0].page_number == 1
        assert results[0].document_id == 1

    asyncio.run(scenario())


def test_search_orders_by_distance_ascending(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        chunks = [
            Chunk(text="far", embedding=[0.0, 1.0], document_id=1, page_number=1),
            Chunk(text="near", embedding=[0.99, 0.01], document_id=1, page_number=2),
            Chunk(text="mid", embedding=[0.7, 0.3], document_id=1, page_number=3),
        ]
        await vector_store.upsert_chunks("subject_2", chunks)

        results = await vector_store.search("subject_2", [1.0, 0.0], top_k=3)

        assert [r.text for r in results] == ["near", "mid", "far"]
        assert results[0].distance < results[1].distance < results[2].distance

    asyncio.run(scenario())


def test_collections_do_not_leak_into_each_other(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        await vector_store.upsert_chunks("subject_french", [Chunk(text="french content", embedding=[1.0, 0.0])])
        await vector_store.upsert_chunks("subject_maths", [Chunk(text="maths content", embedding=[1.0, 0.0])])

        french_results = await vector_store.search("subject_french", [1.0, 0.0], top_k=5)

        assert [r.text for r in french_results] == ["french content"]

    asyncio.run(scenario())


def test_upsert_chunks_is_a_noop_for_empty_list(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        # Must not raise even though no collection/table exists yet.
        await vector_store.upsert_chunks("subject_empty", [])

    asyncio.run(scenario())


def test_delete_document_chunks_removes_only_target_document_with_no_orphans(tmp_path: Path) -> None:
    """Exercises the real sqlite-vec deletion SQL (app.knowledge.store),
    not a fake -- the DELETE /documents/{id} endpoint's tests only prove it
    calls VectorStore.delete_document_chunks(), via a FakeVectorStore. This
    verifies the actual implementation: deleting one document's chunks must
    not touch another document's chunks in the same collection (a missing
    document_id filter would wipe the whole collection instead), and must
    leave no orphaned rows in either the chunks metadata table or the vec0
    virtual table underneath it.
    """

    async def scenario() -> None:
        db_path = str(tmp_path / "vector_store.db")
        vector_store = SqliteVecStore(db_path=db_path)

        doc_a_chunks = [
            Chunk(text="doc A chunk 1", embedding=[1.0, 0.0], document_id=1, page_number=1),
            Chunk(text="doc A chunk 2", embedding=[0.9, 0.1], document_id=1, page_number=2),
        ]
        doc_b_chunks = [Chunk(text="doc B chunk 1", embedding=[0.0, 1.0], document_id=2, page_number=1)]
        await vector_store.upsert_chunks("subject_1", doc_a_chunks + doc_b_chunks)

        await vector_store.delete_document_chunks("subject_1", document_id=1)

        # Document B's chunk is untouched and still searchable.
        remaining = await vector_store.search("subject_1", [0.0, 1.0], top_k=5)
        assert [r.text for r in remaining] == ["doc B chunk 1"]

        conn = vec_store_sql.connect(db_path)
        try:
            meta_rows_for_doc_a = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection = ? AND document_id = ?",
                ("subject_1", 1),
            ).fetchone()[0]
            assert meta_rows_for_doc_a == 0

            table = vec_store_sql.collection_table("subject_1")
            total_vec_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert total_vec_rows == 1  # only doc B's chunk remains, not orphaned doc A rows
        finally:
            conn.close()

    asyncio.run(scenario())


def test_delete_document_chunks_is_a_noop_for_unknown_document(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        chunks = [Chunk(text="doc A chunk", embedding=[1.0, 0.0], document_id=1, page_number=1)]
        await vector_store.upsert_chunks("subject_1", chunks)

        # Must not raise even though document_id 999 has no chunks here.
        await vector_store.delete_document_chunks("subject_1", document_id=999)

        remaining = await vector_store.search("subject_1", [1.0, 0.0], top_k=5)
        assert [r.text for r in remaining] == ["doc A chunk"]

    asyncio.run(scenario())


# --- check_ingestion_consistency (Part 1: automatic post-ingestion check) ---


def test_check_ingestion_consistency_passes_when_every_chunk_finds_itself(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        chunks = [
            Chunk(text="chunk A", embedding=[1.0, 0.0], page_number=1),
            Chunk(text="chunk B", embedding=[0.0, 1.0], page_number=2),
        ]
        await vector_store.upsert_chunks("subject_1", chunks)

        result = await check_ingestion_consistency(vector_store, "subject_1", chunks, high_risk=[False, False])

        assert result.sampled == 2
        assert result.passed == 2
        assert result.failed == []

    asyncio.run(scenario())


def test_check_ingestion_consistency_samples_high_risk_chunks_first(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        # 3 low-risk chunks + 1 high-risk (OCR) chunk, sample_size=1 -- the
        # OCR chunk must be the one sampled and checked, since that's the
        # highest-risk content per the ingestion "Ensuring retrieval
        # accuracy" checkpoint.
        chunks = [
            Chunk(text="normal 1", embedding=[1.0, 0.0, 0.0, 0.0]),
            Chunk(text="normal 2", embedding=[0.0, 1.0, 0.0, 0.0]),
            Chunk(text="ocr chunk", embedding=[0.0, 0.0, 1.0, 0.0], page_number=9),
            Chunk(text="normal 3", embedding=[0.0, 0.0, 0.0, 1.0]),
        ]
        await vector_store.upsert_chunks("subject_1", chunks)

        result = await check_ingestion_consistency(
            vector_store,
            "subject_1",
            chunks,
            high_risk=[False, False, True, False],
            sample_size=1,
        )

        assert result.sampled == 1
        assert result.passed == 1

    asyncio.run(scenario())


def test_check_ingestion_consistency_reports_failure_for_broken_storage() -> None:
    class BrokenVectorStore(VectorStore):
        """search() always returns content unrelated to the query -- simulates
        a corrupted/mismatched embedding link between storage and retrieval.
        """

        async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
            return None

        async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
            return [SearchResult(text="totally unrelated content", document_id=None, page_number=None, distance=99.0)]

        async def delete_document_chunks(self, collection: str, document_id: int) -> None:
            return None

    async def scenario() -> None:
        chunks = [Chunk(text="chunk A", embedding=[1.0, 0.0], page_number=3)]

        result = await check_ingestion_consistency(BrokenVectorStore(), "subject_1", chunks, high_risk=[True])

        assert result.sampled == 1
        assert result.passed == 0
        assert len(result.failed) == 1
        assert result.failed[0].chunk_index == 0
        assert result.failed[0].page_number == 3

    asyncio.run(scenario())


def test_check_ingestion_consistency_handles_no_chunks(vector_store: VectorStore) -> None:
    async def scenario() -> None:
        result = await check_ingestion_consistency(vector_store, "subject_1", [], high_risk=[])

        assert result.sampled == 0
        assert result.passed == 0
        assert result.failed == []

    asyncio.run(scenario())


# --- run_golden_qa_check / summarize_golden_qa_results (Part 2: golden QA endpoint) ---


class _FixedResultsVectorStore(VectorStore):
    """Returns the same fixed search results regardless of query embedding,
    so page_match/low_confidence math can be tested precisely without a
    real embedding model."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self._hits[:top_k]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


class _FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]  # value is irrelevant -- the fake store ignores it


def test_parse_expected_page_extracts_first_integer_after_p() -> None:
    assert parse_expected_page("Leçon 1, p.4-5") == 4
    assert parse_expected_page("p. 13 (conjugation table)") == 13
    assert parse_expected_page("Not in the textbook") is None


def test_run_golden_qa_check_page_match_true_within_tolerance() -> None:
    async def scenario() -> None:
        store = _FixedResultsVectorStore([SearchResult(text="x", document_id=1, page_number=5, distance=0.1)])
        questions = [GoldenQuestion(question="q", expected_answer="a", source="p.4", type="grammar")]

        results = await run_golden_qa_check(store, _FakeEmbeddingProvider(), "subject_1", questions)

        assert results[0].page_match is True  # |5-4| == 1, within tolerance
        assert results[0].low_confidence is False  # similarity 0.9 >= 0.5

    asyncio.run(scenario())


def test_run_golden_qa_check_page_match_false_outside_tolerance() -> None:
    async def scenario() -> None:
        store = _FixedResultsVectorStore([SearchResult(text="x", document_id=1, page_number=99, distance=0.1)])
        questions = [GoldenQuestion(question="q", expected_answer="a", source="p.4", type="grammar")]

        results = await run_golden_qa_check(store, _FakeEmbeddingProvider(), "subject_1", questions)

        assert results[0].page_match is False

    asyncio.run(scenario())


def test_run_golden_qa_check_out_of_scope_question_has_no_page_match_and_is_low_confidence() -> None:
    """An out-of-scope golden question ("Not in the textbook") has no page
    number to parse, so page_match is None (not applicable) -- and
    low_confidence=True here is the *expected*, correct outcome, not a bug.
    """

    async def scenario() -> None:
        store = _FixedResultsVectorStore([SearchResult(text="x", document_id=1, page_number=5, distance=1.5)])
        questions = [
            GoldenQuestion(
                question="What does the textbook say about Victor Hugo?",
                expected_answer="Not covered.",
                source="Not in the textbook",
                type="out-of-scope",
            )
        ]

        results = await run_golden_qa_check(store, _FakeEmbeddingProvider(), "subject_1", questions)

        assert results[0].page_match is None
        assert results[0].low_confidence is True  # similarity = 1 - 1.5 = -0.5 < 0.5

    asyncio.run(scenario())


def test_run_golden_qa_check_no_hits_is_low_confidence() -> None:
    async def scenario() -> None:
        store = _FixedResultsVectorStore([])
        questions = [GoldenQuestion(question="q", expected_answer="a", source="p.1", type="grammar")]

        results = await run_golden_qa_check(store, _FakeEmbeddingProvider(), "subject_1", questions)

        assert results[0].retrieved == []
        assert results[0].low_confidence is True
        assert results[0].page_match is False

    asyncio.run(scenario())


def test_summarize_golden_qa_results_splits_by_type() -> None:
    async def scenario() -> None:
        store = _FixedResultsVectorStore([SearchResult(text="x", document_id=1, page_number=4, distance=0.1)])
        questions = [
            GoldenQuestion(question="q1", expected_answer="a", source="p.4", type="grammar"),
            GoldenQuestion(question="q2", expected_answer="a", source="Not in the textbook", type="out-of-scope"),
        ]

        results = await run_golden_qa_check(store, _FakeEmbeddingProvider(), "subject_1", questions)
        summary = summarize_golden_qa_results(results)

        assert summary.overall.total == 2
        assert summary.overall.page_match_true == 1
        assert summary.overall.page_match_unknown == 1
        assert summary.by_type["grammar"].total == 1
        assert summary.by_type["grammar"].page_match_true == 1
        assert summary.by_type["out-of-scope"].total == 1
        assert summary.by_type["out-of-scope"].page_match_unknown == 1

    asyncio.run(scenario())
