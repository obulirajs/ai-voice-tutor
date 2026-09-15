from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models import EmbeddingProvider

from .base import VectorStore

_PAGE_PATTERN = re.compile(r"p\.?\s*(\d+)", re.IGNORECASE)
_SNIPPET_LENGTH = 200
_PAGE_MATCH_TOLERANCE = 1
_DEFAULT_TOP_K = 5
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    expected_answer: str
    source: str
    type: str


@dataclass(frozen=True)
class RetrievedChunk:
    page_number: int | None
    snippet: str
    similarity: float


@dataclass(frozen=True)
class GoldenQuestionResult:
    question: GoldenQuestion
    retrieved: list[RetrievedChunk]
    page_match: bool | None
    low_confidence: bool


@dataclass(frozen=True)
class TypeSummary:
    total: int = 0
    page_match_true: int = 0
    page_match_false: int = 0
    page_match_unknown: int = 0
    low_confidence_true: int = 0
    low_confidence_false: int = 0


@dataclass(frozen=True)
class GoldenQASummary:
    overall: TypeSummary
    by_type: dict[str, TypeSummary] = field(default_factory=dict)


def parse_expected_page(source: str) -> int | None:
    """Best-effort page number extraction, e.g. "p.34, Chapter 3" -> 34.

    Lenient by design: takes the first integer after "p."/"p ", nothing
    more. Returns None (never a guess) when the source has no such pattern
    — notably for out-of-scope golden questions whose source reads
    something like "Not in the textbook", where no page match is expected
    or applicable.
    """
    match = _PAGE_PATTERN.search(source)
    return int(match.group(1)) if match else None


async def run_golden_qa_check(
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    collection: str,
    questions: list[GoldenQuestion],
    *,
    top_k: int = _DEFAULT_TOP_K,
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> list[GoldenQuestionResult]:
    """Read-only retrieval-quality check against already-stored vectors.

    Embeds each question, retrieves the top_k chunks from the subject's
    collection, and reports two structural signals (page_match,
    low_confidence). Does not grade against expected_answer text — no LLM
    call, matching the "no chat loop exists yet" constraint; that grading
    is Phase 3 territory once the grounded chat loop and its confidence
    handling exist. Never touches ingestion — this only calls
    VectorStore.search() against whatever was already ingested.

    For an out-of-scope question (a source with no parseable page number,
    e.g. "Not in the textbook"), page_match is naturally None, and
    low_confidence=True is the *expected*, correct outcome — not a bug.
    """
    results: list[GoldenQuestionResult] = []

    for golden_question in questions:
        (vector,) = embedding_provider.embed([golden_question.question]).embeddings
        hits = await vector_store.search(collection, vector, top_k=top_k)

        retrieved = [
            RetrievedChunk(
                page_number=hit.page_number,
                snippet=hit.text[:_SNIPPET_LENGTH],
                similarity=1.0 - hit.distance,
            )
            for hit in hits
        ]

        expected_page = parse_expected_page(golden_question.source)
        page_match: bool | None = None
        if expected_page is not None:
            page_match = any(
                r.page_number is not None and abs(r.page_number - expected_page) <= _PAGE_MATCH_TOLERANCE
                for r in retrieved
            )

        low_confidence = not retrieved or retrieved[0].similarity < low_confidence_threshold

        results.append(
            GoldenQuestionResult(
                question=golden_question,
                retrieved=retrieved,
                page_match=page_match,
                low_confidence=low_confidence,
            )
        )

    return results


def summarize_golden_qa_results(results: list[GoldenQuestionResult]) -> GoldenQASummary:
    def accumulate(summary: TypeSummary, result: GoldenQuestionResult) -> TypeSummary:
        return TypeSummary(
            total=summary.total + 1,
            page_match_true=summary.page_match_true + (1 if result.page_match is True else 0),
            page_match_false=summary.page_match_false + (1 if result.page_match is False else 0),
            page_match_unknown=summary.page_match_unknown + (1 if result.page_match is None else 0),
            low_confidence_true=summary.low_confidence_true + (1 if result.low_confidence else 0),
            low_confidence_false=summary.low_confidence_false + (0 if result.low_confidence else 1),
        )

    overall = TypeSummary()
    by_type: dict[str, TypeSummary] = {}

    for result in results:
        overall = accumulate(overall, result)
        by_type[result.question.type] = accumulate(by_type.get(result.question.type, TypeSummary()), result)

    return GoldenQASummary(overall=overall, by_type=by_type)
