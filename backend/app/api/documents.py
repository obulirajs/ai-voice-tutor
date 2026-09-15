from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion import ingest_pdf
from app.knowledge import (
    GoldenQASummary,
    GoldenQuestion,
    GoldenQuestionResult,
    TypeSummary,
    VectorStore,
    get_vector_store,
    run_golden_qa_check,
    summarize_golden_qa_results,
)
from app.models import (
    EmbeddingProvider,
    VisionProvider,
    get_embedding_provider,
    get_vision_provider,
)
from app.storage import (
    delete_document,
    get_db,
    get_document,
    get_or_create_subject,
    get_subject_by_name,
)

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


class ConsistencyCheckFailureResponse(BaseModel):
    chunk_index: int
    page_number: int | None


class ConsistencyCheckResponse(BaseModel):
    sampled: int
    passed: int
    failed: list[ConsistencyCheckFailureResponse]


class DocumentIngestResponse(BaseModel):
    document_id: int
    page_count: int
    scanned_page_count: int
    chunk_count: int
    consistency_check: ConsistencyCheckResponse


@router.post("/subjects/{subject}/documents", response_model=DocumentIngestResponse)
async def upload_document(
    subject: str,
    file: UploadFile = File(...),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    vision_provider: VisionProvider = Depends(get_vision_provider),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> DocumentIngestResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    pdf_bytes = await file.read()
    subject_row = await get_or_create_subject(db, subject)

    result = await ingest_pdf(
        db,
        embedding_provider,
        vision_provider,
        vector_store,
        subject_id=subject_row.id,
        filename=file.filename or "upload.pdf",
        pdf_bytes=pdf_bytes,
    )

    return DocumentIngestResponse(
        document_id=result.document_id,
        page_count=result.page_count,
        scanned_page_count=result.scanned_page_count,
        chunk_count=result.chunk_count,
        consistency_check=ConsistencyCheckResponse(
            sampled=result.consistency_check.sampled,
            passed=result.consistency_check.passed,
            failed=[
                ConsistencyCheckFailureResponse(chunk_index=f.chunk_index, page_number=f.page_number)
                for f in result.consistency_check.failed
            ],
        ),
    )


@router.delete("/subjects/{subject}/documents/{document_id}", status_code=204)
async def delete_document_endpoint(
    subject: str,
    document_id: int,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> None:
    """Deletes a document's row and its stored chunks.

    Intended for re-ingesting a document after an ingestion-pipeline fix
    (e.g. a corrected scanned-page heuristic): delete, then re-upload the
    same file via POST, so old chunks don't linger alongside corrected ones.
    """
    subject_row = await get_subject_by_name(db, subject)
    document = await get_document(db, document_id)
    if subject_row is None or document is None or document.subject_id != subject_row.id:
        raise HTTPException(status_code=404, detail=f"No document {document_id} for subject {subject!r}")

    await vector_store.delete_document_chunks(document.collection, document.id)
    await delete_document(db, document.id)


class GoldenQuestionRequest(BaseModel):
    question: str
    expected_answer: str
    source: str
    type: str


class SanityCheckRequest(BaseModel):
    questions: list[GoldenQuestionRequest]


class RetrievedChunkResponse(BaseModel):
    page_number: int | None
    snippet: str
    similarity: float


class GoldenQuestionResultResponse(BaseModel):
    question: str
    expected_answer: str
    source: str
    type: str
    retrieved: list[RetrievedChunkResponse]
    page_match: bool | None
    low_confidence: bool


class TypeSummaryResponse(BaseModel):
    total: int
    page_match_true: int
    page_match_false: int
    page_match_unknown: int
    low_confidence_true: int
    low_confidence_false: int


class GoldenQASummaryResponse(BaseModel):
    overall: TypeSummaryResponse
    by_type: dict[str, TypeSummaryResponse]


class SanityCheckResponse(BaseModel):
    document_id: int
    results: list[GoldenQuestionResultResponse]
    summary: GoldenQASummaryResponse


def _to_type_summary_response(summary: TypeSummary) -> TypeSummaryResponse:
    return TypeSummaryResponse(
        total=summary.total,
        page_match_true=summary.page_match_true,
        page_match_false=summary.page_match_false,
        page_match_unknown=summary.page_match_unknown,
        low_confidence_true=summary.low_confidence_true,
        low_confidence_false=summary.low_confidence_false,
    )


def _to_summary_response(summary: GoldenQASummary) -> GoldenQASummaryResponse:
    return GoldenQASummaryResponse(
        overall=_to_type_summary_response(summary.overall),
        by_type={type_name: _to_type_summary_response(s) for type_name, s in summary.by_type.items()},
    )


def _to_result_response(result: GoldenQuestionResult) -> GoldenQuestionResultResponse:
    return GoldenQuestionResultResponse(
        question=result.question.question,
        expected_answer=result.question.expected_answer,
        source=result.question.source,
        type=result.question.type,
        retrieved=[
            RetrievedChunkResponse(page_number=r.page_number, snippet=r.snippet, similarity=r.similarity)
            for r in result.retrieved
        ],
        page_match=result.page_match,
        low_confidence=result.low_confidence,
    )


@router.post(
    "/subjects/{subject}/documents/{document_id}/sanity-check",
    response_model=SanityCheckResponse,
)
async def check_document_sanity(
    subject: str,
    document_id: int,
    request: SanityCheckRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> SanityCheckResponse:
    """Read-only retrieval-quality check against an already-ingested document.

    Never touches ingest_pdf() or re-runs any part of ingestion — this only
    calls VectorStore.search() against whatever was already stored.
    """
    subject_row = await get_subject_by_name(db, subject)
    document = await get_document(db, document_id)
    if subject_row is None or document is None or document.subject_id != subject_row.id:
        raise HTTPException(status_code=404, detail=f"No document {document_id} for subject {subject!r}")

    golden_questions = [
        GoldenQuestion(question=q.question, expected_answer=q.expected_answer, source=q.source, type=q.type)
        for q in request.questions
    ]

    results = await run_golden_qa_check(vector_store, embedding_provider, document.collection, golden_questions)
    summary = summarize_golden_qa_results(results)

    return SanityCheckResponse(
        document_id=document.id,
        results=[_to_result_response(r) for r in results],
        summary=_to_summary_response(summary),
    )
