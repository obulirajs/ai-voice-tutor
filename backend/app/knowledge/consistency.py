from __future__ import annotations

from dataclasses import dataclass, field

from .base import Chunk, VectorStore

_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class ConsistencyFailure:
    chunk_index: int
    page_number: int | None


@dataclass(frozen=True)
class ConsistencyCheckResult:
    sampled: int
    passed: int
    failed: list[ConsistencyFailure] = field(default_factory=list)


async def check_ingestion_consistency(
    vector_store: VectorStore,
    collection: str,
    chunks: list[Chunk],
    high_risk: list[bool],
    *,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    top_k: int = _DEFAULT_TOP_K,
) -> ConsistencyCheckResult:
    """Automatic, zero-input consistency check run at the end of every ingestion.

    Samples a small number of the chunks just embedded and stored (weighted
    toward high-risk chunks, e.g. OCR/vision-sourced ones, since that's the
    highest-risk content), and for each confirms it retrieves itself — or a
    near-identical neighbor, if content is genuinely duplicated — as a top
    result. Pure internal consistency: no LLM call, no golden answers
    needed. A failure means something broke between embedding and storage,
    or the OCR text was too garbled to embed meaningfully.

    `chunks` and `high_risk` must be the same length, aligned by index —
    `high_risk[i]` marks whether `chunks[i]` came from a scanned/OCR page.
    """
    high_risk_indices = [i for i, flagged in enumerate(high_risk) if flagged]
    other_indices = [i for i, flagged in enumerate(high_risk) if not flagged]
    sample_indices = (high_risk_indices + other_indices)[:sample_size]

    failed: list[ConsistencyFailure] = []
    for index in sample_indices:
        chunk = chunks[index]
        hits = await vector_store.search(collection, chunk.embedding, top_k=top_k)
        if not any(hit.text == chunk.text for hit in hits):
            failed.append(ConsistencyFailure(chunk_index=index, page_number=chunk.page_number))

    return ConsistencyCheckResult(
        sampled=len(sample_indices),
        passed=len(sample_indices) - len(failed),
        failed=failed,
    )
