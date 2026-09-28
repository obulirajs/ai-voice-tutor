"""POST /api/diagram -- the Visual Companion's "Draw diagram" button.

Turns one assistant reply into a Mermaid flowchart via a small model (see
app.orchestration.diagrams.generate_diagram()), independent of the main
grounded chat loop in chat.py. Both roles may call it (no role gate --
drawing a diagram isn't cost/usage-sensitive data the way the parent-only
views are).

Passage lookup: the task this endpoint backs asked for the cited passages'
text (by document_id/page_number) to be included as extra grounding context.
There's no repository/VectorStore lookup for "the chunk(s) at this
(document_id, page_number)" today -- VectorStore only supports similarity
search, not a direct fetch -- so this passes only `message_text`, which is
itself already grounded in the textbook (chat.py's guardrail guarantees
that), as the diagram-model instructions call out is an acceptable
fallback. Revisit if diagrams from short replies prove to need more context
than the reply text alone provides.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DIAGRAM_DEFAULTS
from app.models import ModelProvider, estimate_cost_usd, get_diagram_model_provider
from app.orchestration import DiagramResult, generate_diagram
from app.storage import get_db, get_session, log_usage_event

router = APIRouter()


class DiagramSourceRequest(BaseModel):
    document_id: int
    page_number: int | None = None


class DiagramRequest(BaseModel):
    session_id: str | None = None
    message_text: str
    sources: list[DiagramSourceRequest] = []
    previous_error: str | None = None


class DiagramResponse(BaseModel):
    mermaid: str | None
    reason: str | None = None


# In-memory LRU cache, keyed by sha256(message_text + previous_error) --
# re-opening a message (or re-rendering after a client-side mermaid.parse()
# failure with the same previous_error) shouldn't hit the model again.
# Module-level like documents.py's _ingesting_subjects: fine for this app's
# single-process, single-household scale (see technical-design.md's "Auth"
# guardrail note on the same theme).
_cache: OrderedDict[str, DiagramResult] = OrderedDict()


def _cache_key(message_text: str, previous_error: str | None) -> str:
    raw = f"{message_text}\x00{previous_error or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> DiagramResult | None:
    result = _cache.get(key)
    if result is not None:
        _cache.move_to_end(key)
    return result


def _cache_set(key: str, result: DiagramResult) -> None:
    _cache[key] = result
    _cache.move_to_end(key)
    if len(_cache) > DIAGRAM_DEFAULTS.cache_size:
        _cache.popitem(last=False)


@router.post("/diagram", response_model=DiagramResponse)
async def draw_diagram(
    request: DiagramRequest,
    provider: ModelProvider | None = Depends(get_diagram_model_provider),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DiagramResponse:
    cache_key = _cache_key(request.message_text, request.previous_error)
    cached = _cache_get(cache_key)
    if cached is not None:
        return DiagramResponse(mermaid=cached.mermaid, reason=cached.reason)

    result = await generate_diagram(request.message_text, "", provider, request.previous_error)

    # No provider call was made for a disabled feature -- nothing to log.
    if result.reason != "disabled":
        subject_id = None
        if request.session_id is not None:
            session = await get_session(db, request.session_id)
            subject_id = session.subject_id if session is not None else None

        cost_usd = (
            estimate_cost_usd(result.model_name, result.usage.input_tokens, result.usage.output_tokens)
            if result.model_name is not None
            else 0.0
        )
        await log_usage_event(
            db,
            session_id=request.session_id,
            subject_id=subject_id,
            provider=result.provider_name or "unknown",
            model=result.model_name or "unknown",
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=cost_usd,
            latency_ms=result.latency_ms,
            event_type="diagram",
        )

    _cache_set(cache_key, result)

    return DiagramResponse(mermaid=result.mermaid, reason=result.reason)
