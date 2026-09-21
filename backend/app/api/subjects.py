from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import get_db, list_subjects

router = APIRouter()


class SubjectResponse(BaseModel):
    id: int
    name: str
    document_count: int
    created_at: datetime


@router.get("/subjects", response_model=list[SubjectResponse])
async def get_subjects(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[SubjectResponse]:
    rows = await list_subjects(db)
    return [
        SubjectResponse(id=subject.id, name=subject.name, document_count=count, created_at=subject.created_at)
        for subject, count in rows
    ]
