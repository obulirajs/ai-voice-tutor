"""Role gate — a lightweight student/parent distinction, NOT authentication.

technical-design.md's guardrail note: "the backend needs a basic role
concept gating what the API returns and allows, not just what the frontend
chooses to display" -- but also that full auth is unnecessary "while this
runs only on your own machine for your own household." get_role() reads
the X-Tutor-Role request header and is injected via Depends() on any
endpoint whose response should differ by role (see api/parent.py).
"""

from __future__ import annotations

from enum import StrEnum

from fastapi import Header

__all__ = ["Role", "get_role"]


class Role(StrEnum):
    STUDENT = "student"
    PARENT = "parent"


def get_role(x_tutor_role: str = Header(default="student")) -> Role:
    """Missing or unrecognized values fall back to the least-privileged
    role (student) rather than failing the request -- there's no login to
    reject, just a header an unaware/older client simply won't send.
    """
    try:
        return Role(x_tutor_role.lower())
    except ValueError:
        return Role.STUDENT
