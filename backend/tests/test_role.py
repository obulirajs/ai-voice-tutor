from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.role import Role, get_role
from app.main import app


def test_get_role_defaults_to_student_when_header_absent() -> None:
    # get_role()'s default is a FastAPI Header() sentinel, only resolved to
    # the string "student" by the dependency-injection machinery -- calling
    # the function directly means supplying that resolved default ourselves.
    assert get_role(x_tutor_role="student") == Role.STUDENT


def test_get_role_recognizes_parent_header() -> None:
    assert get_role(x_tutor_role="parent") == Role.PARENT


def test_get_role_falls_back_to_student_for_unrecognized_value() -> None:
    assert get_role(x_tutor_role="admin") == Role.STUDENT


def test_get_role_is_case_insensitive() -> None:
    assert get_role(x_tutor_role="PARENT") == Role.PARENT


def test_parent_settings_endpoint_403s_without_role_header() -> None:
    client = TestClient(app)

    response = client.get("/api/parent/settings")

    assert response.status_code == 403


def test_parent_settings_endpoint_403s_for_student_role() -> None:
    client = TestClient(app)

    response = client.get("/api/parent/settings", headers={"X-Tutor-Role": "student"})

    assert response.status_code == 403


def test_parent_settings_endpoint_200s_for_parent_role() -> None:
    client = TestClient(app)

    response = client.get("/api/parent/settings", headers={"X-Tutor-Role": "parent"})

    assert response.status_code == 200
    assert response.json() == {"web_search_enabled": False}


def test_parent_settings_endpoint_falls_back_to_student_for_unrecognized_role() -> None:
    client = TestClient(app)

    response = client.get("/api/parent/settings", headers={"X-Tutor-Role": "teacher"})

    assert response.status_code == 403
