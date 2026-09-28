"""Shared fixtures for the whole backend/tests/ suite -- pytest auto-discovers
this file, no explicit import needed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points app.storage.files' DATA_DIR at a fresh per-test tmp dir.

    Without this, any test that uploads a document (api/documents.py
    persists the PDF to disk on every successful ingest -- see
    save_document_pdf()) would write real files into the developer's actual
    backend/data/ directory, and different tests' in-memory DBs reusing the
    same document ids (each starts fresh at 1) would overwrite each other's
    files on disk.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
