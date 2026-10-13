# Project Tutor — Backend

A voice-interactive AI tutor for school students. Speak a question, hear an answer grounded in your own textbook, with the source page shown on screen. Built first for CBSE Grade 10 French.

> Personal project. No textbooks or student data are included in this repo. Bring your own PDF.

## What it does
- **Voice in, voice out:** local speech recognition (faster-whisper) and speech synthesis (Piper / edge-tts).
- **Grounded answers:** retrieves passages from an uploaded textbook and answers from them, with page citations. Off-topic questions get an explicit "not covered" reply.
- **Visual companion:** shows tables, formulas, diagrams and the source textbook page next to the conversation.
- **Parent controls:** role-based views (student / parent), a cost and usage dashboard for parents, and an off-by-default switch for live web references.

## Architecture
A modular monolith: one deployable app, with each module (`voice`, `models`, `ingestion`, `knowledge`, `orchestration`, `observability`, `storage`, `api`) exposed through a clean interface. Model, embedding and vector-store providers are swappable adapters (Anthropic API by default, Ollama for offline use).

Deliberately not used: LangChain/LangGraph, message queues, multi-agent frameworks. See `../docs/` for the overview, architecture, technical design and development plan.

## Tech stack
Python, FastAPI, SQLite, sqlite-vec, faster-whisper, Piper, Anthropic SDK, Ollama · React, Vite, TypeScript (frontend, see `../frontend/README.md`)

## Quality checks
    pytest && ruff check . && mypy .

400+ automated tests, linted and type-checked.

## Setup
    python -m venv .venv
    # Windows:  .venv\Scripts\activate
    # macOS/Linux/WSL/Git Bash:  source .venv/bin/activate
    pip install -r requirements-dev.txt
    cp .env.example .env   # then fill in ANTHROPIC_API_KEY if using cloud mode

Ingestion's default vision/OCR provider (`VISION_PROVIDER=tesseract`) needs
the `tesseract-ocr` system binary too -- `pip install` alone isn't enough:

    # Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
    #   Include the language packs you need (English + French, at minimum).
    #   If tesseract.exe isn't on PATH afterward, set TESSERACT_CMD in .env.
    # Linux:  sudo apt install tesseract-ocr tesseract-ocr-fra
    # macOS:  brew install tesseract

## Run
    uvicorn app.main:app --reload
    # then open http://127.0.0.1:8000/health

## Page view

The Visual Companion's page-view feature (clicking a "Page N" source badge)
renders pages from the *stored* PDF, not from the vector store -- documents
uploaded before this feature existed have no stored PDF yet. Back-fill them
once, without re-uploading, by hash-matching against
`backend/textbooks_input/`:

    cd backend && python -m scripts.import_existing_pdfs

Add `--dry-run` to preview matches without copying files. Any document the
script reports as "no match" needs a real re-upload (delete + re-upload the
same subject/file) to get page view.

## Test
    pytest
