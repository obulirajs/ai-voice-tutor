# Project Tutor — Backend

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

## Test
    pytest
