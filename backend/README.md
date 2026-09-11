# Project Tutor — Backend

## Setup
    python -m venv .venv
    # Windows:  .venv\Scripts\activate
    # macOS/Linux/WSL/Git Bash:  source .venv/bin/activate
    pip install -r requirements-dev.txt
    cp .env.example .env   # then fill in ANTHROPIC_API_KEY if using cloud mode

## Run
    uvicorn app.main:app --reload
    # then open http://127.0.0.1:8000/health

## Test
    pytest
