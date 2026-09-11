#!/usr/bin/env bash
# Project Tutor — repo scaffold
# Run this from inside the folder where you want the project to live
# (e.g. C:\Raj\Virtual_Tutor), using Git Bash, WSL, or any bash shell.
#
#   bash scaffold-project-tutor.sh
#
set -e

echo "Scaffolding Project Tutor..."

# --- Backend ---
mkdir -p backend/app/{voice,models,ingestion,knowledge,orchestration,observability,storage,api}
mkdir -p backend/tests

touch backend/app/__init__.py
touch backend/tests/__init__.py
for m in voice models ingestion knowledge orchestration observability storage api; do
  touch "backend/app/$m/__init__.py"
  touch "backend/tests/test_${m}.py"
done

cat > backend/requirements.txt << 'EOF'
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
sqlalchemy>=2.0
aiosqlite>=0.20
python-multipart>=0.0.9
ollama>=0.3
anthropic>=0.34
EOF

cat > backend/requirements-dev.txt << 'EOF'
-r requirements.txt
pytest>=8.0
ruff>=0.6
mypy>=1.11
EOF

cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI

app = FastAPI(title="Project Tutor")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
EOF

cat > backend/.env.example << 'EOF'
# Copy to .env and fill in. Never commit the real .env.
ANTHROPIC_API_KEY=
MODEL_PROVIDER=ollama   # ollama | anthropic
OLLAMA_MODEL=llama3.1
ANTHROPIC_MODEL=claude-sonnet-5
EOF

cat > backend/README.md << 'EOF'
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
EOF

# --- Frontend ---
npm create vite@latest frontend -- --template react-ts

cat > frontend-README-append.txt << 'EOF'

## Project Tutor frontend notes
Run `npm install` then `npm run dev` inside frontend/.
Planned structure: src/components (VisualCompanion, SubjectNav, UploadPanel, SettingsPanel),
src/theme (design tokens + presets), src/views (StudentView, ParentView), src/voice, src/api.
EOF
cat frontend-README-append.txt >> frontend/README.md
rm frontend-README-append.txt

# --- Root ---
cat > README.md << 'EOF'
# Project Tutor

Voice-interactive AI tutor. Two independent projects: `backend/` (FastAPI) and `frontend/` (React + Vite + TS).
See the project docs (architecture.md, technical-design.md) for the full design.

## Quick start
    cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
    uvicorn app.main:app --reload

    # in a second terminal
    cd frontend && npm install && npm run dev
EOF

cat > .gitignore << 'EOF'
backend/.venv/
backend/.env
backend/**/__pycache__/
backend/*.db
frontend/node_modules/
frontend/dist/
.DS_Store
EOF

echo ""
echo "Scaffold complete."
echo ""
echo "Next steps:"
echo "  1) cd backend && python -m venv .venv && source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)"
echo "  2) pip install -r requirements-dev.txt"
echo "  3) cp .env.example .env   # fill in ANTHROPIC_API_KEY if you'll use cloud mode"
echo "  4) uvicorn app.main:app --reload   -> check http://127.0.0.1:8000/health"
echo "  5) In another terminal: cd frontend && npm install && npm run dev"
