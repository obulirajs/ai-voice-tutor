# Project Tutor

Voice-interactive AI tutor. Before doing any work in this repo, read these
in order:

1. `docs/overview.md` — why this exists, for whom.
2. `docs/architecture.md` — the module map and data flow.
3. `docs/technical-design.md` — stack, conventions, guardrails, the decided
   UI direction, and the hardware-driven provider default.

## Non-negotiable conventions

- **Modular monolith, not microservices.** One deployable app, but every
  module (`voice`, `models`, `ingestion`, `knowledge`, `orchestration`,
  `observability`, `storage`, `api`) exposes a clean interface via its
  `__init__.py`. Other modules depend on that interface, never on internals.
- **Dependency inversion for providers.** Orchestration and API code depend
  on a `Protocol`/ABC (e.g. the Model Provider interface), never on a
  concrete adapter (Ollama, Anthropic) directly. Adding a new provider means
  writing a new adapter, not touching the code that uses it.
- **No LangChain/LangGraph, no message queues, no multi-agent framework** at
  this stage — deliberate, see technical-design.md. Don't introduce one
  without flagging it first.
- **Default provider is the Anthropic API for live conversation** (see the
  hardware assessment in technical-design.md) — Ollama stays wired in for
  offline/dev use via the same interface, but isn't the default active
  provider.
- **Grounding guardrail:** anything that answers a tutoring question must
  answer only from retrieved context and say explicitly when something
  isn't covered, never fill gaps from general knowledge.
- **Tests alongside code, not after.** Every module's tests live in
  `backend/tests/test_<module>.py`, written as that module is built. Run
  `pytest`, `ruff check`, and `mypy` before considering a module done.
- **Type hints throughout** (mypy) on the backend; **TypeScript** on the
  frontend — contracts should be checked, not just documented.
- **Config over hardcoding.** Model choice, provider, theme, and role all
  come from config/env/user preference, never literals in code.
- **Two roles, locked for v1: `student` and `parent`.** They see different
  data (cost/usage detail especially) — implement as an extensible role
  field, not a hardcoded two-way branch. Role checks happen on the backend
  (what the API returns), not just hidden in the frontend.

## Repo layout

```
backend/   FastAPI app — see backend/README.md for setup
frontend/  React + Vite + TypeScript — see frontend/README.md
docs/      the three files above
design/    UI direction mockups (published design canvas, see technical-design.md)
```

When in doubt about a design decision, check `docs/technical-design.md`
first — most cross-cutting questions (patterns to use, what's deliberately
skipped, guardrails, validation checkpoints) are already answered there.
