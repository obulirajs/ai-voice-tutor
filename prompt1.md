Read docs/overview.md, docs/architecture.md, and docs/technical-design.md
before writing anything.

Implement the Model Provider module (backend/app/models/):
- A Protocol/ABC interface: generate(messages, tools) -> response
- An Ollama adapter and an Anthropic adapter implementing it
- Provider selection driven by the MODEL_PROVIDER env var (see .env.example)
- Unit tests in backend/tests/test_models.py that mock both adapters
  (no real API calls in tests)

Then add a POST /chat endpoint in backend/app/api/ that takes a text
message, sends it through the configured provider, and returns the
reply. Add a test for that endpoint too using FastAPI's TestClient.

Follow the module-boundary and dependency-inversion conventions in
technical-design.md — orchestration/api code should depend on the
interface, never import a concrete adapter directly.