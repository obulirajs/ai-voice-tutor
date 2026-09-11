# Project Tutor — Architecture (v1)

Status: draft, first pass. Reflects decisions through the feasibility discussion (see overview.md for origin/vision). Design patterns, guardrails/security, retrieval-accuracy, and code-clarity/team-scaling conventions are follow-up discussions — noted at the bottom as open threads, not yet resolved here. (See technical-design.md — those threads are resolved there.)

## Guiding shape: modular monolith

Not microservices — one deployable app, but internally split into modules with explicit interfaces (Python `Protocol`/ABC contracts) so any module can be swapped, tested, or later extracted into its own service without touching the others. This is the "harness" — the LLM is one interchangeable piece; everything below is the scaffolding around it.

## Modules

**1. Voice I/O**
- ASR wrapper: default local (`faster-whisper`), swappable for a cloud ASR provider via the same interface.
- TTS wrapper: default local (Piper), swappable for a cloud TTS provider. Needs a formula/notation → spoken-form preprocessor (e.g. "x^2" → "x squared") since raw LaTeX/notation can't be read aloud as-is.
- Both are pure `audio in/out ↔ text` contracts — orchestration never talks to Whisper/Piper directly, only to this interface.

**2. Model Provider (LLM abstraction)**
- Common interface (`generate(messages, tools) -> response`) implemented by an Ollama adapter and an Anthropic API adapter.
- Runtime-selectable: a config value (dev) → a settings screen (production) lets an advanced user pick provider + model per session.
- Same pattern reused for the *embedding* model (local via Ollama vs. API), since retrieval needs its own model choice independent of the chat model.

**3. Ingestion & Knowledge Base**
- Upload handler: accepts PDF (and later other formats), detects scanned-vs-text-layer PDFs (heuristic: extracted-text density vs. page image density), and warns the user before proceeding if it looks scanned.
- OCR/vision pipeline: text-layer PDFs go through direct extraction; scanned pages and any diagrams/formulae go through a vision-capable LLM pass that produces (a) a clean text/description for indexing and (b) LaTeX-ish transcription for formulae, plus a stored reference to the original image for the visual companion to show later.
- Chunking + embedding + a lightweight local vector store (e.g. `sqlite-vec` or Chroma — no need for a server-based vector DB at this scale).
- Each subject gets its own namespace/collection. Ad-hoc uploads (e.g. an assignment) get a session-scoped namespace that's layered into retrieval alongside the active subject's namespace, then discarded or archived after the session.

**4. Orchestration (the core harness)**
- Owns session state: active subject, ad-hoc-doc context, conversation history.
- Parses incoming text for control commands ("switch to French") vs. tutoring content, before treating input as a question.
- Runs retrieval (subject KB + any ad-hoc doc) → builds a grounded prompt → calls the Model Provider → post-processes the response into (a) spoken text for TTS and (b) an optional "visual directive" (show this image / render this formula / highlight this passage) for the companion UI.
- This is also where grounding/guardrail checks belong.

**5. Visual Companion**
- A small UI surface, not a full app screen — driven entirely by directives from Orchestration ("show diagram X", "render formula Y", "display text Z"). It doesn't call the LLM or retrieval itself; it's a dumb renderer with one contract in.

**6. App Shell / Frontend**
- Mic capture, audio playback, the visual companion panel, subject switcher, upload UI, and the model-selection settings screen. Talks to the backend over one API; no business logic lives here.

**7. Storage**
- SQLite for metadata (subjects, sessions, conversation history, uploaded-file records) + filesystem for the actual documents/images and the vector store.

## Data flow (happy path)

Mic → ASR → Orchestration (control-command check) → Retrieval (active subject + ad-hoc scope) → Model Provider → response split into spoken text + optional visual directive → TTS → speaker, and Visual Companion → screen, in parallel.

## Suggested stack

- Backend: Python + FastAPI (matches your existing stack).
- Frontend: React/Vite (matches your Generic ChatUI project), with the visual companion as one panel alongside the voice interaction surface.
- No heavy agent/orchestration framework (LangChain etc.) for v1 — a hand-rolled provider/retrieval interface is simpler to debug and keeps you in full control. Revisit only if the hand-rolled version starts creaking (see the migration-path note in technical-design.md).
- Local models via Ollama; cloud via the Anthropic API — both behind the Model Provider interface from day one, even though v1 dev keeps it simple/hybrid rather than fully configurable.

## See also

`technical-design.md` in this same folder — tech stack & libraries, repo layout, software principles, AI design patterns used/skipped, the multi-agent migration path, observability & cost tracking, guardrails & security, retrieval-accuracy approach, code-clarity conventions, validation checkpoints, commercialization considerations, theming, and the decided UI direction (including the student/parent role split). Read both before starting a module.
