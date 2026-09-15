# Project Tutor — Architecture (v1)

Status: draft, first pass. Reflects decisions through the feasibility discussion (see overview.md for origin/vision). Design patterns, guardrails/security, retrieval-accuracy, and code-clarity/team-scaling conventions are follow-up discussions — noted at the bottom as open threads, not yet resolved here.

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
- Chunking + embedding + a lightweight local vector store, behind a `VectorStore` interface (Protocol/ABC) — the same dependency-inversion pattern as Model Provider — so swapping the backend later is a new adapter, not a rewrite. **sqlite-vec is the chosen adapter** (a single lightweight wheel vs. Chroma's much larger dependency tree — still no server-based vector DB at this scale; see technical-design.md's "Vector store choice" for why LEANN was considered and rejected).
- Each subject gets its own namespace/collection. Ad-hoc uploads (e.g. an assignment) get a session-scoped namespace that's layered into retrieval alongside the active subject's namespace, then discarded or archived after the session.

**4. Orchestration (the core harness)**
- Owns session state: active subject, ad-hoc-doc context, conversation history.
- Parses incoming text for control commands ("switch to French") vs. tutoring content, before treating input as a question.
- Runs retrieval (subject KB + any ad-hoc doc) → builds a grounded prompt → calls the Model Provider → post-processes the response into (a) spoken text for TTS and (b) an optional "visual directive" (show this image / render this formula / highlight this passage) for the companion UI.
- This is also where grounding/guardrail checks belong (flagged as an open thread below, but the hook lives here).
- Also owns the optional **web search tool** (see technical-design.md — "Web search (user-invoked, config-gated)"): a `web_search` tool the Model Provider can call, but only in response to an explicit user request to search the web — never triggered automatically as a fallback when the knowledge base doesn't cover a question. Gated by a config flag the parent controls (see Storage below); off by default. Responses sourced this way are tagged so they render as visually distinct from grounded-in-textbook answers.

**5. Visual Companion**
- A small UI surface, not a full app screen — driven entirely by directives from Orchestration ("show diagram X", "render formula Y", "display text Z", "show as web result"). It doesn't call the LLM or retrieval itself; it's a dumb renderer with one contract in.

**6. App Shell / Frontend**
- Mic capture, audio playback, the visual companion panel, subject switcher, upload UI, and the model-selection settings screen. The parent-only settings screen also carries the web-search on/off toggle. Talks to the backend over one API; no business logic lives here.

**7. Storage**
- SQLite for metadata (subjects, sessions, conversation history, uploaded-file records) + filesystem for the actual documents/images and the vector store.
- User preferences table also holds the web-search enable/disable flag (parent-set, read by Orchestration before honoring any web-search request).

## Data flow (happy path)

Mic → ASR → Orchestration (control-command check) → Retrieval (active subject + ad-hoc scope) → Model Provider → response split into spoken text + optional visual directive → TTS → speaker, and Visual Companion → screen, in parallel.

A user-invoked web search is a variant branch off the same loop: Orchestration recognizes the explicit request, checks the config flag, and — only if enabled — lets the Model Provider call the `web_search` tool instead of (or alongside) KB retrieval, with the result tagged as web-sourced through to TTS/Visual Companion.

## Suggested stack

- Backend: Python + FastAPI (matches your existing stack).
- Frontend: React/Vite (matches your Generic ChatUI project), with the visual companion as one panel alongside the voice interaction surface.
- No heavy agent/orchestration framework (LangChain etc.) for v1 — a hand-rolled provider/retrieval interface is simpler to debug and keeps you in full control, per your point about wanting to debug this yourself. Revisit only if the hand-rolled version starts creaking.
- Local models via Ollama; cloud via the Anthropic API — both behind the Model Provider interface from day one, even though v1 dev keeps it simple/hybrid rather than fully configurable.

## Open threads (next discussions, not yet resolved)

1. Applying broader software-design best practices to this build.
2. Which AI/agent design patterns actually earn their place here, without over-engineering.
3. Guardrails and security for the AI layer.
4. Ensuring accuracy of retrieved data (grounding/verification, referenced above but not designed yet).
5. Code clarity/modularity for solo debugging now, and team-scale extension later.
