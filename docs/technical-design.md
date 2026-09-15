# Project Tutor — Technical Design (v1)

Companion to architecture.md. Covers stack, repo layout, and the five cross-cutting concerns raised after the architecture pass: software principles, AI design patterns, guardrails/security, retrieval accuracy, and code clarity for solo debugging now / team scaling later. Also covers observability/cost tracking, the multi-agent migration path, human validation checkpoints, commercialization considerations, end-user theming, the chosen UI direction, a hardware assessment of Raj's dev machine, and the user-invoked web search enhancement.

## Dev machine hardware assessment (informs the local/cloud default)

Checked before starting the build. Raj's laptop: Intel Core i7-8550U (4 cores / 8 threads, 2018-era low-power mobile CPU), 16 GB RAM, NVIDIA GeForce MX150 (entry-level laptop GPU, effectively ~2 GB VRAM — too small/weak for meaningful LLM acceleration), ~250 GB free disk.

**Implication:** local LLM inference via Ollama on this machine would run essentially CPU-only, likely too slow (single-digit tokens/sec range) to feel conversational once ASR and TTS latency stack on top. So for v1 development on this machine:
- **Default the Model Provider to the Anthropic API** for the actual tutoring/chat calls in the voice-loop path — that's where latency is felt directly. Confirmed in practice: the Anthropic API responds meaningfully faster than a local `llama3.2:1b` run on this machine.
- **Keep ASR (faster-whisper) and TTS (Piper) local** — both are lightweight enough to run comfortably on this CPU.
- **Ollama stays wired into the provider abstraction** (it's still useful for offline dev/testing, and the seam is free since the adapter pattern was built for exactly this), but isn't the default active provider for live conversation on this hardware. Revisit if Raj builds/uses a machine with a real GPU later — the provider config, not the code, is what would change.
- Embeddings for retrieval: worth benchmarking once ingestion exists — likely fine locally for the (probably small) size of a single subject's textbook, but keep an eye on it; fall back to the Anthropic API's embeddings if local indexing feels slow.

## Tech stack & libraries

**Backend (Python)**
- FastAPI + Uvicorn — API and WebSocket server (WebSocket matters for the voice loop — need duplex, low-latency streaming, not request/response polling)
- Pydantic — request/response and config validation
- SQLAlchemy + SQLite (aiosqlite) — metadata: subjects, sessions, conversation history, uploaded-file records
- faster-whisper — local ASR
- Piper (piper-tts) — local TTS
- Ollama Python client — local LLM + local embeddings (dev/offline use; not the default live-conversation provider on this hardware — see above)
- Anthropic Python SDK — default cloud LLM for the conversation path, vision calls for OCR/diagram/formula ingestion, and (optionally) the user-invoked web search tool call
- PyMuPDF (fitz) or pdfplumber — PDF text extraction + scanned-page detection
- sqlite-vec — local vector store, one collection per subject, behind a `VectorStore` interface (see "Vector store choice" below)
- python-multipart — file upload handling in FastAPI
- pytest, ruff, mypy — tests, linting, type checking (these are your "sensors" from the harness-engineering discussion)

**Frontend (React + Vite)**
- React + TypeScript — TypeScript specifically because it makes the frontend↔backend contract explicit and catches mismatches at compile time, which matters both for your own debugging and for a future team
- Vite — dev server/build
- MediaRecorder / Web Audio API — mic capture, playback
- KaTeX (or react-katex) — rendering formulae in the visual companion
- Zustand (or plain React context) — lightweight state, deliberately not Redux — avoids ceremony a single-user app doesn't need
- Native WebSocket client — talks to the backend's WS endpoint
- CSS custom properties (design tokens) — backbone of the theming system, see below

**Not using, deliberately:** LangChain/LangGraph or similar agent frameworks, and no message-queue/microservices infrastructure. Both would add real complexity (harder to trace, more moving parts to debug) for capability this app doesn't need at v1 scale. Revisit only if a specific limitation is hit — see the migration-path note below for how costly that revisit would actually be.

## Repo layout

Two independent projects in one repo — separate dependency manifests, separately runnable, but co-located so a solo dev isn't juggling two repos day to day. Trivial to split into separate repos later if a team wants that.

```
project-tutor/
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── voice/            # ASR + TTS wrappers (interface + adapters)
│   │   ├── models/           # LLM + embedding provider abstraction (interface + Ollama/Anthropic adapters)
│   │   ├── ingestion/        # upload handling, scan detection, OCR/vision pipeline, chunking
│   │   ├── knowledge/        # VectorStore interface + sqlite-vec adapter, retrieval, per-subject namespaces
│   │   ├── orchestration/    # core harness: session state, command routing, grounding, guardrails, web-search tool gating
│   │   ├── observability/    # usage/cost logging, request tracing
│   │   ├── storage/          # SQLite models/repositories (incl. user preferences: model choice, theme, role, web-search toggle)
│   │   └── api/              # FastAPI routers + WebSocket endpoints
│   └── tests/                # mirrors app/ structure, one test module per app module
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/       # VisualCompanion, SubjectNav, UploadPanel, SettingsPanel
│   │   ├── theme/             # design tokens + theme presets (see Theming section)
│   │   ├── views/              # StudentView, ParentView (role-scoped screens, see UI Direction section)
│   │   ├── voice/             # mic capture, audio playback
│   │   ├── api/                # backend client (typed, generated or hand-written from backend schemas)
│   │   └── App.tsx
│   └── tests/
└── README.md
```

Each module folder exposes its public surface via `__init__.py` (backend) or an `index.ts` (frontend) — everything else inside is implementation detail other modules shouldn't reach into directly. That's what keeps modules swappable and keeps a future teammate from having to read the whole codebase to change one part.

## Software principles applied

- Dependency inversion at every module boundary: orchestration depends on the *interface* (Protocol/ABC) for voice/models/knowledge, never on a concrete adapter — swapping Ollama for the Anthropic API, Piper for a cloud TTS, or sqlite-vec for another vector store, means writing a new adapter, not touching orchestration or ingestion.
- Config over hardcoding: model choice, provider, vector-store backend, file paths, thresholds, UI theme, role-based visibility, and the web-search toggle all come from config/user preference, not literals in code.
- Structured logging per module (not prints) — essential once the voice pipeline has five stages; when something's wrong you need to know which stage.
- Tests as sensors, written alongside each module rather than bolted on at the end — this is the harness-engineering point applied concretely: guides (the module contracts above) plus sensors (tests/lint/types) are what let you trust output from Claude Code as it builds each piece.

## AI design patterns used — and what's deliberately skipped

**Used:**
- Adapter/Provider pattern for LLM and embedding models, and for the vector store (see "Vector store choice" below).
- RAG (retrieval-augmented generation) as the core grounding mechanism.
- A lightweight intent router distinguishing control commands ("switch to French") from tutoring questions — start rule-based (keyword/phrase matching); only upgrade to an LLM-based classifier if rules prove too brittle in practice.
- Structured function/tool-calling for actions the model should trigger deliberately (e.g. "render this formula," "switch subject," "search the web") rather than trying to parse free text after the fact. Web search is deliberately modeled as one more tool in this same pattern, not a separate subsystem.

**Deliberately not used (over-engineering risk):** multi-agent frameworks, autonomous multi-step planning loops, graph-based orchestration (LangGraph-style). This app is one user, one conversation thread, one orchestrator — those patterns solve problems (many collaborating agents, complex branching plans) this system doesn't have at v1.

## Web search (user-invoked, config-gated)

An enhancement layered onto the tool-calling pattern above, decided after the core design was drafted. It intentionally sits outside the grounding guardrail's normal path rather than inside it.

- **Trigger: user-invoked only, never model-initiated as a fallback.** The `web_search` tool is only called when the user explicitly asks to search the web. If a question isn't covered by the subject's knowledge base, the system still follows the existing grounding guardrail — say so explicitly — rather than silently reaching for the web as a substitute source. This keeps "the tutor might quietly answer from an unvetted web result" out of the failure surface entirely.
- **Config-gated, parent-controlled.** A boolean preference (`web_search_enabled`), off by default, stored alongside the other user preferences (theme, provider, role) in Storage. Only the parent role can toggle it, from the Settings panel — consistent with the existing role-based access-control guardrail (backend-enforced, not just hidden in the student UI).
- **Implementation:** one more adapter behind the same tool-calling mechanism already used for "render this formula" / "switch subject" — e.g. Anthropic's built-in web search tool, or a swappable search-API adapter if a specific provider is ever needed. No new module; it's a capability of Orchestration + Model Provider, gated by the config flag before Orchestration will honor it.
- **Labeling.** Any response that used web search is tagged distinctly from a KB-grounded answer, carried through to both the spoken response (a brief verbal cue) and the Visual Companion (a visible "from the web" indicator with the source link), so it's never ambiguous which answers are backed by the CBSE syllabus material and which came from the open web.
- **Cost & observability.** Each web search call gets its own `usage_events` row (see Observability below), same as an LLM or embedding call, so the parent's cost view stays accurate once this is enabled.
- **Content safety.** Unlike the curated, parent-approved subject KB, the open web isn't vetted — this is a sharper version of the existing content guardrail (age-appropriate, on-topic responses for a minor user) and is part of why the feature defaults off and is parent-gated rather than on by default.
- **Data handling.** A web search query is one more thing leaving the device when enabled — the same "be deliberate about what leaves the device" consideration already applied to cloud LLM calls extends to this.
- **Sequencing:** not core to the v1 build order — see development-plan.md for where it lands (after the core grounded loop and role-gating are already proven).

## Vector store choice (decided in Phase 2)

- **sqlite-vec chosen over Chroma.** A single lightweight wheel vs. Chroma's much larger dependency tree, which fits the "no server-based vector DB at this scale" stance from architecture.md better. One collection per subject, matching the per-subject namespacing design.
- **Behind a `VectorStore` interface**, the same Protocol/ABC + adapter + factory pattern as Model Provider (`get_vector_store()` reading a `VECTOR_STORE` config value, default `sqlite_vec`) — so ingestion/knowledge code depends on the interface, never imports the sqlite-vec adapter directly, and a future backend swap is a new adapter rather than a rewrite.
- **LEANN considered and explicitly rejected.** LEANN's value proposition is trading storage for on-demand embedding recomputation at query time (~97% smaller index). Two reasons it doesn't fit here: (1) a single subject's textbook is a tiny corpus — storage was never the bottleneck this app needed to solve — and (2) recomputing embeddings at query time adds latency directly against the hardware-driven constraint already established in the dev-machine assessment above (weak CPU, cloud LLM as the default specifically to protect voice-loop latency). It also needs Visual Studio 2022 Build Tools on Windows, one more setup dependency on an already resource-constrained machine. Because the vector store sits behind an interface, this isn't a closed door — just not worth it at this project's actual scale; revisit only if the knowledge base genuinely grows large enough that storage becomes the real constraint, which is unlikely for a single-family, one-subject-at-a-time app.

## Migration path: single-orchestrator → multi-agent (e.g. LangChain-based)

Because Model Provider and Orchestration are already isolated behind interfaces, an eventual move to a multi-agent design (e.g. a "tutor" agent, a "grader" agent, a "content-ingestion" agent coordinating via LangChain/LangGraph) would **not** require rewriting Voice I/O, Ingestion, Knowledge Base, Visual Companion, or Storage — those modules talk to Orchestration through their own contracts, not through anything agent-framework-specific.

What *would* need rework: the internals of the Orchestration module (today a single hand-rolled loop; it would become a set of agent definitions and a coordination graph), and possibly the Model Provider adapters if LangChain's model-wrapper interface doesn't line up with the hand-rolled one (usually a thin adapter, not a redesign). Realistically: a moderate, contained rewrite of one-to-two modules, not a rewrite of the application. That containment is a direct payoff of the modular-monolith approach — it's worth treating as a validation of the design, not just a hope.

## Observability & cost tracking

- **Where it hooks in:** the Model Provider interface is the single choke point every LLM call passes through (chat completions, embeddings, vision calls for ingestion, and the web search tool call) — that's the natural place for a logging wrapper, so no module needs to know observability exists.
- **What gets logged per call:** timestamp, session id, subject, provider (Ollama/Anthropic/web-search), model or tool name, input/output token counts, computed cost (token counts × that provider's pricing, or per-query pricing for web search — $0 for local Ollama, real cost for the API and for web search), and latency.
- **Storage:** a `usage_events` table in the same SQLite database as everything else — no separate observability stack needed at this scale.
- **What this gets you:**
  - Per-request cost: read directly off one row.
  - Per-session cost: sum of that session's rows.
  - Weekly/monthly reports: a simple aggregation query grouped by week/month (and, later, by user/tenant) — no new infrastructure needed.
- **Also worth logging:** retrieval hits/misses and confidence scores per query, tied to the same request id as its cost record — lets you correlate "expensive session" with "session where retrieval kept missing." Web search calls are logged distinctly so parent-view reporting can separate "textbook-grounded" usage from "web search" usage.
- **UI exposure is role-scoped** — see UI Direction below: the student view gets a toggle-able, coarse cost indicator at most; the full breakdown (per-request, per-session, weekly/monthly, by subject) lives on the parent-only view. This matters more now that the Anthropic API is the default conversation provider (real cost, not $0 local) — cost tracking is live from day one of the voice loop, not a later add-on.

## Guardrails & security

- **Grounding guardrail:** the model is instructed to answer only from retrieved context and to say explicitly when something isn't covered by the uploaded material, rather than filling gaps from general knowledge — critical for a tutor whose answers need to match what's actually in the textbook/syllabus. Web search is a deliberate, user-invoked exception to this path (see "Web search" above), not a loophole in it — it never fires automatically when grounding comes up empty.
- **Content guardrail:** age-appropriate, on-topic responses, given the primary user is a minor. Applies with extra weight to web search results, since that content isn't pre-vetted the way the uploaded syllabus material is.
- **Upload validation:** file size/type limits, and treating uploaded content as untrusted input (no executable content, sanitized before indexing).
- **Secrets management:** API keys via env vars / a local secrets file, never committed to the repo.
- **Cost guardrail:** a simple usage cap/alert on cloud API calls (backed by the observability layer above) so a bug or runaway loop can't produce a surprise bill — higher priority now that cloud is the default path, not a fallback, and extends to web search calls once that feature is enabled.
- **Data handling:** be deliberate about what leaves the device when cloud mode is active — a minor's conversation/voice data going to an external API is worth a conscious decision, not a default. This becomes a harder requirement, not just a nicety, if this is ever sold as a product to other families. The same applies to web search queries once enabled.
- **Role-based access:** now a real requirement, not just a nicety — the student view and the parent view show different data (cost/usage detail especially) and now different *capabilities* (the web-search toggle is parent-only), so the backend needs a basic role concept (`student` / `parent`) gating what the API returns and allows, not just what the frontend chooses to display. Client-side hiding alone would not be a real guardrail. Modeled as a simple role field/enum rather than a hardcoded two-way branch, specifically so a third role (e.g. a teacher, or a second parent) can be added later without restructuring the access-control logic.
- **Auth:** still minimal while this runs only on your own machine for your own household (a lightweight local role switch is enough — full authentication becomes necessary once this is exposed beyond localhost or to other users).

## Ensuring retrieval accuracy

- Chunking aligned to document structure (sections/paragraphs) rather than arbitrary fixed-size splits, so a formula or diagram description stays attached to its surrounding explanation.
- A small set of known-answer test questions per uploaded subject, run after ingestion, to sanity-check that retrieval is actually finding the right material — especially important right after OCR, where garbled text would otherwise fail silently.
- Surfacing the source passage alongside answers so your son — and you — can verify an answer against the actual textbook, which also catches retrieval mistakes by inspection.
- A confidence/low-relevance signal: if nothing in the knowledge base scores as a good match, the system says so or asks a clarifying question, instead of answering confidently from a weak match — and, separately, instead of automatically reaching for web search.

### Known gap: multi-row/column table extraction can scramble row alignment (flagged, not yet fixed)

Found while spot-checking the French textbook's grammar tables post-ingestion (not a scanned-page issue — these pages extract via plain PyMuPDF text, never flagged for the OCR/vision path). Most tables survive fine: single-answer tables (conjugation tables, the pronoms possessifs table, the computer-parts diagram's label↔number pairs) extract with their row pairings intact even when the visual column layout is flattened.

Two tables did not: a fill-in-the-blank grammar table (Y vs. EN usage) and a multi-row phrase-transformation table (Style Direct/Indirect), both on PDF pages in this document. In both cases, a bold/colored inline answer span (or, for the direct/indirect table, an entire late table row) gets pulled out of visual reading order by PyMuPDF's linear text extraction — landing detached from the sentence it belongs to, or shifted to the wrong position in a multi-row table. The rule/header text and overall vocabulary are still correct; only the row-to-row correspondence for rows beyond the first breaks down. Root cause looks like a content-stream ordering quirk for those specific spans, not something the current ingestion code (plain `page.get_text()`, no table-structure reconstruction) accounts for.

Not fixed yet — flagged for a later retrieval-accuracy pass. If it recurs across other documents, options worth evaluating then: `page.get_text("dict")`/`"rawdict"` with position-based row reconstruction, or pdfplumber's explicit table-extraction mode (already in the approved stack per "Tech stack & libraries" above) for pages that look like a grid.

## Code clarity — now and at team scale

- One responsibility per module, explicit interfaces, no cross-module reaching into internals — this is what makes solo debugging tractable (a bug in TTS output can only be in the `voice/` module or the contract into it) and what makes onboarding a future teammate to one module possible without reading the whole system.
- Type hints throughout (mypy) and TypeScript on the frontend, so contracts are checked, not just documented.
- Tests per module, kept close to the code they test, so a change's blast radius is visible immediately rather than discovered later.
- A short README per backend module stating its contract (inputs, outputs, what it deliberately does *not* do) — cheap now, valuable the day someone else joins.

## Validation & human checkpoints

- **Per-module, as it's built:** each backend module is independently runnable and testable — you can validate Ingestion + Knowledge Base purely by uploading a document and inspecting retrieval results, before ASR/TTS exist at all.
- **Natural human checkpoints:** after Ingestion, after the first end-to-end text-only chat loop, after the voice loop is wired, after the visual companion is connected, and after web search (once built) — confirm it only fires on explicit request and never bypasses grounding silently. Each is a point where you review and sign off before the next module builds on it.
- **Final validation:** a full end-to-end pass with your son as the actual user.

## Commercialization considerations (kept in mind, not built for v1)

- **Multi-tenancy:** today's data model extends naturally to per-user/per-tenant namespacing later — and the student/parent role split decided now is itself an early, small step in that direction (a real product would generalize these into per-family account membership, with parent remaining the natural owner/billing role).
- **Billing:** the observability/cost-tracking design already captures per-session, per-user cost, including web search once enabled — the exact data a usage-based billing system would need.
- **Compliance:** selling a product that tutors other people's children raises real, non-technical requirements (data privacy for minors, consent, data retention policy) that need attention before any commercial step — web search adds sending queries to a third-party search provider as one more thing that policy would need to cover.
- **Configurability over hardcoding:** branding, subjects, provider choices, vector-store backend, UI theme, role-based views, and the web-search toggle are all designed as config/preference, not hardcoded.
- **Deployment shape:** the local-first, modular-monolith design intentionally leaves room to swap local Ollama for a hosted inference backend without touching the rest of the app. On typical consumer laptop hardware (as this assessment shows), a real commercial product likely leans cloud-inference by default anyway — worth keeping in mind if "runs entirely offline" was ever assumed as a selling point.

## Theming (end-user selectable UI themes)

- **Mechanism:** a small set of design tokens (CSS custom properties) rather than colors hardcoded per component. A theme is a named set of token values; switching themes swaps which token set is active.
- **Presets to ship with v1:** Light (default), Dark, and one High-contrast/accessible option.
- **Where the choice lives:** exposed in the Settings panel next to the model-provider selector, persisted per user (`localStorage` initially, or a `user_preferences` row in the backend if it should follow the user across devices).
- **Scope:** applies to the whole app shell so a themed screenshot looks coherent.

## UI direction (decided, from the design-exploration canvas)

Three directions were explored on a canvas (Calm & Focused, Playful & Energetic, Dense & Information-forward — see the published design artifact). Decision:

- **Base aesthetic: Calm & Focused** — serif/sans pairing, warm neutrals, generous whitespace. This is the look the rest of the UI builds on.
- **Navigation: adopt the Dense direction's left-side subject/chapter nav** — a persistent left sidebar listing subjects and, nested under each, chapters/lessons, replacing the Calm direction's bottom subject-chip strip. This also gives ad-hoc uploads a natural home in the same sidebar (as in the Dense mockup).
- **Two actors, two role-scoped views — locked for v1:**
  - **Student view:** the conversation + visual companion + nav, with a cost indicator that's **off by default and toggle-able** — visible only if the student chooses to see it, never the detailed breakdown. If web search is enabled by the parent, the student can invoke it explicitly but cannot turn it on/off.
  - **Parent view:** the same core screens plus the **detailed consumption view** — per-session and per-request cost, token counts, weekly/monthly aggregates, retrieval health — pulling directly from the `usage_events` data described in Observability above, plus the **web-search enable/disable toggle**. This is a separate screen/route, not just a toggle on the student screen, and is gated by the role check described in Guardrails.
  - Locked to exactly these two for v1, but the role is implemented as an extensible field (not a hardcoded student/parent-only branch), so a future role — a teacher, a tutor-admin, a second parent — can be added later by extending that field rather than reworking the access model.
- **Not re-mocked yet, deliberately** — this decision is recorded here as the brief for the UI module; the actual layout (sidebar width, nav tree behavior, the parent view's own layout) gets built directly in the frontend code rather than as another round of static mockups, per the validation-checkpoint approach: judge the real, editable thing once it exists rather than a picture of it twice.

## Recommended reading order for whoever builds a module

1. `overview.md` — why this exists, for whom.
2. `architecture.md` — the module map and data flow.
3. This file — the specifics for whichever module you're building: stack, patterns to follow, guardrails to respect, the hardware-driven provider default, the vector store choice, the web search enhancement's rules if relevant, and (for anything UI) the decided direction.
4. `development-plan.md` — where this fits in the build order (which phase, what its exit checkpoint is).
