# Project Tutor — Development Plan (v1)

Companion to overview.md, architecture.md, and technical-design.md. Turns those decisions into a build order: what gets built first, what depends on what, and where to stop and sign off before the next piece builds on it. A formatted .docx of this same plan was delivered to Raj directly.

## 1. Sequencing principle

Build bottom-up through the dependency graph, not top-down through the user journey. Orchestration depends on Model Provider and Storage; Ingestion/Knowledge depend on Model Provider (embeddings); Voice I/O and the Visual Companion are thin edges added once there's something real underneath them to drive. Each phase produces something independently testable — most without a UI, several without voice — so validation happens continuously rather than only at the end.

Two things confirmed since the design docs were written, folded into this plan:

- **Provider default confirmed in practice:** the Anthropic API responds meaningfully faster than a local `llama3.2:1b` run, matching the hardware assessment's call. Ollama stays wired in behind the Model Provider interface for offline/dev use, but the Anthropic API is what Phase 1 builds and tests against.
- **Typed chat mode is a small addition, not new backend work:** the orchestration/API layer is already text-in, text-out (voice is ASR/TTS bolted onto that same contract), so a typed-chat input next to the mic is a Phase 4 frontend addition — a text box calling the same endpoint.

## 2. Phases

**Phase 0 — Repo scaffold.** Backend + frontend scaffolds per the agreed repo layout, health-check endpoint, env/config wiring (provider switch, API keys via `.env`, never committed). Modules: repo layout only. Exit: `uvicorn` runs, `/health` returns ok, `npm run dev` serves the Vite shell.

**Phase 1 — Model Provider + Storage + minimal Orchestration.** Model Provider interface + Anthropic adapter (default) + Ollama adapter; SQLite schema for sessions/subjects/messages/usage_events; Orchestration's session-state loop calling the provider with a plain (ungrounded) prompt. Modules: models, storage, orchestration. Exit: a scripted conversation (no retrieval, no voice) runs end-to-end against both adapters via a config flip; usage_events rows written per call.

**Phase 2 — Ingestion + Knowledge Base.** Upload handler + scanned-vs-text-layer detection; OCR/vision pass for scanned pages and diagrams/formulae; structure-aware chunking; embeddings into the per-subject vector store; known-answer sanity-check questions post-ingestion. Modules: ingestion, knowledge, models (embeddings). Exit: upload a real CBSE French chapter PDF; known-answer questions retrieve the right passages; a scanned page triggers the OCR path and produces sane text.

**Phase 3 — Grounded chat loop (text-only).** Retrieval → grounded prompt → Model Provider → response, with the grounding guardrail and low-confidence/no-good-match signal. Modules: orchestration, knowledge, models. Exit: a covered question is answered grounded/correct with source passage cited; an uncovered question gets an explicit "not covered" answer — both via plain text API, no frontend needed.

**Phase 4 — Frontend shell + typed chat.** React/Vite shell, left-sidebar subject/chapter nav (Dense-direction navigation), Calm-and-Focused aesthetic, theming tokens, upload panel, settings panel (provider + theme), typed chat calling the Phase 3 endpoint, student-view cost toggle (off by default) wired to real usage_events. Modules: api (frontend client), frontend views/components, storage (preferences). Exit: hold a full grounded conversation by typing, switch subjects/themes, upload an ad-hoc document from the browser.

**Phase 5 — Voice I/O.** ASR wrapper (faster-whisper) and TTS wrapper (Piper) behind their interfaces; formula/notation → spoken-form preprocessor; WebSocket streaming endpoint for duplex audio; control-command routing ahead of tutoring content. Modules: voice, orchestration (command routing), api (WebSocket). Exit: a spoken question gets a spoken, grounded answer at acceptable latency; a spoken control command switches subject without being treated as a tutoring question.

**Phase 6 — Visual Companion + Parent view.** Visual Companion panel (KaTeX for formulae, render-on-directive); Orchestration emits visual directives alongside spoken text; parent-only route with per-request/session/weekly cost, token counts, retrieval health, gated by backend role check. Modules: orchestration (directives), frontend (VisualCompanion, ParentView), api (role gate). Exit: a formula renders correctly and stays synced with speech; the parent role changes what the API returns, not just what's displayed.

**Phase 6.5 — User-invoked web search (enhancement).** A `web_search` tool behind the same tool-calling mechanism as "render this formula"/"switch subject", callable only on an explicit user request — never as an automatic fallback when the knowledge base doesn't cover a question, so the grounding guardrail from Phase 3 stays intact. Ships behind a `web_search_enabled` config flag, off by default, toggleable only from the parent-only Settings screen (backend-enforced, per the Phase 6 role gate — not just hidden in the student UI). Includes: the tool adapter itself, its own `usage_events` logging (so parent-view cost stays accurate), and visibly distinct labeling of web-sourced answers (spoken cue + a "from the web" indicator with source link in the Visual Companion) so it's never ambiguous whether an answer came from the syllabus KB or the open web. Modules: orchestration (tool + gating), models (tool adapter), storage (preference flag), frontend (Settings toggle, web-source labeling), observability. Exit: with the flag off, an explicit "search the web for X" request is declined/ignored; with it on, the same request returns a web-sourced, clearly labeled answer, logged with its own cost row, and a KB question with no good match still falls back to "not covered," never silently to a web search.

**Phase 7 — End-to-end validation with your son.** A real CBSE French study session, voice-first, covering a full chapter, with Raj observing. Modules: all, integrated. Exit: he can hold a useful, on-syllabus French practice conversation unassisted; issues found feed a punch-list, not a redesign.

## 3. Cross-cutting work (ongoing, not a phase)

- Tests alongside each module as built (pytest/ruff/mypy backend, typed contracts frontend) — not deferred to a hardening pass.
- Structured logging per module from Phase 1 onward, since the voice pipeline (Phase 5) has five stages and failures need to be traceable to one.
- Cost guardrail (usage cap/alert on the Anthropic API) turned on as soon as Phase 1's usage_events table exists; extended to cover web search once Phase 6.5 ships.
- Config-over-hardcoding discipline (provider, model, theme, role, web-search toggle) enforced from Phase 0.

## 4. What's explicitly deferred

- Multi-agent / LangChain-style orchestration — only revisited if Phase 3's hand-rolled loop runs into a concrete limitation.
- Full authentication — the lightweight student/parent role switch is sufficient while this runs on one machine for one household.
- Commercialization work (multi-tenancy, billing, compliance) — the data model and role split extend into these later, but nothing is built for v1.
- Phase 2 of the vision (non-French subjects) — contingent on Phase 7 succeeding for French first.
