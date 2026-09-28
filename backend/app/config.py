"""Central defaults — every tunable default value that isn't itself an env
var, grouped by the module that owns it.

This is deliberately NOT where env var names or .env.example documentation
live — those stay exactly where they are, next to the os.getenv() call that
reads them (app.models.get_provider(), app.voice.get_tts_provider(), ...).
What lives here is the second argument to that os.getenv() call: the
fallback value a module uses when the env var isn't set, plus the handful
of thresholds/constants that have no env var at all today (chunk size,
PDF quality-gate cutoffs, retry policy, pricing).

Two things this fixes:
  - Discoverability: one file to scan for every tunable default, instead of
    ~15 files each hiding one or two.
  - Duplication: a few defaults (e.g. Whisper's "base"/"cpu", edge-tts's
    voice, Piper's voice/data dir) were previously hardcoded twice --
    once as the provider factory's os.getenv() fallback, once again as the
    adapter class's own __init__ default -- with no guarantee the two
    copies stayed in sync. Each now has exactly one source of truth here.

Each group below is a frozen dataclass instantiated once at the bottom of
this file; modules import the instance (never the class) and never
redefine these values locally. Kept as plain stdlib dataclasses rather than
a settings/validation library -- these aren't read from the environment
directly, so there's nothing for a validation layer to do.

Deliberately excluded: env var names themselves (stay at the call site),
user-facing copy text (e.g. the grounding guardrail's "not covered" reply),
and algorithm data that isn't really a tunable knob (regex tables, the
subject/language lookup table, the curated TTS voice catalog).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelDefaults:
    """app.models — Model Provider, embeddings, vision, and pricing."""

    provider: str = "anthropic"
    ollama_model: str = "llama3.2:1b"
    # Off by default -- a thinking-capable model (e.g. Qwen3.5) emits a
    # <think>...</think> reasoning block before its answer, which adds
    # latency the voice loop can't afford. See OllamaProvider.
    ollama_think: bool = False
    anthropic_model: str = "claude-sonnet-5"

    embedding_provider: str = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"

    vision_provider: str = "tesseract"
    tesseract_langs: str = "eng+fra"
    anthropic_vision_model: str = "claude-sonnet-5"

    # Anthropic adapters' max_tokens -- no env var; a hard ceiling per call,
    # not something a deployment would usually tune per-environment.
    anthropic_max_tokens: int = 1024
    anthropic_vision_max_tokens: int = 2048

    # OllamaEmbeddingProvider's retry policy for a transient local
    # model-runner drop mid-ingestion (see ollama_embedding_adapter.py).
    embedding_retry_max_attempts: int = 3
    embedding_retry_delay_seconds: float = 2.0

    # USD per million tokens: {model: (input, output)}. Anything not listed
    # (including any Ollama model) is treated as $0. Source:
    # platform.claude.com/docs pricing.
    pricing_usd_per_mtok: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "claude-sonnet-5": (2.0, 10.0),
        }
    )


@dataclass(frozen=True)
class VoiceDefaults:
    """app.voice — ASR/TTS provider selection and adapter tuning."""

    asr_provider: str = "whisper"
    tts_provider: str = "edge"

    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    # Greedy decoding -- fine for conversational speech and meaningfully
    # faster than faster-whisper's beam_size=5 default on a CPU-only machine.
    whisper_beam_size: int = 1

    edge_voice: str = "en-US-AriaNeural"
    piper_voice: str = "en_US-lessac-medium"
    piper_data_dir: str = "./piper_voices"


@dataclass(frozen=True)
class KnowledgeDefaults:
    """app.knowledge — vector store, retrieval, and QA/consistency checks."""

    vector_store: str = "sqlite_vec"
    vector_store_path: str = "./vector_store.db"

    # The grounding guardrail's cutoff: how many chunks to retrieve, and the
    # minimum cosine similarity for the best match to count as "covered".
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.55

    # Standalone golden-QA sanity check (POST .../sanity-check).
    golden_qa_top_k: int = 5
    golden_qa_low_confidence_threshold: float = 0.5
    golden_qa_snippet_length: int = 200
    golden_qa_page_match_tolerance: int = 1

    # Automatic post-ingestion self-retrieval check.
    consistency_sample_size: int = 5
    consistency_top_k: int = 3

    # Subject/language mismatch sanity check.
    language_check_min_signal_chars: int = 3
    language_check_sample_size: int = 5


@dataclass(frozen=True)
class IngestionDefaults:
    """app.ingestion — upload limits, chunking, and the scan/quality gates."""

    # Mirrors the UPLOAD_MAX_SIZE_MB env var's fallback (api/documents.py).
    upload_max_size_mb: int = 50

    chunk_max_chars: int = 900

    # Scanned-vs-text-layer page detection (pdf.page_is_scanned()).
    min_words_per_page: int = 30
    content_image_coverage_ratio: float = 0.15
    background_image_area_ratio: float = 0.95
    garbled_text_ratio_threshold: float = 0.3

    # Per-page text-quality scoring (pdf.assess_text_quality()).
    quality_min_word_count: int = 5
    quality_target_avg_word_length: float = 5.0
    quality_good_threshold: float = 0.60

    # Ingestion-run-level quality warnings (pipeline._summarize_page_quality()).
    poor_pages_warning_ratio: float = 0.3
    avg_score_warning_threshold: float = 0.5

    # DPI for rendering a page to PNG before a vision/OCR call.
    page_render_dpi: int = 200
    # DPI for rendering a page to PNG for the Visual Companion's page-view
    # feature -- lower than page_render_dpi since this is for a student to
    # read on screen, not for OCR accuracy, and keeps cached files small.
    page_view_dpi: int = 110


@dataclass(frozen=True)
class StorageDefaults:
    """app.storage — the metadata database, plus the on-disk root for
    persisted uploaded PDFs and their cached page-image renders (see
    app.storage.files) -- architecture.md: storage owns "filesystem for the
    actual documents/images", alongside the SQLite metadata DB.
    """

    database_url: str = "sqlite+aiosqlite:///./tutor.db"
    data_dir: str = "./data"


@dataclass(frozen=True)
class OrchestrationDefaults:
    """app.orchestration — the grounded-chat prompt loop."""

    max_history_messages: int = 10
    default_teaching_mode: str = "textbook"
    # Bold terms are noisy in a long reply -- cap how many become highlight
    # directives for the Visual Companion.
    max_highlight_directives: int = 5
    # A reply comparing many things could contain several tables -- cap how
    # many become table directives for the Visual Companion.
    max_table_directives: int = 3


@dataclass(frozen=True)
class DiagramDefaults:
    """app.orchestration.diagrams -- on-demand Mermaid diagram generation
    (the Visual Companion's "Draw diagram" button). A small model turns the
    tutor's last reply into a flowchart; see diagrams.py for the prompt and
    sanitizer this backs.
    """

    provider: str = "ollama"  # ollama (default, local/free) | anthropic (fallback) | off
    ollama_model: str = "qwen2.5-coder:3b"
    anthropic_model: str = "claude-haiku-4-5"

    max_nodes: int = 12
    max_label_words: int = 6
    timeout_s: float = 60.0
    temperature: float = 0.2

    # In-memory LRU cache size, keyed by sha256(message_text + previous_error).
    cache_size: int = 64
    # Cap on how much retrieved-passage text is added to the prompt as
    # grounding context, alongside the tutor's already-grounded reply.
    max_context_chars: int = 1500


MODEL_DEFAULTS = ModelDefaults()
VOICE_DEFAULTS = VoiceDefaults()
KNOWLEDGE_DEFAULTS = KnowledgeDefaults()
INGESTION_DEFAULTS = IngestionDefaults()
STORAGE_DEFAULTS = StorageDefaults()
ORCHESTRATION_DEFAULTS = OrchestrationDefaults()
DIAGRAM_DEFAULTS = DiagramDefaults()
