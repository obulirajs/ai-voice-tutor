// Mirrors the Pydantic response models in backend/app/api/*.py -- kept in
// sync by hand for now (see technical-design.md: TypeScript makes the
// frontend<->backend contract explicit and checked, not just documented).

export interface Subject {
  id: number
  name: string
  document_count: number
  created_at: string
}

export interface Document {
  id: number
  filename: string
  page_count: number
  scanned_page_count: number
  chunk_count: number
  created_at: string
}

export interface ConsistencyCheckFailure {
  chunk_index: number
  page_number: number | null
}

export interface ConsistencyCheckResult {
  sampled: number
  passed: number
  failed: ConsistencyCheckFailure[]
}

// Mirrors backend/app/ingestion/pipeline.py's PageQualitySummary -- the
// per-page assess_text_quality() results from one ingestion run.
export interface QualitySummary {
  total_pages: number
  good_pages: number
  poor_pages: number
  empty_pages: number
  ocr_fallback_pages: number
  avg_score: number
  worst_page: number | null
  worst_score: number
}

export interface DocumentIngestResult {
  document_id: number
  page_count: number
  scanned_page_count: number
  chunk_count: number
  consistency_check: ConsistencyCheckResult
  subject_mismatch_warning: string | null
  quality_summary: QualitySummary | null
  quality_warning: string | null
}

export interface SourceCitation {
  page_number: number | null
  score: number
}

// Mirrors backend/app/orchestration/visual_directives.py's VisualDirective --
// the Visual Companion's render-on-directive contract (architecture.md).
export interface VisualDirective {
  directive_type: 'formula' | 'highlight' | 'text_block' | 'image_ref' | 'supplemented'
  content: string
  label: string | null
}

export interface ChatResponse {
  reply: string
  session_id: string
  sources: SourceCitation[]
  visual_directives: VisualDirective[]
}

// Mirrors backend/app/orchestration/prompt_builder.py's TEACHING_MODES --
// changes only how the model frames its answer, never what gets retrieved.
export type TeachingMode = 'textbook' | 'teacher'

// A lightweight student/parent distinction, NOT authentication -- see
// backend/app/api/role.py.
export type Role = 'student' | 'parent'

export interface UsageSummary {
  total_cost_usd: number
  total_sessions: number
  total_turns: number
  cost_by_event_type: Record<string, number>
  cost_by_subject: Array<{ subject_name: string; cost_usd: number }>
}

export interface SessionListItem {
  session_id: string
  subject_name: string
  created_at: string
  cost_usd: number
  turn_count: number
  event_count: number
}

export interface RetrievalHealth {
  total_queries: number
  guardrail_triggered_count: number
  guardrail_trigger_rate: number
  avg_best_score: number | null
  score_distribution: {
    excellent: number
    good: number
    fair: number
    poor: number
  }
}

export interface ParentSettings {
  web_search_enabled: boolean
}

export interface UsageEvent {
  event_type: string
  provider: string
  model: string
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number
  created_at: string
}

export interface SessionUsage {
  session_id: string
  total_cost_usd: number
  total_input_tokens: number
  total_output_tokens: number
  event_count: number
  events: UsageEvent[]
}

export interface HealthStatus {
  status: string
}

export interface SessionPreview {
  session_id: string
  subject_id: number
  message_count: number
  last_message_preview: string
  last_active: string
  teaching_mode: TeachingMode
}

export interface MessageResponse {
  role: string
  content: string
}

export interface VoiceOption {
  id: string
  name: string
  language: string
  gender: string
  description: string
}

// The voice WebSocket's per-turn "response" frame (see
// backend/app/api/voice.py's protocol docstring) plus the TTS audio blob
// assembled from the binary frame that follows it.
export interface VoiceResponse {
  transcription: string
  reply: string
  sources: SourceCitation[]
  session_id: string
  is_command: boolean
  command_type: string | null
  // Set only when a switch_subject command actually resolved to a real subject.
  new_subject: string | null
  visual_directives: VisualDirective[]
  audio: Blob
}
