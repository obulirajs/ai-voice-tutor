// Typed fetch wrapper for the backend API. Vite's dev proxy (vite.config.ts)
// forwards /api and /health to the backend, so paths here are relative --
// no base URL to configure per environment.

import type {
  ChatResponse,
  Document,
  DocumentIngestResult,
  HealthStatus,
  MessageResponse,
  ParentSettings,
  RetrievalHealth,
  Role,
  SessionListItem,
  SessionPreview,
  SessionUsage,
  Subject,
  TeachingMode,
  UsageSummary,
  VoiceOption,
} from './types'

// Which role's header (X-Tutor-Role) every request carries -- module-level
// since it's a cross-cutting concern of the fetch wrapper itself, not
// per-call data. The store's setRole() action keeps this in sync with the
// persisted role preference; see api/role.py for how the backend reads it.
let currentRole: Role = 'student'

export function setRole(role: Role): void {
  currentRole = role
}

export function getRole(): Role {
  return currentRole
}

// Thrown when the backend couldn't be reached at all -- either fetch()
// itself failed (server down, DNS failure, CORS block) or a proxy in front
// of it (Vite's dev proxy, or a reverse proxy in production) answered with
// a gateway-error status because the upstream it forwards to is down.
// Distinct from a normal HTTP error response, so callers (e.g. the chat
// store) can show a "can't reach the server" banner instead of the
// backend's own error detail.
export class NetworkError extends Error {}

const GATEWAY_ERROR_STATUSES = new Set([502, 503, 504])

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') {
        return detail
      }
    }
  } catch {
    // Body wasn't JSON (or was empty) -- fall through to the generic message.
  }
  return `Request failed with status ${response.status}`
}

async function doFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers)
  headers.set('X-Tutor-Role', currentRole)

  let response: Response
  try {
    response = await fetch(path, { ...init, headers })
  } catch {
    throw new NetworkError('Could not reach the server. Check that the backend is running.')
  }
  if (GATEWAY_ERROR_STATUSES.has(response.status)) {
    throw new NetworkError('Could not reach the server. Check that the backend is running.')
  }
  return response
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await doFetch(path, init)
  if (!response.ok) {
    throw new Error(await extractErrorMessage(response))
  }
  return (await response.json()) as T
}

async function requestNoContent(path: string, init?: RequestInit): Promise<void> {
  const response = await doFetch(path, init)
  if (!response.ok) {
    throw new Error(await extractErrorMessage(response))
  }
}

export function getSubjects(): Promise<Subject[]> {
  return request<Subject[]>('/api/subjects')
}

export function getSubjectDocuments(subjectId: number): Promise<Document[]> {
  return request<Document[]>(`/api/subjects/${subjectId}/documents`)
}

export function uploadDocument(subject: string, file: File): Promise<DocumentIngestResult> {
  const formData = new FormData()
  formData.append('file', file)
  return request<DocumentIngestResult>(`/api/subjects/${encodeURIComponent(subject)}/documents`, {
    method: 'POST',
    body: formData,
  })
}

export function deleteDocument(subject: string, documentId: number): Promise<void> {
  return requestNoContent(`/api/subjects/${encodeURIComponent(subject)}/documents/${documentId}`, {
    method: 'DELETE',
  })
}

export function chat(
  message: string,
  sessionId?: string | null,
  subject?: string | null,
  mode?: TeachingMode,
): Promise<ChatResponse> {
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId ?? null,
      subject: subject ?? null,
      mode: mode ?? 'textbook',
    }),
  })
}

export function getSessionUsage(sessionId: string): Promise<SessionUsage> {
  return request<SessionUsage>(`/api/sessions/${encodeURIComponent(sessionId)}/usage`)
}

export function healthCheck(): Promise<HealthStatus> {
  return request<HealthStatus>('/health')
}

export function getUsageSummary(period = 'week'): Promise<UsageSummary> {
  return request<UsageSummary>(`/api/parent/usage/summary?${new URLSearchParams({ period }).toString()}`)
}

export function getSessionList(period = 'week', limit = 20, offset = 0): Promise<SessionListItem[]> {
  const params = new URLSearchParams({ period, limit: String(limit), offset: String(offset) })
  return request<SessionListItem[]>(`/api/parent/usage/sessions?${params.toString()}`)
}

export function getRetrievalHealth(period = 'week'): Promise<RetrievalHealth> {
  return request<RetrievalHealth>(`/api/parent/retrieval-health?${new URLSearchParams({ period }).toString()}`)
}

export function getParentSettings(): Promise<ParentSettings> {
  return request<ParentSettings>('/api/parent/settings')
}

export function getSubjectSessions(subjectId: number, limit = 20): Promise<SessionPreview[]> {
  return request<SessionPreview[]>(`/api/subjects/${subjectId}/sessions?limit=${limit}`)
}

export function getSessionMessages(sessionId: string): Promise<MessageResponse[]> {
  return request<MessageResponse[]>(`/api/sessions/${encodeURIComponent(sessionId)}/messages`)
}

export async function synthesizeSpeech(text: string, voice?: string | null): Promise<Blob> {
  const response = await doFetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice: voice ?? null }),
  })
  if (!response.ok) {
    throw new Error(await extractErrorMessage(response))
  }
  return response.blob()
}

export function getAvailableVoices(): Promise<VoiceOption[]> {
  return request<VoiceOption[]>('/api/voices')
}
