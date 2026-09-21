import type { SessionPreview } from '../api'
import './SessionHistory.css'

interface SessionHistoryProps {
  sessions: SessionPreview[]
  loading: boolean
  currentSessionId: string | null
  onResume: (sessionId: string) => void
  onNewChat: () => void
}

const MODE_ICON: Record<string, string> = {
  textbook: '📖',
  teacher: '👨‍🏫',
}

function relativeTime(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(isoDate).toLocaleDateString()
}

export function SessionHistory({ sessions, loading, currentSessionId, onResume, onNewChat }: SessionHistoryProps) {
  return (
    <div className="session-history">
      <p className="sidebar__section-label">History</p>

      <button type="button" className="session-history__new-chat" onClick={onNewChat}>
        + New Chat
      </button>

      {loading ? (
        <div className="session-history__skeleton" aria-hidden="true">
          <div className="session-history__skeleton-row" />
          <div className="session-history__skeleton-row" />
        </div>
      ) : sessions.length === 0 ? (
        <p className="session-history__empty">No conversations yet. Ask a question to start!</p>
      ) : (
        <ul className="session-history__list">
          {sessions.map((session) => {
            const isActive = session.session_id === currentSessionId
            return (
              <li key={session.session_id}>
                <button
                  type="button"
                  className={`session-history__item ${isActive ? 'session-history__item--active' : ''}`}
                  onClick={() => onResume(session.session_id)}
                >
                  <span className="session-history__item-top">
                    <span className="session-history__mode" aria-hidden="true">
                      {MODE_ICON[session.teaching_mode] ?? '📖'}
                    </span>
                    <span className="session-history__preview">{session.last_message_preview || 'New conversation'}</span>
                  </span>
                  <span className="session-history__item-bottom">
                    <span className="session-history__time">{relativeTime(session.last_active)}</span>
                    <span className="session-history__count">{session.message_count}</span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
