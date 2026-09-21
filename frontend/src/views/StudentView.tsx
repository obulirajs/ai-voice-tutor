import { useEffect, useState } from 'react'
import '../App.css'
import { ChatView } from '../components/ChatView'
import { CostIndicator } from '../components/CostIndicator'
import { SessionHistory } from '../components/SessionHistory'
import { SettingsPanel } from '../components/SettingsPanel'
import { SubjectNav } from '../components/SubjectNav'
import { UploadPanel } from '../components/UploadPanel'
import { VisualCompanion } from '../components/VisualCompanion'
import { useAppStore } from '../store'
import { useTheme } from '../theme'

// The student's main tutoring interface: sidebar + chat + Visual Companion,
// the app's original single-view shell before react-router split it out
// from the parent-only dashboard (ParentView).
export function StudentView() {
  const [isSidebarOpen, setSidebarOpen] = useState(false)
  // Remounting ChatView on subject switch resets its local UI state (draft
  // text, expanded source rows) for free -- the conversation itself already
  // resets via the store's setActiveSubject.
  const activeSubjectId = useAppStore((state) => state.activeSubjectId)
  const openSettingsPanel = useAppStore((state) => state.openSettingsPanel)
  const setStoreTheme = useAppStore((state) => state.setTheme)
  const sessionId = useAppStore((state) => state.sessionId)
  const sessions = useAppStore((state) => state.sessions)
  const sessionsLoading = useAppStore((state) => state.sessionsLoading)
  const resumeSession = useAppStore((state) => state.resumeSession)
  const startNewChat = useAppStore((state) => state.startNewChat)

  // ThemeContext (localStorage + the data-theme attribute) is the actual
  // theme mechanism; mirroring it into the store just lets any future
  // component read theme reactively from Zustand without using the context.
  const { theme } = useTheme()
  useEffect(() => {
    setStoreTheme(theme)
  }, [theme, setStoreTheme])

  useEffect(() => {
    if (!isSidebarOpen) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSidebarOpen(false)
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isSidebarOpen])

  return (
    <div className="app-shell">
      <aside className={`sidebar ${isSidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar__header">
          <span className="sidebar__brand">Project Tutor</span>
        </div>
        <SubjectNav onNavigate={() => setSidebarOpen(false)} />
        <SessionHistory
          sessions={sessions}
          loading={sessionsLoading}
          currentSessionId={sessionId}
          onResume={(id) => {
            void resumeSession(id)
            setSidebarOpen(false)
          }}
          onNewChat={() => {
            startNewChat()
            setSidebarOpen(false)
          }}
        />
      </aside>

      {isSidebarOpen && (
        <button
          type="button"
          className="sidebar-overlay"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="main">
        {/* DOM order deliberately puts chat before header (order:-1 in App.css
            puts the header back on top visually) so Tab order reads
            sidebar -> chat input -> send -> header controls (settings last). */}
        <main className="main__content">
          <ChatView key={activeSubjectId ?? 'none'} />
          <VisualCompanion />
        </main>

        <header className="main__header">
          <button
            type="button"
            className="main__menu-button"
            aria-label="Toggle menu"
            onClick={() => setSidebarOpen((open) => !open)}
          >
            <span aria-hidden="true">☰</span>
          </button>
          <span className="main__title">Chat</span>
          <div className="main__header-actions">
            <CostIndicator />
            <button
              type="button"
              className="main__settings-button"
              aria-label="Settings"
              onClick={openSettingsPanel}
            >
              <span aria-hidden="true">⚙</span>
            </button>
          </div>
        </header>
      </div>

      <UploadPanel />
      <SettingsPanel />
    </div>
  )
}
