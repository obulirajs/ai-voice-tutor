import { useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'

import type { Role } from '../api'
import { useModalDismiss } from '../hooks/useModalDismiss'
import { useAppStore } from '../store'
import { useTheme } from '../theme'
import type { Theme } from '../theme'
import './SettingsPanel.css'

const THEME_OPTIONS: { value: Theme; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'high-contrast', label: 'High-contrast' },
]

const ROLE_OPTIONS: { value: Role; label: string }[] = [
  { value: 'student', label: 'Student' },
  { value: 'parent', label: 'Parent' },
]

export function SettingsPanel() {
  const isOpen = useAppStore((state) => state.isSettingsPanelOpen)
  const closeSettingsPanel = useAppStore((state) => state.closeSettingsPanel)
  const role = useAppStore((state) => state.role)
  const setRole = useAppStore((state) => state.setRole)
  const availableVoices = useAppStore((state) => state.availableVoices)
  const selectedVoice = useAppStore((state) => state.selectedVoice)
  const loadVoices = useAppStore((state) => state.loadVoices)
  const setSelectedVoice = useAppStore((state) => state.setSelectedVoice)
  const previewVoice = useAppStore((state) => state.previewVoice)
  const { theme, setTheme } = useTheme()
  const panelRef = useModalDismiss(isOpen, closeSettingsPanel)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    if (isOpen && availableVoices.length === 0) void loadVoices()
  }, [isOpen, availableVoices.length, loadVoices])

  if (!isOpen) return null

  const handleVoiceSelect = (voiceId: string) => {
    setSelectedVoice(voiceId)
    void previewVoice(voiceId)
  }

  const handleRoleChange = (nextRole: Role) => {
    setRole(nextRole)
    // The parent-only route isn't reachable once the role switches back --
    // leave it before the next render's role gate would bounce anyway.
    if (nextRole === 'student' && location.pathname === '/parent') {
      navigate('/')
    }
  }

  return (
    <div className="settings-panel-overlay" onClick={closeSettingsPanel}>
      <div
        ref={panelRef}
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-panel-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="settings-panel__header">
          <h2 className="settings-panel__title" id="settings-panel-title">
            Settings
          </h2>
          <button type="button" className="settings-panel__close" aria-label="Close" onClick={closeSettingsPanel}>
            ×
          </button>
        </div>

        <section className="settings-panel__section">
          <h3 className="settings-panel__section-title">Theme</h3>
          <div className="settings-panel__theme-options" role="radiogroup" aria-label="Theme">
            {THEME_OPTIONS.map((option) => (
              <label key={option.value} className="settings-panel__theme-option">
                <input
                  type="radio"
                  name="theme"
                  value={option.value}
                  checked={theme === option.value}
                  onChange={() => setTheme(option.value)}
                />
                {option.label}
              </label>
            ))}
          </div>
        </section>

        <section className="settings-panel__section">
          <h3 className="settings-panel__section-title">Role</h3>
          <div className="settings-panel__theme-options" role="radiogroup" aria-label="Role">
            {ROLE_OPTIONS.map((option) => (
              <label key={option.value} className="settings-panel__theme-option">
                <input
                  type="radio"
                  name="role"
                  value={option.value}
                  checked={role === option.value}
                  onChange={() => handleRoleChange(option.value)}
                />
                {option.label}
              </label>
            ))}
          </div>
          {role === 'parent' && (
            <div className="settings-panel__role-confirmation">
              <p className="settings-panel__role-confirmation-text">You're viewing as Parent.</p>
              <Link to="/parent" className="settings-panel__dashboard-link" onClick={closeSettingsPanel}>
                View Dashboard →
              </Link>
            </div>
          )}
        </section>

        <section className="settings-panel__section">
          <h3 className="settings-panel__section-title">Voice</h3>
          <p className="settings-panel__description">Choose the voice for spoken responses</p>
          <div className="settings-panel__voice-grid">
            {availableVoices.map((voice) => (
              <button
                key={voice.id}
                type="button"
                className={`settings-panel__voice-card ${
                  selectedVoice === voice.id ? 'settings-panel__voice-card--active' : ''
                }`}
                onClick={() => handleVoiceSelect(voice.id)}
              >
                <span className="settings-panel__voice-name">{voice.name}</span>
                <span className="settings-panel__voice-meta">
                  {voice.language} · {voice.gender}
                </span>
                <span className="settings-panel__voice-desc">{voice.description}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="settings-panel__section">
          <h3 className="settings-panel__section-title">Model Provider</h3>
          <p className="settings-panel__provider">
            Anthropic API
            <span className="settings-panel__provider-note"> (configured in backend .env)</span>
          </p>
        </section>

        <section className="settings-panel__section">
          <h3 className="settings-panel__section-title">About</h3>
          <p className="settings-panel__about-name">Project Tutor</p>
          <p className="settings-panel__about-version">v1 — Phase 4</p>
          <p className="settings-panel__about-tagline">Voice-interactive CBSE tutor</p>
        </section>
      </div>
    </div>
  )
}
