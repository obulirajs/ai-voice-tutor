import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import {
  getParentSettings,
  getRetrievalHealth,
  getSessionList,
  getUsageSummary,
  NetworkError,
} from '../api'
import type { ParentSettings, RetrievalHealth, SessionListItem, UsageSummary } from '../api'
import { SettingsPanel } from '../components/SettingsPanel'
import { useAppStore } from '../store'
import './ParentView.css'

type Period = 'day' | 'week' | 'month' | 'all'

const PERIOD_OPTIONS: { value: Period; label: string }[] = [
  { value: 'day', label: 'Today' },
  { value: 'week', label: 'This Week' },
  { value: 'month', label: 'This Month' },
  { value: 'all', label: 'All Time' },
]

const SESSION_PAGE_SIZE = 20

const EVENT_TYPE_LABELS: Record<string, string> = {
  generation: 'LLM Generation',
  embedding: 'Embeddings',
  retrieval: 'Retrieval',
  asr: 'ASR',
  tts: 'TTS',
}

const SCORE_BUCKETS: { key: 'excellent' | 'good' | 'fair' | 'poor'; label: string; color: string }[] = [
  { key: 'excellent', label: 'Excellent (≥ 0.80)', color: 'var(--color-success)' },
  // No dedicated "blue" design token exists yet -- this is the one spot in
  // the app that needs a fourth categorical color alongside success/warning/error.
  { key: 'good', label: 'Good (≥ 0.65)', color: '#3a7ca5' },
  { key: 'fair', label: 'Fair (≥ 0.55)', color: 'var(--color-warning)' },
  { key: 'poor', label: 'Poor (< 0.55)', color: 'var(--color-error)' },
]

function formatCurrency(value: number): string {
  return `$${value.toFixed(2)}`
}

function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function scoreColor(score: number | null): string {
  if (score === null) return 'var(--color-text-muted)'
  if (score >= 0.7) return 'var(--color-success)'
  if (score >= 0.55) return 'var(--color-warning)'
  return 'var(--color-error)'
}

function guardrailRateColor(rate: number): string {
  if (rate < 0.1) return 'var(--color-success)'
  if (rate < 0.25) return 'var(--color-warning)'
  return 'var(--color-error)'
}

// Every parent endpoint 403s with this exact detail (api/parent.py's
// _require_parent) when the role header isn't "parent" -- shouldn't happen
// since this view is only reachable via the role-gated /parent route, but
// handled defensively per the Phase 6c spec.
function isRoleError(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes('parent-only')
}

type SectionStatus = 'loading' | 'ready' | 'role-error' | 'connection-error' | 'error'

function statusOf(error: unknown): SectionStatus {
  if (error instanceof NetworkError) return 'connection-error'
  if (isRoleError(error)) return 'role-error'
  return 'error'
}

function SectionError({ status, message }: { status: SectionStatus; message: string | null }) {
  const openSettingsPanel = useAppStore((state) => state.openSettingsPanel)

  if (status === 'role-error') {
    return (
      <div className="parent-view__section-error">
        <p>Switch to Parent role in Settings to view this page.</p>
        <button type="button" className="parent-view__section-error-action" onClick={openSettingsPanel}>
          Open Settings
        </button>
      </div>
    )
  }
  if (status === 'connection-error') {
    return (
      <div className="parent-view__section-error">
        <p>Can't reach the server. Check that the backend is running.</p>
      </div>
    )
  }
  return (
    <div className="parent-view__section-error">
      <p>{message ?? 'Something went wrong loading this section.'}</p>
    </div>
  )
}

function SkeletonRows({ count }: { count: number }) {
  return (
    <div className="parent-view__skeleton" aria-hidden="true">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="parent-view__skeleton-row" />
      ))}
    </div>
  )
}

// Everything below depends on `period`. It's mounted with key={period} from
// ParentView so switching periods remounts it -- state naturally resets to
// its initial "loading" values instead of the effect reaching back to reset
// it (React's documented pattern for "resetting state when a prop changes").
function PeriodDashboard({ period }: { period: Period }) {
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [summaryStatus, setSummaryStatus] = useState<SectionStatus>('loading')
  const [summaryError, setSummaryError] = useState<string | null>(null)

  const [health, setHealth] = useState<RetrievalHealth | null>(null)
  const [healthStatus, setHealthStatus] = useState<SectionStatus>('loading')
  const [healthError, setHealthError] = useState<string | null>(null)

  const [sessions, setSessions] = useState<SessionListItem[]>([])
  const [sessionsStatus, setSessionsStatus] = useState<SectionStatus>('loading')
  const [sessionsError, setSessionsError] = useState<string | null>(null)
  const [sessionsOffset, setSessionsOffset] = useState(0)
  const [sessionsHasMore, setSessionsHasMore] = useState(false)
  const [sessionsLoadingMore, setSessionsLoadingMore] = useState(false)

  useEffect(() => {
    let cancelled = false
    getUsageSummary(period)
      .then((data) => {
        if (cancelled) return
        setSummary(data)
        setSummaryStatus('ready')
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setSummaryError(error instanceof Error ? error.message : null)
        setSummaryStatus(statusOf(error))
      })
    return () => {
      cancelled = true
    }
  }, [period])

  useEffect(() => {
    let cancelled = false
    getRetrievalHealth(period)
      .then((data) => {
        if (cancelled) return
        setHealth(data)
        setHealthStatus('ready')
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setHealthError(error instanceof Error ? error.message : null)
        setHealthStatus(statusOf(error))
      })
    return () => {
      cancelled = true
    }
  }, [period])

  useEffect(() => {
    let cancelled = false
    getSessionList(period, SESSION_PAGE_SIZE, 0)
      .then((data) => {
        if (cancelled) return
        setSessions(data)
        setSessionsHasMore(data.length === SESSION_PAGE_SIZE)
        setSessionsStatus('ready')
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setSessionsError(error instanceof Error ? error.message : null)
        setSessionsStatus(statusOf(error))
      })
    return () => {
      cancelled = true
    }
  }, [period])

  const loadMoreSessions = useCallback(() => {
    const nextOffset = sessionsOffset + SESSION_PAGE_SIZE
    setSessionsLoadingMore(true)
    getSessionList(period, SESSION_PAGE_SIZE, nextOffset)
      .then((data) => {
        setSessions((previous) => [...previous, ...data])
        setSessionsOffset(nextOffset)
        setSessionsHasMore(data.length === SESSION_PAGE_SIZE)
      })
      .catch(() => {
        // "Show more" failing leaves the already-loaded rows in place --
        // the user can just try again.
      })
      .finally(() => setSessionsLoadingMore(false))
  }, [period, sessionsOffset])

  const totalCostByType = summary
    ? Object.values(summary.cost_by_event_type).reduce((sum, value) => sum + value, 0)
    : 0
  const totalCostBySubject = summary ? summary.cost_by_subject.reduce((sum, s) => sum + s.cost_usd, 0) : 0
  const totalScoreCount = health
    ? health.score_distribution.excellent +
      health.score_distribution.good +
      health.score_distribution.fair +
      health.score_distribution.poor
    : 0

  return (
    <>
      <section className="parent-view__section" aria-label="Usage overview">
        {summaryStatus === 'loading' && <SkeletonRows count={4} />}
        {summaryStatus !== 'loading' && summaryStatus !== 'ready' && (
          <SectionError status={summaryStatus} message={summaryError} />
        )}
        {summaryStatus === 'ready' && summary && (
          <>
            <div className="parent-view__cards">
              <div className="parent-view__card">
                <p className="parent-view__card-label">Total Cost</p>
                <p className="parent-view__card-value">{formatCurrency(summary.total_cost_usd)}</p>
                <p className="parent-view__card-subtitle">across {summary.total_sessions} sessions</p>
              </div>
              <div className="parent-view__card">
                <p className="parent-view__card-label">Total Turns</p>
                <p className="parent-view__card-value">{summary.total_turns}</p>
              </div>
              <div className="parent-view__card">
                <p className="parent-view__card-label">Avg Cost / Session</p>
                <p className="parent-view__card-value">
                  {summary.total_sessions > 0
                    ? formatCurrency(summary.total_cost_usd / summary.total_sessions)
                    : '—'}
                </p>
              </div>
              <div className="parent-view__card">
                <p className="parent-view__card-label">Guardrail Triggers</p>
                {healthStatus === 'ready' && health ? (
                  <>
                    <p className="parent-view__card-value">{health.guardrail_triggered_count}</p>
                    <p className="parent-view__card-subtitle">
                      {formatPercent(health.guardrail_trigger_rate)} of queries
                    </p>
                  </>
                ) : (
                  <p className="parent-view__card-value parent-view__card-value--muted">—</p>
                )}
              </div>
            </div>

            {summary.total_cost_usd === 0 ? (
              <p className="parent-view__empty">No usage data for this period.</p>
            ) : (
              <>
                <div className="parent-view__breakdown">
                  <h2 className="parent-view__breakdown-title">Cost by Category</h2>
                  <ul className="parent-view__bar-list">
                    {Object.entries(summary.cost_by_event_type).map(([eventType, cost]) => (
                      <li key={eventType} className="parent-view__bar-row">
                        <span className="parent-view__bar-label">{EVENT_TYPE_LABELS[eventType] ?? eventType}</span>
                        <span className="parent-view__bar-track">
                          <span
                            className="parent-view__bar-fill"
                            style={{
                              width: `${totalCostByType > 0 ? (cost / totalCostByType) * 100 : 0}%`,
                            }}
                          />
                        </span>
                        <span className="parent-view__bar-value">{formatCurrency(cost)}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="parent-view__breakdown">
                  <h2 className="parent-view__breakdown-title">Cost by Subject</h2>
                  {summary.cost_by_subject.length === 0 ? (
                    <p className="parent-view__empty">No usage data for this period.</p>
                  ) : (
                    <ul className="parent-view__bar-list">
                      {summary.cost_by_subject.map((subject) => (
                        <li key={subject.subject_name} className="parent-view__bar-row">
                          <span className="parent-view__bar-label">{subject.subject_name}</span>
                          <span className="parent-view__bar-track">
                            <span
                              className="parent-view__bar-fill parent-view__bar-fill--accent"
                              style={{
                                width: `${totalCostBySubject > 0 ? (subject.cost_usd / totalCostBySubject) * 100 : 0}%`,
                              }}
                            />
                          </span>
                          <span className="parent-view__bar-value">{formatCurrency(subject.cost_usd)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            )}
          </>
        )}
      </section>

      <section className="parent-view__section" aria-label="Session history">
        <h2 className="parent-view__section-title">Session History</h2>
        {sessionsStatus === 'loading' && <SkeletonRows count={5} />}
        {sessionsStatus !== 'loading' && sessionsStatus !== 'ready' && (
          <SectionError status={sessionsStatus} message={sessionsError} />
        )}
        {sessionsStatus === 'ready' &&
          (sessions.length === 0 ? (
            <p className="parent-view__empty">No usage data for this period.</p>
          ) : (
            <>
              <div className="parent-view__table-wrap">
                <table className="parent-view__table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Subject</th>
                      <th>Turns</th>
                      <th>Cost</th>
                      <th>Session ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.map((session) => (
                      <tr key={session.session_id}>
                        <td data-label="Date">{formatDate(session.created_at)}</td>
                        <td data-label="Subject">{session.subject_name}</td>
                        <td data-label="Turns">{session.turn_count}</td>
                        <td data-label="Cost">{formatCurrency(session.cost_usd)}</td>
                        <td data-label="Session ID" className="parent-view__table-id">
                          {session.session_id.slice(0, 8)}…
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {sessionsHasMore && (
                <button
                  type="button"
                  className="parent-view__show-more"
                  onClick={loadMoreSessions}
                  disabled={sessionsLoadingMore}
                >
                  {sessionsLoadingMore ? 'Loading…' : 'Show more'}
                </button>
              )}
            </>
          ))}
      </section>

      <section className="parent-view__section" aria-label="Retrieval health">
        <h2 className="parent-view__section-title">Retrieval Quality</h2>
        {healthStatus === 'loading' && <SkeletonRows count={3} />}
        {healthStatus !== 'loading' && healthStatus !== 'ready' && (
          <SectionError status={healthStatus} message={healthError} />
        )}
        {healthStatus === 'ready' && health && (
          <>
            {health.total_queries === 0 ? (
              <p className="parent-view__empty">No usage data for this period.</p>
            ) : (
              <>
                <div className="parent-view__cards">
                  <div className="parent-view__card">
                    <p className="parent-view__card-label">Queries</p>
                    <p className="parent-view__card-value">{health.total_queries}</p>
                  </div>
                  <div className="parent-view__card">
                    <p className="parent-view__card-label">Avg Score</p>
                    <p className="parent-view__card-value" style={{ color: scoreColor(health.avg_best_score) }}>
                      {health.avg_best_score === null ? '—' : health.avg_best_score.toFixed(2)}
                    </p>
                  </div>
                  <div className="parent-view__card">
                    <p className="parent-view__card-label">Guardrail Rate</p>
                    <p
                      className="parent-view__card-value"
                      style={{ color: guardrailRateColor(health.guardrail_trigger_rate) }}
                    >
                      {formatPercent(health.guardrail_trigger_rate)}
                    </p>
                  </div>
                </div>

                <div className="parent-view__breakdown">
                  <h3 className="parent-view__breakdown-title">Score Distribution</h3>
                  <div className="parent-view__stacked-bar">
                    {SCORE_BUCKETS.map((bucket) => {
                      const count = health.score_distribution[bucket.key]
                      const width = totalScoreCount > 0 ? (count / totalScoreCount) * 100 : 0
                      if (width === 0) return null
                      return (
                        <span
                          key={bucket.key}
                          className="parent-view__stacked-bar-segment"
                          style={{ width: `${width}%`, background: bucket.color }}
                          title={`${bucket.label}: ${count}`}
                        />
                      )
                    })}
                  </div>
                  <ul className="parent-view__legend">
                    {SCORE_BUCKETS.map((bucket) => (
                      <li key={bucket.key} className="parent-view__legend-item">
                        <span className="parent-view__legend-swatch" style={{ background: bucket.color }} />
                        {bucket.label} — {health.score_distribution[bucket.key]}
                      </li>
                    ))}
                  </ul>
                </div>

                <p className="parent-view__explainer">
                  Score measures how well retrieved textbook passages match the student's questions. "Poor" scores
                  trigger the guardrail, telling the student the topic isn't covered.
                </p>
              </>
            )}
          </>
        )}
      </section>
    </>
  )
}

// Phase 6c of Project Tutor: the full Parent Dashboard, replacing the
// routing/role-gate placeholder from Phase 6b (development-plan.md Phase 6
// exit: "the parent role changes what the API returns, not just what's
// displayed" -- every section here is backed by a parent-only endpoint).
export function ParentView() {
  const [period, setPeriod] = useState<Period>('week')
  const [settings, setSettings] = useState<ParentSettings | null>(null)

  useEffect(() => {
    getParentSettings()
      .then(setSettings)
      .catch(() => {
        // The web-search toggle is a "coming soon" preview -- a failed
        // fetch just leaves it unrendered rather than erroring the page.
      })
  }, [])

  return (
    <div className="parent-view">
      <header className="parent-view__header">
        <div className="parent-view__header-top">
          <h1 className="parent-view__title">Parent Dashboard</h1>
          <Link to="/" className="parent-view__back-link">
            ← Back to Tutoring
          </Link>
        </div>

        <div className="parent-view__header-bottom">
          <div className="parent-view__period" role="radiogroup" aria-label="Time period">
            {PERIOD_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={period === option.value}
                className={`parent-view__period-option ${
                  period === option.value ? 'parent-view__period-option--active' : ''
                }`}
                onClick={() => setPeriod(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>

          {settings && (
            <div className="parent-view__settings-preview">
              <span>Web search</span>
              <label className="parent-view__toggle" title="Coming soon">
                <input type="checkbox" checked={settings.web_search_enabled} disabled readOnly />
                <span className="parent-view__toggle-track" aria-hidden="true" />
              </label>
              <span className="parent-view__settings-preview-note">Coming soon</span>
            </div>
          )}
        </div>
      </header>

      <PeriodDashboard key={period} period={period} />

      <SettingsPanel />
    </div>
  )
}
