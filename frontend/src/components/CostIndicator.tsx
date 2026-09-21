import { useEffect, useState } from 'react'

import { getSessionUsage } from '../api'
import { useAppStore } from '../store'
import './CostIndicator.css'

const STORAGE_KEY = 'project-tutor:cost-indicator-visible'

function readStoredVisibility(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

function formatCost(costUsd: number | null): string {
  return costUsd === null ? '…' : costUsd.toFixed(4)
}

export function CostIndicator() {
  const sessionId = useAppStore((state) => state.sessionId)
  const isLoading = useAppStore((state) => state.isLoading)
  const messageCount = useAppStore((state) => state.messages.length)

  const [isVisible, setIsVisible] = useState(readStoredVisibility)
  const [costUsd, setCostUsd] = useState<number | null>(null)

  useEffect(() => {
    if (!sessionId || !isVisible || isLoading) return

    let cancelled = false
    getSessionUsage(sessionId)
      .then((usage) => {
        if (!cancelled) setCostUsd(usage.total_cost_usd)
      })
      .catch(() => {
        // Cost is a nice-to-have -- a failed fetch just leaves the last known value.
      })

    return () => {
      cancelled = true
    }
    // messageCount ties this to "after each chat response" without polling.
  }, [sessionId, isVisible, isLoading, messageCount])

  const toggleVisible = () => {
    setIsVisible((previous) => {
      const next = !previous
      try {
        localStorage.setItem(STORAGE_KEY, String(next))
      } catch {
        // Per-viewer convenience only -- losing the persisted preference isn't fatal.
      }
      return next
    })
  }

  if (!sessionId) return null

  return (
    <button
      type="button"
      className="cost-indicator"
      onClick={toggleVisible}
      aria-pressed={isVisible}
      aria-label={isVisible ? 'Hide session cost' : 'Show session cost'}
      title={isVisible ? 'Hide session cost' : 'Show session cost'}
    >
      <span aria-hidden="true">$</span>
      {isVisible && <span className="cost-indicator__value">{formatCost(costUsd)}</span>}
    </button>
  )
}
