import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionListItem } from '../api'
import { ThemeProvider } from '../theme/ThemeContext'

const getSessionListMock = vi.fn<
  (period: string, limit: number, offset: number, sortBy: string, sortDir: string) => Promise<SessionListItem[]>
>()

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return {
    ...actual,
    getParentSettings: vi.fn().mockResolvedValue({ web_search_enabled: false }),
    getRetrievalHealth: vi.fn().mockResolvedValue({
      total_queries: 0,
      guardrail_triggered_count: 0,
      guardrail_trigger_rate: 0,
      avg_best_score: null,
      score_distribution: { excellent: 0, good: 0, fair: 0, poor: 0 },
    }),
    getSessionList: (...args: Parameters<typeof getSessionListMock>) => getSessionListMock(...args),
    getUsageSummary: vi.fn().mockResolvedValue({
      total_cost_usd: 0,
      total_sessions: 0,
      total_turns: 0,
      cost_by_event_type: {},
      cost_by_subject: [],
    }),
  }
})

const { ParentView } = await import('./ParentView')

function makePage(limit: number, offset: number): SessionListItem[] {
  return Array.from({ length: limit }, (_, i) => ({
    session_id: `sess${offset + i}`,
    subject_name: 'French',
    created_at: new Date(2026, 0, 1 + offset + i).toISOString(),
    cost_usd: 0.01 * (offset + i + 1),
    turn_count: offset + i + 1,
    event_count: offset + i + 1,
  }))
}

function renderParentView() {
  return render(
    <MemoryRouter>
      <ThemeProvider>
        <ParentView />
      </ThemeProvider>
    </MemoryRouter>,
  )
}

describe('ParentView session history table', () => {
  beforeEach(() => {
    getSessionListMock.mockReset()
    getSessionListMock.mockImplementation((_period, limit, offset) =>
      Promise.resolve(makePage(limit, offset)),
    )
  })

  afterEach(() => {
    cleanup()
  })

  it('loads the first page sorted by date descending by default', async () => {
    renderParentView()

    await waitFor(() => expect(getSessionListMock).toHaveBeenCalled())

    expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 0, 'created_at', 'desc')
    expect(await screen.findByText('sess0…')).toBeTruthy()
  })

  it('does not make the Session ID column sortable', async () => {
    renderParentView()
    await screen.findByText('sess0…')

    const headerRow = screen.getAllByRole('row')[0]
    expect(within(headerRow).getByText('Session ID')).toBeTruthy()
    expect(within(headerRow).queryByRole('button', { name: /session id/i })).toBeNull()
  })

  it('toggles sort direction when clicking the active column twice', async () => {
    renderParentView()
    await screen.findByText('sess0…')

    fireEvent.click(screen.getByRole('button', { name: /date/i }))
    await waitFor(() =>
      expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 0, 'created_at', 'asc'),
    )

    fireEvent.click(screen.getByRole('button', { name: /date/i }))
    await waitFor(() =>
      expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 0, 'created_at', 'desc'),
    )
  })

  it('sorts ascending by default when switching to a new column, and resets to page 1', async () => {
    renderParentView()
    await screen.findByText('sess0…')

    // Move to page 2 first, then switch columns -- the new sort should
    // land back on page 1, not stay on the old page's offset.
    fireEvent.click(screen.getByRole('button', { name: /next/i }))
    await waitFor(() => expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 20, 'created_at', 'desc'))

    fireEvent.click(screen.getByRole('button', { name: /cost/i }))
    await waitFor(() =>
      expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 0, 'cost_usd', 'asc'),
    )
  })

  it('paginates with Previous/Next, disabling Previous on the first page', async () => {
    renderParentView()
    await screen.findByText('sess0…')

    expect((screen.getByRole('button', { name: /previous/i }) as HTMLButtonElement).disabled).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: /next/i }))
    await waitFor(() => expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 20, 'created_at', 'desc'))
    expect(await screen.findByText('sess20…')).toBeTruthy()
    expect((screen.getByRole('button', { name: /previous/i }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /previous/i }))
    await waitFor(() => expect(getSessionListMock).toHaveBeenLastCalledWith('week', 20, 0, 'created_at', 'desc'))
    expect(await screen.findByText('sess0…')).toBeTruthy()
  })
})
