import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DiagramState } from '../store'

const renderMermaidDiagramMock = vi.fn<(id: string, source: string, theme: string) => Promise<string>>()
vi.mock('../diagram', () => ({
  renderMermaidDiagram: (id: string, source: string, theme: string) => renderMermaidDiagramMock(id, source, theme),
  parseMermaidDiagram: vi.fn(),
}))

const { useAppStore } = await import('../store')
const { VisualsTab } = await import('./VisualsTab')

function renderDiagramCard(diagramState: DiagramState | undefined, canDrawDiagram = true) {
  return render(
    <VisualsTab
      tables={[]}
      formulas={[]}
      pageView={null}
      onNavigatePage={() => {}}
      diagramState={diagramState}
      canDrawDiagram={canDrawDiagram}
      onDrawDiagram={() => {}}
      hasCitedPage={false}
      onViewCitedPage={() => {}}
      diagramMessageKey="0"
    />,
  )
}

describe('DiagramCard (via VisualsTab)', () => {
  beforeEach(() => {
    renderMermaidDiagramMock.mockReset()
    useAppStore.setState({ theme: 'light', isPageLightboxOpen: false })
  })

  afterEach(() => {
    cleanup()
  })

  it('shows a Draw diagram button when idle and eligible', () => {
    renderDiagramCard(undefined, true)

    expect(screen.getByRole('button', { name: /draw diagram/i })).toBeTruthy()
  })

  it('renders nothing when idle and not eligible', () => {
    const { container } = renderDiagramCard(undefined, false)

    expect(container.querySelector('.visual-companion__card--diagram')).toBeNull()
    expect(screen.getByText('No visuals for this answer.')).toBeTruthy()
  })

  it('shows a loading skeleton and status text while loading', () => {
    renderDiagramCard({ status: 'loading' })

    expect(screen.getByText(/drawing/i)).toBeTruthy()
  })

  it('shows the not_applicable message', () => {
    renderDiagramCard({ status: 'not_applicable' })

    expect(screen.getByText(/doesn't describe a process/i)).toBeTruthy()
  })

  it('shows the failed message with a "View textbook page" fallback when a page is cited', () => {
    render(
      <VisualsTab
        tables={[]}
        formulas={[]}
        pageView={null}
        onNavigatePage={() => {}}
        diagramState={{ status: 'failed' }}
        canDrawDiagram={true}
        onDrawDiagram={() => {}}
        hasCitedPage={true}
        onViewCitedPage={() => {}}
        diagramMessageKey="0"
      />,
    )

    expect(screen.getByText(/couldn't draw this one/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /view textbook page/i })).toBeTruthy()
  })

  it('renders the SVG and a Redraw link when ready', async () => {
    renderMermaidDiagramMock.mockResolvedValue('<svg><text>diagram</text></svg>')

    renderDiagramCard({ status: 'ready', mermaid: 'flowchart TD\nA-->B' })

    expect(await screen.findByLabelText(/enlarge diagram/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /redraw/i })).toBeTruthy()
    expect(screen.getByText('AI-drawn from this answer')).toBeTruthy()
  })

  it('calls onDrawDiagram with force=true when Redraw is clicked', async () => {
    renderMermaidDiagramMock.mockResolvedValue('<svg></svg>')
    const onDrawDiagram = vi.fn()

    render(
      <VisualsTab
        tables={[]}
        formulas={[]}
        pageView={null}
        onNavigatePage={() => {}}
        diagramState={{ status: 'ready', mermaid: 'flowchart TD\nA-->B' }}
        canDrawDiagram={true}
        onDrawDiagram={onDrawDiagram}
        hasCitedPage={false}
        onViewCitedPage={() => {}}
        diagramMessageKey="0"
      />,
    )

    const redraw = await screen.findByRole('button', { name: /redraw/i })
    fireEvent.click(redraw)

    expect(onDrawDiagram).toHaveBeenCalledWith(true)
  })
})
