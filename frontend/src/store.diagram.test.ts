import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DiagramResponse } from './api'

const requestDiagramMock = vi.fn<(payload: unknown, signal?: AbortSignal) => Promise<DiagramResponse>>()
vi.mock('./api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api')>()
  return { ...actual, requestDiagram: (...args: [unknown, AbortSignal?]) => requestDiagramMock(...args) }
})

const parseMermaidDiagramMock = vi.fn<(source: string) => Promise<string | null>>()
vi.mock('./diagram', () => ({
  parseMermaidDiagram: (source: string) => parseMermaidDiagramMock(source),
  renderMermaidDiagram: vi.fn(),
}))

const { useAppStore } = await import('./store')

describe('drawDiagram', () => {
  beforeEach(() => {
    requestDiagramMock.mockReset()
    parseMermaidDiagramMock.mockReset()
    useAppStore.setState({
      messages: [{ role: 'assistant', content: 'Water evaporates, then condenses, then falls as rain.' }],
      sessionId: 'session-1',
      diagramsByMessageIndex: {},
    })
  })

  it('retries once on a parse error, then marks the diagram failed', async () => {
    requestDiagramMock.mockResolvedValue({ mermaid: 'flowchart TD\nA-->B', reason: null })
    parseMermaidDiagramMock.mockResolvedValue('Unexpected token at line 2')

    await useAppStore.getState().drawDiagram(0)

    expect(requestDiagramMock).toHaveBeenCalledTimes(2)
    const secondCallPayload = requestDiagramMock.mock.calls[1][0] as { previousError: string | null }
    expect(secondCallPayload.previousError).toBe('Unexpected token at line 2')

    const state = useAppStore.getState().diagramsByMessageIndex[0]
    expect(state.status).toBe('failed')
  })

  it('succeeds on the retry after the first parse error', async () => {
    requestDiagramMock.mockResolvedValue({ mermaid: 'flowchart TD\nA-->B', reason: null })
    parseMermaidDiagramMock.mockResolvedValueOnce('Unexpected token').mockResolvedValueOnce(null)

    await useAppStore.getState().drawDiagram(0)

    expect(requestDiagramMock).toHaveBeenCalledTimes(2)
    const state = useAppStore.getState().diagramsByMessageIndex[0]
    expect(state.status).toBe('ready')
    expect(state.mermaid).toBe('flowchart TD\nA-->B')
  })

  it('a cached (ready) diagram is not re-requested', async () => {
    requestDiagramMock.mockResolvedValue({ mermaid: 'flowchart TD\nA-->B', reason: null })
    parseMermaidDiagramMock.mockResolvedValue(null)

    await useAppStore.getState().drawDiagram(0)
    expect(requestDiagramMock).toHaveBeenCalledTimes(1)

    await useAppStore.getState().drawDiagram(0)
    expect(requestDiagramMock).toHaveBeenCalledTimes(1)
  })

  it('force=true re-requests even when a ready diagram is already cached', async () => {
    requestDiagramMock.mockResolvedValue({ mermaid: 'flowchart TD\nA-->B', reason: null })
    parseMermaidDiagramMock.mockResolvedValue(null)

    await useAppStore.getState().drawDiagram(0)
    expect(requestDiagramMock).toHaveBeenCalledTimes(1)

    await useAppStore.getState().drawDiagram(0, true)
    expect(requestDiagramMock).toHaveBeenCalledTimes(2)
  })

  it('reports not_applicable without retrying when the model returns NONE', async () => {
    requestDiagramMock.mockResolvedValue({ mermaid: null, reason: 'not_applicable' })

    await useAppStore.getState().drawDiagram(0)

    expect(requestDiagramMock).toHaveBeenCalledTimes(1)
    expect(parseMermaidDiagramMock).not.toHaveBeenCalled()
    expect(useAppStore.getState().diagramsByMessageIndex[0].status).toBe('not_applicable')
  })
})
