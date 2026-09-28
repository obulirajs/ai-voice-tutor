// The Visual Companion's "Visuals" tab body -- a card grid of the active
// message's table/formula directives (architecture.md: the companion is a
// dumb renderer driven by directives, this just picks the right card per
// directive_type). Kept as its own component so VisualCompanion.tsx can
// stay the panel shell (collapse/mobile/highlight behaviour) per the
// tabbed-panel redesign.

import katex from 'katex'
import { useEffect, useId, useState } from 'react'
import 'katex/dist/katex.min.css'

import type { VisualDirective } from '../api'
import { pageImageUrl } from '../api'
import { renderMermaidDiagram } from '../diagram'
import type { DiagramState, PageViewRequest } from '../store'
import { useAppStore } from '../store'

interface TablePayload {
  headers: string[]
  rows: string[][]
}

function parseTablePayload(content: string): TablePayload | null {
  try {
    const parsed: unknown = JSON.parse(content)
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      Array.isArray((parsed as TablePayload).headers) &&
      Array.isArray((parsed as TablePayload).rows)
    ) {
      return parsed as TablePayload
    }
    return null
  } catch {
    return null
  }
}

function TableCard({ directive }: { directive: VisualDirective }) {
  const table = parseTablePayload(directive.content)
  if (table === null) return null

  return (
    <div className="visual-companion__card visual-companion__card--table visual-companion__card--wide">
      <h5 className="visual-companion__card-heading">{directive.label ?? 'Table'}</h5>
      <div className="visual-companion__table-scroll">
        <table className="visual-companion__table">
          <thead>
            <tr>
              {table.headers.map((header, index) => (
                <th key={index}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

type PageImageStatus = 'loading' | 'loaded' | 'error'

interface TextbookPageCardProps {
  documentId: number
  pageNumber: number
  onNavigate: (documentId: number, pageNumber: number) => void
}

function TextbookPageCard({ documentId, pageNumber, onNavigate }: TextbookPageCardProps) {
  const imageUrl = pageImageUrl(documentId, pageNumber)
  const [status, setStatus] = useState<PageImageStatus>('loading')
  // Lifted into the store (rather than local state) so VoiceButton's Esc
  // handler can check it and defer to the lightbox instead of also
  // interrupting voice on the same keypress.
  const isLightboxOpen = useAppStore((state) => state.isPageLightboxOpen)
  const setIsLightboxOpen = useAppStore((state) => state.setPageLightboxOpen)
  // The most recent imageUrl we've already reset `status` for -- lets a new
  // page (prev/next, or a different source badge) reset to the loading
  // state derived during render, the same pattern VisualCompanion.tsx uses
  // for its own "react once per change" state, rather than a
  // setState-in-effect. Without this, navigating from an errored page
  // straight to a working one would keep showing the old error state.
  const [statusForUrl, setStatusForUrl] = useState(imageUrl)
  if (imageUrl !== statusForUrl) {
    setStatusForUrl(imageUrl)
    setStatus('loading')
  }

  useEffect(() => {
    if (!isLightboxOpen) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsLightboxOpen(false)
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isLightboxOpen, setIsLightboxOpen])

  // Unmounting (tab/message switch, or navigating away entirely) must not
  // leave the store's isPageLightboxOpen stuck true -- that would make
  // VoiceButton's Esc handler defer to a lightbox that's no longer on screen.
  useEffect(() => () => setIsLightboxOpen(false), [setIsLightboxOpen])

  return (
    <div className="visual-companion__card visual-companion__card--page visual-companion__card--wide">
      <h5 className="visual-companion__card-heading">Textbook page</h5>

      <div className="visual-companion__page-viewer">
        {status === 'loading' && <div className="visual-companion__page-skeleton" aria-hidden="true" />}
        {status === 'error' ? (
          <p className="visual-companion__page-error">
            This page isn't available — re-upload the textbook to enable page view.
          </p>
        ) : (
          <img
            src={imageUrl}
            alt={`Textbook page ${pageNumber}`}
            className="visual-companion__page-image"
            style={{ display: status === 'loaded' ? 'block' : 'none' }}
            onLoad={() => setStatus('loaded')}
            onError={() => setStatus('error')}
            onClick={() => setIsLightboxOpen(true)}
          />
        )}
      </div>

      <div className="visual-companion__page-footer">
        <span className="visual-companion__page-caption">Textbook · PDF page {pageNumber}</span>
        <div className="visual-companion__page-nav">
          {pageNumber > 1 && (
            <button type="button" onClick={() => onNavigate(documentId, pageNumber - 1)}>
              ‹ Prev
            </button>
          )}
          <button type="button" onClick={() => onNavigate(documentId, pageNumber + 1)}>
            Next ›
          </button>
        </div>
      </div>

      {isLightboxOpen && status === 'loaded' && (
        <button
          type="button"
          className="visual-companion__lightbox"
          aria-label="Close enlarged page"
          onClick={() => setIsLightboxOpen(false)}
        >
          <img src={imageUrl} alt={`Textbook page ${pageNumber}, enlarged`} className="visual-companion__lightbox-image" />
        </button>
      )}
    </div>
  )
}

function renderFormula(content: string): string | null {
  try {
    return katex.renderToString(content, { throwOnError: false, displayMode: true })
  } catch {
    return null
  }
}

function FormulaCard({ directive }: { directive: VisualDirective }) {
  const html = renderFormula(directive.content)
  return (
    <div className="visual-companion__card visual-companion__card--formula">
      <h5 className="visual-companion__card-heading">{directive.label ?? 'Formula'}</h5>
      {html === null ? (
        <code className="visual-companion__formula-fallback">{directive.content}</code>
      ) : (
        <div className="visual-companion__formula" dangerouslySetInnerHTML={{ __html: html }} />
      )}
    </div>
  )
}

interface DiagramCardProps {
  state: DiagramState | undefined
  canDraw: boolean
  onDraw: (force?: boolean) => void
  hasCitedPage: boolean
  onViewCitedPage: () => void
}

// Keyed by the active message (see VisualsTab's diagramMessageKey prop) so
// this remounts fresh whenever the student switches which message is
// active -- no stale svg/lightbox state to carry across messages, and
// mermaid.render()'s DOM id (useId()) stays unique per mounted card.
function DiagramCard({ state, canDraw, onDraw, hasCitedPage, onViewCitedPage }: DiagramCardProps) {
  const theme = useAppStore((s) => s.theme)
  const isLightboxOpen = useAppStore((s) => s.isPageLightboxOpen)
  const setIsLightboxOpen = useAppStore((s) => s.setPageLightboxOpen)

  const diagramId = `companion-diagram-${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const [svg, setSvg] = useState<string | null>(null)

  const mermaidSource = state?.status === 'ready' ? state.mermaid : undefined

  useEffect(() => {
    if (!mermaidSource) return
    let cancelled = false
    renderMermaidDiagram(diagramId, mermaidSource, theme)
      .then((rendered) => {
        if (!cancelled) setSvg(rendered)
      })
      .catch(() => {
        if (!cancelled) setSvg(null)
      })
    return () => {
      cancelled = true
    }
  }, [mermaidSource, theme, diagramId])

  useEffect(() => {
    if (!isLightboxOpen) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsLightboxOpen(false)
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isLightboxOpen, setIsLightboxOpen])

  useEffect(() => () => setIsLightboxOpen(false), [setIsLightboxOpen])

  const status = state?.status ?? 'idle'

  if (status === 'idle') {
    if (!canDraw) return null
    return (
      <div className="visual-companion__card visual-companion__card--diagram visual-companion__card--wide">
        <button type="button" className="visual-companion__diagram-button" onClick={() => onDraw()}>
          🧩 Draw diagram
        </button>
      </div>
    )
  }

  if (status === 'loading') {
    return (
      <div className="visual-companion__card visual-companion__card--diagram visual-companion__card--wide">
        <div className="visual-companion__page-skeleton" aria-hidden="true" />
        <p className="visual-companion__diagram-status">Drawing… (first time can take ~20s)</p>
      </div>
    )
  }

  if (status === 'not_applicable') {
    return (
      <div className="visual-companion__card visual-companion__card--diagram visual-companion__card--wide">
        <p className="visual-companion__empty">This answer doesn't describe a process to draw.</p>
      </div>
    )
  }

  if (status === 'failed') {
    return (
      <div className="visual-companion__card visual-companion__card--diagram visual-companion__card--wide">
        <p className="visual-companion__empty">Couldn't draw this one.</p>
        {hasCitedPage && (
          <button type="button" className="visual-companion__diagram-view-page" onClick={onViewCitedPage}>
            View textbook page
          </button>
        )}
      </div>
    )
  }

  // status === 'ready'
  return (
    <div className="visual-companion__card visual-companion__card--diagram visual-companion__card--wide">
      <h5 className="visual-companion__card-heading">Diagram</h5>
      {svg === null ? (
        <div className="visual-companion__page-skeleton" aria-hidden="true" />
      ) : (
        <button
          type="button"
          className="visual-companion__diagram-svg-button"
          onClick={() => setIsLightboxOpen(true)}
          aria-label="Enlarge diagram"
        >
          <div className="visual-companion__diagram-svg" dangerouslySetInnerHTML={{ __html: svg }} />
        </button>
      )}
      <div className="visual-companion__page-footer">
        <span className="visual-companion__page-caption">AI-drawn from this answer</span>
        <button type="button" className="visual-companion__diagram-redraw" onClick={() => onDraw(true)}>
          Redraw
        </button>
      </div>
      {isLightboxOpen && svg !== null && (
        <button
          type="button"
          className="visual-companion__lightbox"
          aria-label="Close enlarged diagram"
          onClick={() => setIsLightboxOpen(false)}
        >
          <div className="visual-companion__lightbox-diagram" dangerouslySetInnerHTML={{ __html: svg }} />
        </button>
      )}
    </div>
  )
}

interface VisualsTabProps {
  tables: VisualDirective[]
  formulas: VisualDirective[]
  pageView: PageViewRequest | null
  onNavigatePage: (documentId: number, pageNumber: number) => void
  diagramState: DiagramState | undefined
  canDrawDiagram: boolean
  onDrawDiagram: (force?: boolean) => void
  hasCitedPage: boolean
  onViewCitedPage: () => void
  // A stable-per-active-message key (VisualCompanion's messageKey) --
  // DiagramCard is keyed by this so it remounts fresh (fresh svg, fresh
  // lightbox state) whenever the active message changes.
  diagramMessageKey: string
}

export function VisualsTab({
  tables,
  formulas,
  pageView,
  onNavigatePage,
  diagramState,
  canDrawDiagram,
  onDrawDiagram,
  hasCitedPage,
  onViewCitedPage,
  diagramMessageKey,
}: VisualsTabProps) {
  const showDiagramCard = canDrawDiagram || diagramState !== undefined

  if (tables.length === 0 && formulas.length === 0 && pageView === null && !showDiagramCard) {
    return <p className="visual-companion__empty">No visuals for this answer.</p>
  }

  return (
    <div className="visual-companion__grid">
      {pageView && (
        <TextbookPageCard
          documentId={pageView.documentId}
          pageNumber={pageView.pageNumber}
          onNavigate={onNavigatePage}
        />
      )}
      {showDiagramCard && (
        <DiagramCard
          key={diagramMessageKey}
          state={diagramState}
          canDraw={canDrawDiagram}
          onDraw={onDrawDiagram}
          hasCitedPage={hasCitedPage}
          onViewCitedPage={onViewCitedPage}
        />
      )}
      {tables.map((table, index) => (
        <TableCard key={`table-${index}`} directive={table} />
      ))}
      {formulas.map((formula, index) => (
        <FormulaCard key={`formula-${index}`} directive={formula} />
      ))}
    </div>
  )
}
