// The Visual Companion -- a side panel (desktop), bottom drawer (tablet), or
// overlay (mobile) that renders the active chat message's visual directives
// (architecture.md: "a dumb renderer... driven entirely by directives from
// Orchestration"). "Active" is the latest assistant message by default, or
// whichever message the user clicked in ChatView (store.activeMessageIndex).

import katex from 'katex'
import { useEffect, useState } from 'react'
import 'katex/dist/katex.min.css'

import type { SourceCitation } from '../api'
import { useAppStore } from '../store'
import './VisualCompanion.css'

function renderFormula(content: string): string | null {
  try {
    return katex.renderToString(content, { throwOnError: false, displayMode: true })
  } catch {
    return null
  }
}

function FormulaCard({ content }: { content: string }) {
  const html = renderFormula(content)
  return (
    <div className="visual-companion__card visual-companion__card--formula">
      {html === null ? (
        <code className="visual-companion__formula-fallback">{content}</code>
      ) : (
        <div className="visual-companion__formula" dangerouslySetInnerHTML={{ __html: html }} />
      )}
    </div>
  )
}

export function VisualCompanion() {
  const messages = useAppStore((state) => state.messages)
  const activeMessageIndex = useAppStore((state) => state.activeMessageIndex)

  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const [isHighlighted, setIsHighlighted] = useState(false)
  // The most recent "new directives just arrived" signature we've already
  // reacted to -- lets the expand/pulse trigger be derived during render
  // (React's documented alternative to setState-in-effect for "adjust state
  // when something changes") instead of needing an effect just to detect it.
  const [seenSignature, setSeenSignature] = useState<string | null>(null)

  let lastAssistantIndex: number | null = null
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant') {
      lastAssistantIndex = i
      break
    }
  }

  const effectiveIndex = activeMessageIndex ?? lastAssistantIndex
  const activeMessage = effectiveIndex !== null ? messages[effectiveIndex] : null
  const directives = activeMessage?.visual_directives ?? []
  const sources: SourceCitation[] = activeMessage?.sources ?? []

  const formulas = directives.filter((d) => d.directive_type === 'formula')
  const keyTerms = directives.filter((d) => d.directive_type === 'highlight')
  const isSupplemented = directives.some((d) => d.directive_type === 'supplemented')
  const hasContent = formulas.length > 0 || keyTerms.length > 0 || sources.length > 0 || isSupplemented

  // Only auto-expand/pulse when following the latest message -- clicking an
  // older one in ChatView shouldn't yank the panel open again.
  const signature = activeMessageIndex === null && hasContent ? `${effectiveIndex}:${messages.length}` : null
  if (signature !== null && signature !== seenSignature) {
    setSeenSignature(signature)
    setIsCollapsed(false)
    setIsHighlighted(true)
  }

  useEffect(() => {
    if (!isHighlighted) return
    const timeout = setTimeout(() => setIsHighlighted(false), 1500)
    return () => clearTimeout(timeout)
  }, [isHighlighted])

  const panelClassName = [
    'visual-companion',
    isCollapsed ? 'visual-companion--collapsed' : '',
    isMobileOpen ? 'visual-companion--mobile-open' : '',
    isHighlighted ? 'visual-companion--highlight' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <>
      <aside className={panelClassName} aria-label="Study Notes">
        <div className="visual-companion__header">
          <span className="visual-companion__header-title">Study Notes</span>
          <button
            type="button"
            className="visual-companion__toggle"
            onClick={() => setIsCollapsed((collapsed) => !collapsed)}
            aria-label={isCollapsed ? 'Expand Study Notes' : 'Collapse Study Notes'}
          >
            <span aria-hidden="true">{isCollapsed ? '⟨' : '⟩'}</span>
          </button>
          <button
            type="button"
            className="visual-companion__mobile-close"
            onClick={() => setIsMobileOpen(false)}
            aria-label="Close Study Notes"
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="visual-companion__body">
          {!hasContent ? (
            <p className="visual-companion__empty">Study notes will appear here when relevant.</p>
          ) : (
            <>
              {formulas.length > 0 && (
                <section className="visual-companion__section">
                  <h4 className="visual-companion__section-title">📐 Formulas</h4>
                  {formulas.map((formula, index) => (
                    <FormulaCard key={index} content={formula.content} />
                  ))}
                </section>
              )}

              {keyTerms.length > 0 && (
                <section className="visual-companion__section">
                  <h4 className="visual-companion__section-title">📝 Key Terms</h4>
                  <ul className="visual-companion__terms">
                    {keyTerms.map((term, index) => (
                      <li key={index}>
                        <strong>{term.content}</strong>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {sources.length > 0 && (
                <section className="visual-companion__section">
                  <h4 className="visual-companion__section-title">📖 Sources</h4>
                  <div className="visual-companion__sources">
                    {sources.map((source, index) => (
                      <span key={index} className="visual-companion__page-badge">
                        Page {source.page_number ?? '?'}
                      </span>
                    ))}
                  </div>
                </section>
              )}

              {isSupplemented && (
                <section className="visual-companion__section">
                  <h4 className="visual-companion__section-title">💡 Beyond the Textbook</h4>
                  <p className="visual-companion__supplemented-note">
                    This response includes an example from outside the textbook.
                  </p>
                </section>
              )}
            </>
          )}
        </div>
      </aside>

      {isMobileOpen && (
        <button
          type="button"
          className="visual-companion__backdrop"
          aria-label="Close Study Notes"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      <button
        type="button"
        className={`visual-companion__mobile-toggle ${isHighlighted ? 'visual-companion__mobile-toggle--pulse' : ''}`}
        onClick={() => setIsMobileOpen((open) => !open)}
        aria-label="Toggle Study Notes"
      >
        <span aria-hidden="true">𝑓(x)</span>
      </button>
    </>
  )
}
