// The Visual Companion -- a side panel (desktop), bottom drawer (tablet), or
// overlay (mobile) that renders the active chat message's visual directives
// (architecture.md: "a dumb renderer... driven entirely by directives from
// Orchestration"). "Active" is the latest assistant message by default, or
// whichever message the user clicked in ChatView (store.activeMessageIndex).
//
// This file is the panel shell only (collapse, mobile overlay, highlight
// pulse, the tab bar) -- each tab's body is its own component (VisualsTab,
// StudyNotesTab) so this file stays readable.

import type { KeyboardEvent } from 'react'
import { useEffect, useState } from 'react'

import type { SourceCitation } from '../api'
import { useAppStore } from '../store'
import { StudyNotesTab } from './StudyNotesTab'
import { VisualsTab } from './VisualsTab'
import './VisualCompanion.css'

type CompanionTab = 'visuals' | 'notes'

export function VisualCompanion() {
  const messages = useAppStore((state) => state.messages)
  const activeMessageIndex = useAppStore((state) => state.activeMessageIndex)
  const pageView = useAppStore((state) => state.pageView)
  const viewTextbookPage = useAppStore((state) => state.viewTextbookPage)
  const clearPageView = useAppStore((state) => state.clearPageView)
  const diagramsByMessageIndex = useAppStore((state) => state.diagramsByMessageIndex)
  const diagramVoiceRequest = useAppStore((state) => state.diagramVoiceRequest)
  const drawDiagram = useAppStore((state) => state.drawDiagram)

  const [isCollapsed, setIsCollapsed] = useState(true)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const [isHighlighted, setIsHighlighted] = useState(false)
  // The most recent "new directives just arrived" signature we've already
  // reacted to -- lets the expand/pulse trigger be derived during render
  // (React's documented alternative to setState-in-effect for "adjust state
  // when something changes") instead of needing an effect just to detect it.
  const [seenSignature, setSeenSignature] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<CompanionTab>('visuals')
  const [lastMessageKey, setLastMessageKey] = useState<string | null>(null)
  // The most recent pageView.requestId we've already switched tabs for --
  // same "react once per change" pattern as seenSignature, so a manual
  // switch back to Study Notes afterward isn't immediately undone.
  const [seenPageViewRequestId, setSeenPageViewRequestId] = useState<number | null>(null)
  // Same pattern again, for a voice "draw a diagram" command -- it doesn't
  // click anything, so nothing else would open/switch the panel for it.
  const [seenDiagramVoiceRequestId, setSeenDiagramVoiceRequestId] = useState<number | null>(null)

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

  const tables = directives.filter((d) => d.directive_type === 'table')
  const formulas = directives.filter((d) => d.directive_type === 'formula')
  const keyTerms = directives.filter((d) => d.directive_type === 'highlight')
  const isSupplemented = directives.some((d) => d.directive_type === 'supplemented')

  // A pageView only belongs to the Visuals tab while it's still the active
  // message's -- the message-change branch below clears it, but effectiveIndex
  // can change the same render clearPageView's state update lands, so this
  // check keeps a stale pageView from flashing against the new message too.
  const activePageView = pageView !== null && pageView.messageIndex === effectiveIndex ? pageView : null

  const diagramState = effectiveIndex !== null ? diagramsByMessageIndex[effectiveIndex] : undefined
  // A "Draw diagram" button only makes sense on a real, cited answer -- a
  // voice command's own reply has no sources/visual_directives at all
  // (undefined, not just empty -- see store.ts's sendVoiceAudio), and both
  // guardrail-refusal replies cite nothing either, so "has at least one
  // source" is a simple, reliable-enough stand-in for "this is a real
  // grounded answer" without needing a dedicated backend flag.
  const canDrawDiagram =
    activeMessage !== null &&
    activeMessage.role === 'assistant' &&
    activeMessage.sources !== undefined &&
    activeMessage.sources.length > 0

  // Only a diagram that's actually ready counts as Visuals content for the
  // tab auto-select below -- an idle button alone shouldn't force-switch
  // tabs (per the task brief), only a diagram that's finished rendering.
  const hasVisualsContent =
    tables.length > 0 || formulas.length > 0 || activePageView !== null || diagramState?.status === 'ready'
  const hasNotesContent = keyTerms.length > 0 || sources.length > 0 || isSupplemented
  const hasContent = hasVisualsContent || hasNotesContent

  // Auto-select whichever tab has content when the active message changes;
  // a manual pick (selectTab, below) isn't touched again until then, since
  // this only re-runs when messageKey itself changes. Derived during
  // render, same pattern as the highlight-pulse signature below.
  const messageKey = effectiveIndex === null ? 'none' : String(effectiveIndex)
  if (messageKey !== lastMessageKey) {
    setLastMessageKey(messageKey)
    setActiveTab(hasVisualsContent ? 'visuals' : 'notes')
    if (pageView !== null) clearPageView()
  }

  // A fresh "Page N" badge click switches to Visuals and opens the panel
  // (it may be collapsed, or hidden behind the mobile toggle) exactly once,
  // even if the same page is clicked again -- see PageViewRequest's requestId.
  if (activePageView !== null && activePageView.requestId !== seenPageViewRequestId) {
    setSeenPageViewRequestId(activePageView.requestId)
    setActiveTab('visuals')
    setIsCollapsed(false)
    setIsMobileOpen(true)
  }

  // A voice "draw a diagram" command switches to Visuals and opens the
  // panel exactly once per command, same as a page-badge click above --
  // see store.ts's sendVoiceAudio, which also points activeMessageIndex at
  // the diagrammed message so effectiveIndex lines up with it here.
  if (diagramVoiceRequest !== null && diagramVoiceRequest.requestId !== seenDiagramVoiceRequestId) {
    setSeenDiagramVoiceRequestId(diagramVoiceRequest.requestId)
    setActiveTab('visuals')
    setIsCollapsed(false)
    setIsMobileOpen(true)
  }

  const selectTab = (tab: CompanionTab) => setActiveTab(tab)

  const handleNavigatePage = (documentId: number, pageNumber: number) => {
    if (effectiveIndex === null) return
    viewTextbookPage(effectiveIndex, documentId, pageNumber)
  }

  const handleDrawDiagram = (force?: boolean) => {
    if (effectiveIndex === null) return
    void drawDiagram(effectiveIndex, force)
  }

  const handleTabsKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    selectTab(activeTab === 'visuals' ? 'notes' : 'visuals')
  }

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
      <aside className={panelClassName} aria-label="Visual Companion">
        <div className="visual-companion__header">
          <div className="visual-companion__header-titles">
            <span className="visual-companion__header-title">Visual Companion</span>
            <span className="visual-companion__header-subtitle">linked to current turn</span>
          </div>
          <button
            type="button"
            className="visual-companion__toggle"
            onClick={() => setIsCollapsed((collapsed) => !collapsed)}
            aria-label={isCollapsed ? 'Expand Visual Companion' : 'Collapse Visual Companion'}
          >
            <span aria-hidden="true">{isCollapsed ? '⟨' : '⟩'}</span>
          </button>
          <button
            type="button"
            className="visual-companion__mobile-close"
            onClick={() => setIsMobileOpen(false)}
            aria-label="Close Visual Companion"
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="visual-companion__body">
          <div
            className="visual-companion__tabs"
            role="tablist"
            aria-label="Visual Companion sections"
            onKeyDown={handleTabsKeyDown}
          >
            <button
              type="button"
              role="tab"
              id="visual-companion-tab-visuals"
              aria-selected={activeTab === 'visuals'}
              aria-controls="visual-companion-panel-visuals"
              tabIndex={activeTab === 'visuals' ? 0 : -1}
              className={[
                'visual-companion__tab',
                activeTab === 'visuals' ? 'visual-companion__tab--active' : '',
                hasVisualsContent ? '' : 'visual-companion__tab--empty',
              ]
                .filter(Boolean)
                .join(' ')}
              onClick={() => selectTab('visuals')}
            >
              Visuals
            </button>
            <button
              type="button"
              role="tab"
              id="visual-companion-tab-notes"
              aria-selected={activeTab === 'notes'}
              aria-controls="visual-companion-panel-notes"
              tabIndex={activeTab === 'notes' ? 0 : -1}
              className={[
                'visual-companion__tab',
                activeTab === 'notes' ? 'visual-companion__tab--active' : '',
                hasNotesContent ? '' : 'visual-companion__tab--empty',
              ]
                .filter(Boolean)
                .join(' ')}
              onClick={() => selectTab('notes')}
            >
              Study Notes
            </button>
          </div>

          <div
            className="visual-companion__tabpanel"
            role="tabpanel"
            id={activeTab === 'visuals' ? 'visual-companion-panel-visuals' : 'visual-companion-panel-notes'}
            aria-labelledby={activeTab === 'visuals' ? 'visual-companion-tab-visuals' : 'visual-companion-tab-notes'}
          >
            {activeTab === 'visuals' ? (
              <VisualsTab
                tables={tables}
                formulas={formulas}
                pageView={activePageView}
                onNavigatePage={handleNavigatePage}
                diagramState={diagramState}
                canDrawDiagram={canDrawDiagram}
                onDrawDiagram={handleDrawDiagram}
                diagramMessageKey={messageKey}
                hasCitedPage={sources.some((s) => s.document_id !== null && s.page_number !== null)}
                onViewCitedPage={() => {
                  const firstCited = sources.find((s) => s.document_id !== null && s.page_number !== null)
                  if (firstCited && effectiveIndex !== null) {
                    viewTextbookPage(effectiveIndex, firstCited.document_id as number, firstCited.page_number as number)
                  }
                }}
              />
            ) : (
              <StudyNotesTab
                keyTerms={keyTerms}
                sources={sources}
                isSupplemented={isSupplemented}
                onViewPage={(documentId, pageNumber) => {
                  if (effectiveIndex !== null) viewTextbookPage(effectiveIndex, documentId, pageNumber)
                }}
              />
            )}
          </div>
        </div>
      </aside>

      {isMobileOpen && (
        <button
          type="button"
          className="visual-companion__backdrop"
          aria-label="Close Visual Companion"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      <button
        type="button"
        className={`visual-companion__mobile-toggle ${isHighlighted ? 'visual-companion__mobile-toggle--pulse' : ''}`}
        onClick={() => setIsMobileOpen((open) => !open)}
        aria-label="Toggle Visual Companion"
      >
        <span aria-hidden="true">𝑓(x)</span>
      </button>
    </>
  )
}
