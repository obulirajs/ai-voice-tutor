// The Visual Companion's "Study Notes" tab body -- key terms, sources, and
// the "beyond the textbook" note. Formulas/tables live in VisualsTab now;
// this tab is left with the non-visual supporting material.

import type { SourceCitation, VisualDirective } from '../api'

interface StudyNotesTabProps {
  keyTerms: VisualDirective[]
  sources: SourceCitation[]
  isSupplemented: boolean
  onViewPage: (documentId: number, pageNumber: number) => void
}

export function StudyNotesTab({ keyTerms, sources, isSupplemented, onViewPage }: StudyNotesTabProps) {
  if (keyTerms.length === 0 && sources.length === 0 && !isSupplemented) {
    return <p className="visual-companion__empty">No notes for this answer.</p>
  }

  return (
    <>
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
            {sources.map((source, index) =>
              source.document_id !== null && source.page_number !== null ? (
                <button
                  key={index}
                  type="button"
                  className="visual-companion__page-badge visual-companion__page-badge--button"
                  title="View textbook page"
                  onClick={() => onViewPage(source.document_id as number, source.page_number as number)}
                >
                  PDF page {source.page_number}
                </button>
              ) : (
                <span key={index} className="visual-companion__page-badge">
                  PDF page {source.page_number ?? '?'}
                </span>
              ),
            )}
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
  )
}
