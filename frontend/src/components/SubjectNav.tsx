import { useEffect, useState } from 'react'

import { useAppStore } from '../store'
import type { Subject } from '../api'
import './SubjectNav.css'

interface SubjectNavProps {
  // Called after a subject is selected -- lets the app shell close the
  // mobile sidebar overlay without SubjectNav knowing about it.
  onNavigate?: () => void
}

export function SubjectNav({ onNavigate }: SubjectNavProps) {
  const subjects = useAppStore((state) => state.subjects)
  const activeSubjectId = useAppStore((state) => state.activeSubjectId)
  const documentsBySubjectId = useAppStore((state) => state.documentsBySubjectId)
  const loadSubjects = useAppStore((state) => state.loadSubjects)
  const setActiveSubject = useAppStore((state) => state.setActiveSubject)
  const loadSubjectDocuments = useAppStore((state) => state.loadSubjectDocuments)
  const openUploadPanel = useAppStore((state) => state.openUploadPanel)

  const [subjectsLoaded, setSubjectsLoaded] = useState(false)
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set())
  const [loadingDocIds, setLoadingDocIds] = useState<Set<number>>(new Set())

  useEffect(() => {
    void loadSubjects().finally(() => setSubjectsLoaded(true))
  }, [loadSubjects])

  const handleSelect = (subject: Subject) => {
    setActiveSubject(subject.id)
    onNavigate?.()

    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(subject.id)) {
        next.delete(subject.id)
        return next
      }
      next.add(subject.id)
      return next
    })

    if (!documentsBySubjectId[subject.id]) {
      setLoadingDocIds((prev) => new Set(prev).add(subject.id))
      void loadSubjectDocuments(subject.id).finally(() => {
        setLoadingDocIds((prev) => {
          const next = new Set(prev)
          next.delete(subject.id)
          return next
        })
      })
    }
  }

  return (
    <>
      <nav className="subject-nav" aria-label="Subjects">
        <p className="sidebar__section-label">Subjects</p>

        {!subjectsLoaded ? (
          <div className="subject-nav__skeleton" aria-hidden="true">
            <div className="subject-nav__skeleton-row" />
            <div className="subject-nav__skeleton-row" />
            <div className="subject-nav__skeleton-row" />
          </div>
        ) : subjects.length === 0 ? (
          <div className="subject-nav__empty">
            <p>No subjects yet. Upload a textbook to get started.</p>
            <button type="button" className="subject-nav__empty-upload" onClick={() => openUploadPanel()}>
              Upload a textbook
            </button>
          </div>
        ) : (
          <ul className="subject-nav__list">
            {subjects.map((subject) => {
              const isActive = subject.id === activeSubjectId
              const isExpanded = expandedIds.has(subject.id)
              const documents = documentsBySubjectId[subject.id]

              return (
                <li key={subject.id} className="subject-item">
                  <div className="subject-item__row">
                    <button
                      type="button"
                      className={`subject-item__button ${isActive ? 'subject-item__button--active' : ''}`}
                      onClick={() => handleSelect(subject)}
                      aria-expanded={isExpanded}
                    >
                      <span className="subject-item__chevron" aria-hidden="true">
                        {isExpanded ? '▾' : '▸'}
                      </span>
                      <span className="subject-item__name">{subject.name}</span>
                      <span className="subject-item__badge">{subject.document_count}</span>
                    </button>
                    <button
                      type="button"
                      className="subject-item__upload"
                      aria-label={`Upload a document to ${subject.name}`}
                      onClick={(event) => {
                        event.stopPropagation()
                        openUploadPanel(subject.name)
                      }}
                    >
                      +
                    </button>
                  </div>

                  {isExpanded && (
                    <div className="subject-item__documents">
                      {loadingDocIds.has(subject.id) && <p className="subject-item__documents-status">Loading…</p>}
                      {!loadingDocIds.has(subject.id) && documents?.length === 0 && (
                        <p className="subject-item__documents-status">No documents yet.</p>
                      )}
                      {documents?.map((document) => (
                        <div key={document.id} className="subject-item__document">
                          <span className="subject-item__document-name">{document.filename}</span>
                          <span className="subject-item__document-pages">{document.page_count}p</span>
                        </div>
                      ))}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </nav>

      <div className="sidebar__section">
        <p className="sidebar__section-label">Upload</p>
        <button type="button" className="subject-nav__upload-button" onClick={() => openUploadPanel()}>
          + Upload textbook
        </button>
      </div>
    </>
  )
}
