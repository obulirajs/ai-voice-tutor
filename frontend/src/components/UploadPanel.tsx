import { useEffect, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'

import { uploadDocument } from '../api'
import type { DocumentIngestResult } from '../api'
import { useModalDismiss } from '../hooks/useModalDismiss'
import { useAppStore } from '../store'
import './UploadPanel.css'

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

const AUTO_CLOSE_DELAY_MS = 4000

export function UploadPanel() {
  const isOpen = useAppStore((state) => state.isUploadPanelOpen)
  const prefillSubject = useAppStore((state) => state.uploadPanelSubject)
  const closeUploadPanel = useAppStore((state) => state.closeUploadPanel)
  const loadSubjects = useAppStore((state) => state.loadSubjects)
  const loadSubjectDocuments = useAppStore((state) => state.loadSubjectDocuments)

  const [subjectName, setSubjectName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<UploadStatus>('idle')
  const [result, setResult] = useState<DocumentIngestResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  // Reset the form each time the panel opens -- tracked against the previous
  // render's isOpen value (not an effect) so it runs synchronously with the
  // render that flips isOpen, per React's "adjusting state on prop change".
  const [wasOpen, setWasOpen] = useState(isOpen)
  if (isOpen !== wasOpen) {
    setWasOpen(isOpen)
    if (isOpen) {
      setSubjectName(prefillSubject ?? '')
      setFile(null)
      setStatus('idle')
      setResult(null)
      setErrorMessage(null)
      setIsDragOver(false)
    }
  }

  useEffect(() => {
    if (status !== 'success') return
    const timeout = setTimeout(() => closeUploadPanel(), AUTO_CLOSE_DELAY_MS)
    return () => clearTimeout(timeout)
  }, [status, closeUploadPanel])

  const panelRef = useModalDismiss(isOpen, closeUploadPanel)

  if (!isOpen) return null

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] ?? null)
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) setFile(dropped)
  }

  const handleUpload = async () => {
    const trimmedSubject = subjectName.trim()
    if (!trimmedSubject || !file) return

    setStatus('uploading')
    setErrorMessage(null)

    try {
      const ingestResult = await uploadDocument(trimmedSubject, file)
      setResult(ingestResult)
      setStatus('success')

      await loadSubjects()
      const updatedSubject = useAppStore.getState().subjects.find((subject) => subject.name === trimmedSubject)
      if (updatedSubject) {
        await loadSubjectDocuments(updatedSubject.id)
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Upload failed.')
      setStatus('error')
    }
  }

  return (
    <div className="upload-panel-overlay" onClick={closeUploadPanel}>
      <div
        ref={panelRef}
        className="upload-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-panel-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="upload-panel__header">
          <h2 className="upload-panel__title" id="upload-panel-title">
            Upload textbook
          </h2>
          <button type="button" className="upload-panel__close" aria-label="Close" onClick={closeUploadPanel}>
            ×
          </button>
        </div>

        {status === 'success' && result ? (
          <div className="upload-panel__result">
            <p>Uploaded successfully.</p>
            <ul className="upload-panel__result-list">
              <li>{result.page_count} pages</li>
              <li>{result.scanned_page_count} scanned pages (processed with OCR)</li>
              <li>{result.chunk_count} chunks indexed</li>
            </ul>
            {result.subject_mismatch_warning && (
              <div className="upload-panel__warning">{result.subject_mismatch_warning}</div>
            )}
            {result.quality_warning && <div className="upload-panel__warning">{result.quality_warning}</div>}
            <button type="button" className="upload-panel__done" onClick={closeUploadPanel}>
              Done
            </button>
          </div>
        ) : (
          <>
            <label className="upload-panel__field">
              <span>Subject</span>
              <input
                type="text"
                value={subjectName}
                onChange={(event) => setSubjectName(event.target.value)}
                placeholder="e.g. French"
                disabled={status === 'uploading'}
              />
            </label>

            <div
              className={`upload-panel__dropzone ${isDragOver ? 'upload-panel__dropzone--active' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setIsDragOver(true)
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
            >
              <p className="upload-panel__dropzone-text">{file ? file.name : 'Drag a PDF here, or'}</p>
              <label className="upload-panel__file-picker">
                Choose file
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  disabled={status === 'uploading'}
                  hidden
                />
              </label>
            </div>

            {status === 'uploading' && (
              <p className="upload-panel__progress" role="status">
                Uploading and processing… this can take a while for large or scanned PDFs.
              </p>
            )}
            {status === 'error' && errorMessage && <p className="upload-panel__error">{errorMessage}</p>}

            <button
              type="button"
              className="upload-panel__submit"
              onClick={() => void handleUpload()}
              disabled={!subjectName.trim() || !file || status === 'uploading'}
            >
              {status === 'uploading' ? 'Uploading…' : 'Upload'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}
