import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'

import type { VisualDirective } from '../api'
import { useAppStore } from '../store'
import { MathText } from './MathText'
import { ModeToggle } from './ModeToggle'
import { VoiceButton } from './VoiceButton'
import './ChatView.css'

const STARTER_QUESTIONS = [
  'Explain the main topics in this chapter',
  'Quiz me on vocabulary',
  'Help me practice grammar',
]

function ReplaySpinnerIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" className="chat-message__replay-spinner" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

// The directive-indicator badge's glyph and tooltip should reflect what's
// actually in the message -- a **bold** vocabulary term (a "highlight"
// directive) is the common case and has nothing to do with math, so it
// must not show the same 𝑓(x) formula glyph a real LaTeX expression would.
function directiveIndicator(directives: VisualDirective[]): { glyph: string; title: string } {
  const types = new Set(directives.map((d) => d.directive_type))
  if (types.has('supplemented')) return { glyph: '💡', title: 'Includes an example from outside the textbook — click to view the Visual Companion' }
  if (types.has('table')) return { glyph: '▦', title: 'Has a comparison table — click to view the Visual Companion' }
  if (types.has('formula')) return { glyph: '𝑓(x)', title: 'Has a formula — click to view the Visual Companion' }
  if (types.has('highlight')) return { glyph: '✦', title: 'Has key terms — click to view the Visual Companion' }
  if (types.has('text_block')) return { glyph: '❝', title: 'Has a cited passage — click to view the Visual Companion' }
  return { glyph: '🖼', title: 'Has visual content — click to view the Visual Companion' }
}

export function ChatView() {
  const subjects = useAppStore((state) => state.subjects)
  const activeSubjectId = useAppStore((state) => state.activeSubjectId)
  const messages = useAppStore((state) => state.messages)
  const isLoading = useAppStore((state) => state.isLoading)
  const chatError = useAppStore((state) => state.chatError)
  const connectionError = useAppStore((state) => state.connectionError)
  const voiceState = useAppStore((state) => state.voiceState)
  const activeMessageIndex = useAppStore((state) => state.activeMessageIndex)
  const setActiveMessageIndex = useAppStore((state) => state.setActiveMessageIndex)
  const sendMessage = useAppStore((state) => state.sendMessage)
  const retryLastMessage = useAppStore((state) => state.retryLastMessage)
  const openUploadPanel = useAppStore((state) => state.openUploadPanel)
  const playMessageAudio = useAppStore((state) => state.playMessageAudio)
  const activePlaybackIndex = useAppStore((state) => state.activePlaybackIndex)
  const viewTextbookPage = useAppStore((state) => state.viewTextbookPage)
  const stopSpeaking = useAppStore((state) => state.stopSpeaking)
  const cancelVoiceTurn = useAppStore((state) => state.cancelVoiceTurn)

  const [draft, setDraft] = useState('')
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const activeSubject = subjects.find((subject) => subject.id === activeSubjectId) ?? null

  let lastAssistantIndex: number | null = null
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant') {
      lastAssistantIndex = i
      break
    }
  }
  const effectiveActiveIndex = activeMessageIndex ?? lastAssistantIndex

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isLoading])

  const toggleSources = (index: number) => {
    setExpandedSources((prev) => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  // Typing is fine while the tutor speaks -- only a voice turn actually
  // being transcribed/generated blocks the text input (sending mid-reply
  // would race the same session/message list a voice turn is about to update).
  const voiceBusy = voiceState === 'processing'

  const handleSend = () => {
    const text = draft.trim()
    if (!text || isLoading || voiceBusy) return
    // Sending stops any current playback first, same as starting a recording does.
    if (voiceState === 'playing') stopSpeaking()
    setDraft('')
    void sendMessage(text)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  if (!activeSubject) {
    return (
      <div className="chat-view chat-view--empty">
        <p>Select a subject from the sidebar to start studying.</p>
      </div>
    )
  }

  if (activeSubject.document_count === 0) {
    return (
      <div className="chat-view chat-view--empty">
        <p>No textbook uploaded for {activeSubject.name}. Upload one to start.</p>
        <button type="button" className="chat-view__upload-prompt" onClick={() => openUploadPanel(activeSubject.name)}>
          Upload a textbook
        </button>
      </div>
    )
  }

  return (
    <div className="chat-view">
      {connectionError && (
        <div className="chat-view__connection-banner" role="alert">
          <span>Can't reach the server. Check that the backend is running.</span>
          <button
            type="button"
            className="chat-view__connection-banner-retry"
            onClick={() => void retryLastMessage()}
          >
            Retry
          </button>
        </div>
      )}

      <div className="chat-view__messages">
        {messages.length === 0 ? (
          <div className="chat-view__welcome">
            <p className="chat-view__welcome-title">Ask me about your {activeSubject.name} textbook!</p>
            <div className="chat-view__starters">
              {STARTER_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="chat-view__starter"
                  onClick={() => void sendMessage(question)}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => {
            const isAssistant = message.role === 'assistant'
            const hasDirectives = Boolean(message.visual_directives && message.visual_directives.length > 0)
            const indicator = hasDirectives ? directiveIndicator(message.visual_directives ?? []) : null
            const isActive = isAssistant && index === effectiveActiveIndex
            // Which face the 🔊 button shows: this message's own
            // cancel-synthesis / stop-playback state, or the plain "play"
            // state (disabled only while a voice turn owns voiceState
            // entirely -- recording, or a different message synthesizing).
            const isThisProcessing = activePlaybackIndex === index && voiceState === 'processing'
            const isThisPlaying = activePlaybackIndex === index && voiceState === 'playing'

            return (
            <div
              key={index}
              className={`chat-message chat-message--${isAssistant ? 'assistant' : 'user'} ${
                isActive ? 'chat-message--active' : ''
              }`}
              {...(isAssistant
                ? {
                    role: 'button' as const,
                    tabIndex: 0,
                    onClick: () => setActiveMessageIndex(index),
                    onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        setActiveMessageIndex(index)
                      }
                    },
                  }
                : {})}
            >
              <p className="chat-message__content">
                <MathText content={message.content} />
                {indicator && (
                  <span className="chat-message__directive-indicator" aria-hidden="true" title={indicator.title}>
                    {indicator.glyph}
                  </span>
                )}
                {isAssistant && isThisProcessing && (
                  <button
                    type="button"
                    className="chat-message__replay"
                    onClick={(event) => {
                      event.stopPropagation()
                      cancelVoiceTurn()
                    }}
                    title="Cancel"
                    aria-label="Cancel synthesizing audio"
                  >
                    <ReplaySpinnerIcon />
                    <span className="chat-message__replay-cancel" aria-hidden="true">
                      ✕
                    </span>
                  </button>
                )}
                {isAssistant && isThisPlaying && (
                  <button
                    type="button"
                    className="chat-message__replay"
                    onClick={(event) => {
                      event.stopPropagation()
                      stopSpeaking()
                    }}
                    title="Stop"
                    aria-label="Stop reading this response aloud"
                  >
                    ⏹
                  </button>
                )}
                {isAssistant && !isThisProcessing && !isThisPlaying && (
                  <button
                    type="button"
                    className="chat-message__replay"
                    onClick={(event) => {
                      event.stopPropagation()
                      void playMessageAudio(index)
                    }}
                    disabled={voiceState === 'recording' || voiceState === 'processing'}
                    title="Read aloud"
                    aria-label="Read this response aloud"
                  >
                    🔊
                  </button>
                )}
              </p>
              {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                <div className="chat-message__sources">
                  <button
                    type="button"
                    className="chat-message__sources-toggle"
                    onClick={() => toggleSources(index)}
                  >
                    {expandedSources.has(index) ? 'Hide sources' : 'Show sources'}
                  </button>
                  {expandedSources.has(index) && (
                    <span className="chat-message__sources-list">
                      Sources:{' '}
                      {message.sources.map((source, sourceIndex) => (
                        <span key={sourceIndex}>
                          {sourceIndex > 0 && ', '}
                          {source.document_id !== null && source.page_number !== null ? (
                            <button
                              type="button"
                              className="chat-message__source-link"
                              title="View textbook page"
                              onClick={(event) => {
                                event.stopPropagation()
                                viewTextbookPage(index, source.document_id as number, source.page_number as number)
                              }}
                            >
                              PDF page {source.page_number} ({source.score.toFixed(2)})
                            </button>
                          ) : (
                            <>
                              PDF page {source.page_number ?? '?'} ({source.score.toFixed(2)})
                            </>
                          )}
                        </span>
                      ))}
                    </span>
                  )}
                </div>
              )}
            </div>
            )
          })
        )}

        {isLoading && (
          <div className="chat-message chat-message--assistant chat-message--typing">
            <span className="chat-typing-dot" />
            <span className="chat-typing-dot" />
            <span className="chat-typing-dot" />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {chatError && <p className="chat-view__error">{chatError}</p>}

      <ModeToggle />

      <div className="chat-input-bar">
        <textarea
          className="chat-input-bar__field"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`Ask a question about ${activeSubject.name}…`}
          rows={1}
          disabled={isLoading || voiceBusy}
        />
        <button
          type="button"
          className="chat-input-bar__send"
          onClick={handleSend}
          disabled={isLoading || voiceBusy || draft.trim().length === 0}
        >
          Send
        </button>
        <VoiceButton />
      </div>
    </div>
  )
}
