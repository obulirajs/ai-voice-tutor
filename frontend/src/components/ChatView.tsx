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
  if (types.has('supplemented')) return { glyph: '💡', title: 'Includes an example from outside the textbook — click to view Study Notes' }
  if (types.has('formula')) return { glyph: '𝑓(x)', title: 'Has a formula — click to view Study Notes' }
  if (types.has('highlight')) return { glyph: '✦', title: 'Has key terms — click to view Study Notes' }
  if (types.has('text_block')) return { glyph: '❝', title: 'Has a cited passage — click to view Study Notes' }
  return { glyph: '▦', title: 'Has visual content — click to view Study Notes' }
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

  // Voice and typed chat can't run at the same time in v1 -- once a voice
  // turn is transcribing/generating/speaking, the text input steps aside.
  const voiceBusy = voiceState === 'processing' || voiceState === 'playing'

  const handleSend = () => {
    const text = draft.trim()
    if (!text || isLoading || voiceBusy) return
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
                {isAssistant && (
                  <button
                    type="button"
                    className="chat-message__replay"
                    onClick={(event) => {
                      event.stopPropagation()
                      void playMessageAudio(index)
                    }}
                    disabled={voiceState === 'playing' || voiceState === 'processing'}
                    title="Read aloud"
                    aria-label="Read this response aloud"
                  >
                    {voiceState === 'processing' && activePlaybackIndex === index ? <ReplaySpinnerIcon /> : '🔊'}
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
                      {message.sources
                        .map((source) => `p.${source.page_number ?? '?'} (${source.score.toFixed(2)})`)
                        .join(', ')}
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
