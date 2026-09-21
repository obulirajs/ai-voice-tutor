import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { useVoiceRecorder } from '../hooks/useVoiceRecorder'
import { useAppStore } from '../store'
import './VoiceButton.css'

const HINT_DISPLAY_MS = 4000

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  )
}

function MicOffIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
      <line x1="3" y1="3" x2="21" y2="21" />
    </svg>
  )
}

function SpeakerIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <polygon points="4,9 8,9 12,5 12,19 8,15 4,15" fill="currentColor" />
      <path d="M16 9a5 5 0 0 1 0 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M18.5 6.5a9 9 0 0 1 0 11" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" className="voice-button__spinner" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function WarningIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3 21 20 3 20Z" />
      <line x1="12" y1="9" x2="12" y2="14" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function VoiceButton() {
  const voiceState = useAppStore((state) => state.voiceState)
  const voiceError = useAppStore((state) => state.voiceError)
  const setVoiceEnabled = useAppStore((state) => state.setVoiceEnabled)
  const startVoiceRecordingAction = useAppStore((state) => state.startVoiceRecording)
  const stopVoiceRecordingAction = useAppStore((state) => state.stopVoiceRecording)
  const sendVoiceAudio = useAppStore((state) => state.sendVoiceAudio)

  const { isRecording, startRecording, stopRecording, audioBlob, hasPermission, silenceProgress } = useVoiceRecorder()
  const { stop: stopPlayback } = useAudioPlayer()

  const [showHint, setShowHint] = useState(false)
  const sentBlobRef = useRef<Blob | null>(null)
  const hintTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (hasPermission !== null) setVoiceEnabled(hasPermission)
  }, [hasPermission, setVoiceEnabled])

  // Once the recorder hands back a finished clip, send it for processing.
  useEffect(() => {
    if (audioBlob && audioBlob !== sentBlobRef.current) {
      sentBlobRef.current = audioBlob
      void sendVoiceAudio(audioBlob)
    }
  }, [audioBlob, sendVoiceAudio])

  useEffect(() => {
    return () => {
      if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    }
  }, [])

  const flashHint = () => {
    setShowHint(true)
    if (hintTimeoutRef.current) clearTimeout(hintTimeoutRef.current)
    hintTimeoutRef.current = setTimeout(() => setShowHint(false), HINT_DISPLAY_MS)
  }

  const isBusy = voiceState === 'processing'

  const toggleRecording = () => {
    if (isBusy) return

    if (isRecording) {
      stopRecording()
      stopVoiceRecordingAction()
      return
    }

    if (hasPermission === false || hasPermission === null) {
      flashHint()
      if (hasPermission === false) return
    }

    // Can't listen and speak at once -- stop any TTS playback first.
    if (voiceState === 'playing') stopPlayback()

    startVoiceRecordingAction()
    void startRecording()
  }

  // Space bar toggles recording, except while the user is typing.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target as HTMLElement | null
      const isTyping = target?.tagName === 'TEXTAREA' || target?.tagName === 'INPUT'
      if (isTyping || isBusy || voiceState === 'playing') return
      event.preventDefault()
      toggleRecording()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording, hasPermission, voiceState, isBusy])

  const visualState = hasPermission === false ? 'no-permission' : voiceState

  let icon = <MicIcon />
  let label = 'Start voice input'
  if (visualState === 'no-permission') {
    icon = <MicOffIcon />
    label = 'Microphone access required'
  } else if (visualState === 'recording') {
    icon = <MicIcon />
    label = 'Stop recording'
  } else if (visualState === 'processing') {
    icon = <SpinnerIcon />
    label = 'Processing voice input'
  } else if (visualState === 'playing') {
    icon = <SpeakerIcon />
    label = 'Playing response'
  } else if (visualState === 'error') {
    icon = <WarningIcon />
    label = 'Voice input failed'
  }

  return (
    <div className="voice-button-wrapper">
      {showHint && (
        <div className="voice-button__hint" role="status">
          {hasPermission === false
            ? 'Microphone access was denied. Enable it in your browser settings to use voice.'
            : 'Click Allow to enable voice input.'}
        </div>
      )}
      {voiceState === 'error' && voiceError && (
        <div className="voice-button__hint voice-button__hint--error" role="alert">
          {voiceError}
        </div>
      )}
      <button
        type="button"
        className={`voice-button voice-button--${visualState} ${
          visualState === 'recording' && silenceProgress > 0 ? 'voice-button--silence-ring' : ''
        }`}
        style={visualState === 'recording' ? ({ '--silence-progress': silenceProgress } as CSSProperties) : undefined}
        onClick={toggleRecording}
        disabled={isBusy}
        aria-label={label}
        aria-pressed={isRecording}
        title="Voice input (Space)"
      >
        {icon}
      </button>
    </div>
  )
}
