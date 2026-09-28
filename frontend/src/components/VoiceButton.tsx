import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { useBargeInMonitor, useVoiceRecorder } from '../hooks/useVoiceRecorder'
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

// Shown on the mic button while the tutor is speaking -- a stop square
// (interrupt) with the mic's own outline still visible, since clicking it
// does both: stop playback AND start listening.
function InterruptIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="currentColor" stroke="none" />
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
  const stopSpeaking = useAppStore((state) => state.stopSpeaking)
  const cancelVoiceTurn = useAppStore((state) => state.cancelVoiceTurn)
  const interruptBySpeaking = useAppStore((state) => state.interruptBySpeaking)

  const { isRecording, startRecording, stopRecording, audioBlob, hasPermission, silenceProgress } = useVoiceRecorder()

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

    // Can't listen and speak at once -- interrupt any TTS playback first
    // (stopSpeaking also clears activePlaybackIndex, so a message's 🔊
    // button doesn't stay stuck showing "stop" after this).
    if (voiceState === 'playing') stopSpeaking()

    startVoiceRecordingAction()
    void startRecording()
  }

  // Space bar toggles recording (including interrupting playback -- same as
  // clicking the mic), except while the user is typing or a turn is
  // processing (Esc/the Stop pill handle cancelling that).
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return
      const target = event.target as HTMLElement | null
      const isTyping = target?.tagName === 'TEXTAREA' || target?.tagName === 'INPUT'
      if (isTyping || isBusy) return
      event.preventDefault()
      toggleRecording()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording, hasPermission, voiceState, isBusy])

  // Esc interrupts/cancels, unless a modal or the page lightbox is open --
  // those get first claim on Esc (see useModalDismiss + VisualsTab's
  // TextbookPageCard). Reads fresh state via getState() rather than
  // subscribing, since this only needs to act at keypress time.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      const state = useAppStore.getState()
      if (state.isUploadPanelOpen || state.isSettingsPanelOpen || state.isPageLightboxOpen) return
      if (state.voiceState === 'playing') {
        event.preventDefault()
        state.stopSpeaking()
      } else if (state.voiceState === 'processing') {
        event.preventDefault()
        state.cancelVoiceTurn()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  // "Interrupt by speaking" (Settings -> Voice, off by default): while ON
  // and the tutor is talking, a separate mic monitor listens for sustained
  // speech and barges in the same way clicking the mic does.
  useBargeInMonitor(interruptBySpeaking && voiceState === 'playing', () => {
    stopSpeaking()
    startVoiceRecordingAction()
    void startRecording()
  })

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
    icon = <InterruptIcon />
    label = 'Interrupt and ask'
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
        title={visualState === 'playing' ? label : 'Voice input (Space)'}
      >
        {icon}
      </button>
      {(voiceState === 'playing' || voiceState === 'processing') && (
        <button
          type="button"
          className="voice-stop-pill"
          onClick={voiceState === 'playing' ? stopSpeaking : cancelVoiceTurn}
          aria-label={voiceState === 'playing' ? 'Stop speaking' : 'Cancel'}
        >
          <span className="voice-stop-pill__icon" aria-hidden="true">
            ■
          </span>
          Stop
          <span className="voice-stop-pill__status">{voiceState === 'playing' ? 'Speaking…' : 'Thinking…'}</span>
        </button>
      )}
    </div>
  )
}
