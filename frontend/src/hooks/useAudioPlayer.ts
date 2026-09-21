import { useCallback, useEffect, useState } from 'react'

// Module-level singleton: only one TTS clip plays at a time, and the
// Zustand store (which can't use hooks) drives playback directly via the
// plain playAudioBlob()/stopAudioPlayback() functions below. useAudioPlayer()
// is a thin reactive view over the same singleton for any component that
// wants to render playback state.
let activeAudio: HTMLAudioElement | null = null
let activeUrl: string | null = null
let isPlaying = false
let lastError: string | null = null
const listeners = new Set<() => void>()

function notify(): void {
  listeners.forEach((listener) => listener())
}

function releaseActiveAudio(): void {
  if (activeAudio) {
    activeAudio.pause()
    activeAudio.src = ''
    activeAudio = null
  }
  if (activeUrl) {
    URL.revokeObjectURL(activeUrl)
    activeUrl = null
  }
}

/** Stops in-progress playback, if any. Safe to call when nothing is playing. */
export function stopAudioPlayback(): void {
  releaseActiveAudio()
  if (isPlaying) {
    isPlaying = false
    notify()
  }
}

// Safety net: if 'ended'/'error' never fire for any reason (a stuck
// play() promise, an unusual stream the browser can't cleanly report the
// end of, ...), force playback to finish rather than leaving voiceState
// stuck on 'playing' forever -- which would permanently disable the text
// input and mic button.
//
// A fixed timeout here is wrong by construction: a real tutor answer can
// run well past 30s (a multi-paragraph explanation easily synthesizes to
// 45-60+ seconds of audio), so any fixed cap eventually kills a genuine,
// still-playing response -- exactly the "voice stops abruptly mid-read"
// symptom this was meant to prevent, not cause. Instead: a short watchdog
// only for the window before the browser knows the clip's real duration
// (a truly stuck play() call), then re-armed against that duration once
// 'loadedmetadata' fires, with a generous buffer for decode/seek slack.
const INITIAL_WATCHDOG_MS = 10000
const DURATION_BUFFER_MS = 10000
// Guards the rare case where `duration` comes back Infinity/NaN (some
// browsers report this for certain streamed media) instead of a real number.
const FALLBACK_DURATION_WATCHDOG_MS = 300000

/**
 * Plays one audio blob to completion, resolving when it ends (or fails).
 * Starting a new clip stops whatever was already playing.
 */

export function playAudioBlob(audioBlob: Blob): Promise<void> {
  releaseActiveAudio()
  lastError = null

  return new Promise<void>((resolve) => {
    const url = URL.createObjectURL(audioBlob)
    const audioEl = new Audio(url)
    activeUrl = url
    activeAudio = audioEl

    let watchdog: ReturnType<typeof setTimeout> | null = null

    const armWatchdog = (ms: number) => {
      if (watchdog) clearTimeout(watchdog)
      watchdog = setTimeout(() => finish('Audio playback timed out.'), ms)
    }

    const finish = (errorMessage?: string) => {
      if (watchdog) clearTimeout(watchdog)
      if (errorMessage) lastError = errorMessage
      isPlaying = false
      releaseActiveAudio()
      notify()
      resolve()
    }

    audioEl.onloadedmetadata = () => {
      const { duration } = audioEl
      armWatchdog(Number.isFinite(duration) && duration > 0 ? duration * 1000 + DURATION_BUFFER_MS : FALLBACK_DURATION_WATCHDOG_MS)
    }
    audioEl.onended = () => finish()
    audioEl.onerror = () => finish('Could not play the audio response.')

    armWatchdog(INITIAL_WATCHDOG_MS)

    isPlaying = true
    notify()
    audioEl.play().catch(() => finish('Could not play the audio response.'))
  })
}

interface UseAudioPlayerResult {
  isPlaying: boolean
  play: (audioBlob: Blob) => Promise<void>
  stop: () => void
  error: string | null
}

export function useAudioPlayer(): UseAudioPlayerResult {
  const [, setTick] = useState(0)

  useEffect(() => {
    const listener = () => setTick((tick) => tick + 1)
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  }, [])

  const play = useCallback((audioBlob: Blob) => playAudioBlob(audioBlob), [])
  const stop = useCallback(() => stopAudioPlayback(), [])

  return { isPlaying, play, stop, error: lastError }
}
