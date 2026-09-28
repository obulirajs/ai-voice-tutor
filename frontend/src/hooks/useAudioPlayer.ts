import { useCallback, useEffect, useState } from 'react'

// Module-level singleton: only one TTS clip plays at a time, and the
// Zustand store (which can't use hooks) drives playback directly via the
// plain playAudioBlob()/stopAudioPlayback() functions below. useAudioPlayer()
// is a thin reactive view over the same singleton for any component that
// wants to render playback state.
export type PlaybackOutcome = 'ended' | 'interrupted' | 'error'

let activeAudio: HTMLAudioElement | null = null
let activeUrl: string | null = null
let isPlaying = false
let lastError: string | null = null
// The in-flight clip's own finish() closure, if any -- lets
// stopAudioPlayback() (called from outside playAudioBlob, e.g. the mic
// button or Esc) resolve that clip's pending promise cleanly instead of
// abandoning it. null whenever nothing is playing.
let activeFinish: ((outcome: PlaybackOutcome, errorMessage?: string) => void) | null = null
const listeners = new Set<() => void>()

function notify(): void {
  listeners.forEach((listener) => listener())
}

function detachAndReleaseAudio(): void {
  if (activeAudio) {
    // Detach handlers before pausing/clearing src -- on some browsers,
    // clearing `src` itself fires a spurious 'error' event, which must
    // never be mistaken for a real playback failure (especially during an
    // intentional interrupt).
    activeAudio.onended = null
    activeAudio.onerror = null
    activeAudio.pause()
    activeAudio.src = ''
    activeAudio = null
  }
  if (activeUrl) {
    URL.revokeObjectURL(activeUrl)
    activeUrl = null
  }
}

/**
 * Stops in-progress playback, if any -- resolving the pending playAudioBlob()
 * promise with 'interrupted' (never setting an error). Safe to call when
 * nothing is playing.
 */
export function stopAudioPlayback(): void {
  if (activeFinish) {
    activeFinish('interrupted')
    return
  }
  // Defensive fallback -- activeFinish should always be set whenever
  // isPlaying is true, but never leave isPlaying stuck true.
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
 * Plays one audio blob, resolving once it's done: 'ended' (played through
 * naturally), 'interrupted' (stopAudioPlayback() was called, or a newer
 * playAudioBlob() call superseded this one), or 'error'. Starting a new
 * clip stops whatever was already playing (that older clip resolves
 * 'interrupted', same as an explicit stopAudioPlayback() call).
 */
export function playAudioBlob(audioBlob: Blob): Promise<PlaybackOutcome> {
  stopAudioPlayback()
  lastError = null

  return new Promise<PlaybackOutcome>((resolve) => {
    const url = URL.createObjectURL(audioBlob)
    const audioEl = new Audio(url)
    activeUrl = url
    activeAudio = audioEl

    let watchdog: ReturnType<typeof setTimeout> | null = null

    const armWatchdog = (ms: number) => {
      if (watchdog) clearTimeout(watchdog)
      watchdog = setTimeout(() => finish('error', 'Audio playback timed out.'), ms)
    }

    const finish = (outcome: PlaybackOutcome, errorMessage?: string) => {
      if (watchdog) clearTimeout(watchdog)
      if (outcome === 'error' && errorMessage) lastError = errorMessage
      isPlaying = false
      activeFinish = null
      detachAndReleaseAudio()
      notify()
      resolve(outcome)
    }

    activeFinish = finish

    audioEl.onloadedmetadata = () => {
      const { duration } = audioEl
      armWatchdog(Number.isFinite(duration) && duration > 0 ? duration * 1000 + DURATION_BUFFER_MS : FALLBACK_DURATION_WATCHDOG_MS)
    }
    audioEl.onended = () => finish('ended')
    audioEl.onerror = () => finish('error', 'Could not play the audio response.')

    armWatchdog(INITIAL_WATCHDOG_MS)

    isPlaying = true
    notify()
    audioEl.play().catch(() => finish('error', 'Could not play the audio response.'))
  })
}

interface UseAudioPlayerResult {
  isPlaying: boolean
  play: (audioBlob: Blob) => Promise<PlaybackOutcome>
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
