import { useCallback, useEffect, useRef, useState } from 'react'

// Opus in a WebM container is compact and something faster-whisper decodes
// natively (via PyAV) -- preferred whenever the browser supports it;
// anything else falls back to whatever MediaRecorder picks by default.
const PREFERRED_MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
// 128 kbps is good-quality speech audio without being wasteful -- explicit
// rather than leaving it to the browser's own default, which varies.
const AUDIO_BITS_PER_SECOND = 128_000
// A recording this short (an accidental tap, a click-and-immediately-stop)
// can't contain real speech -- send it to ASR and it'll just produce garbage.
const MIN_RECORDING_DURATION_MS = 500

// Voice Activity Detection tuning -- see docs/development-plan.md's VAD
// auto-stop prompt for the reasoning behind each value.
const VAD_SILENCE_THRESHOLD = 0.02 // Time-domain RMS (0-1 scale) below this counts as silence.
const VAD_SILENCE_DURATION_MS = 2000 // How long silence must persist before auto-stop.
const VAD_INITIAL_GRACE_MS = 2000 // Ignore silence detection right after recording starts.

// Barge-in ("Interrupt by speaking", Settings -> Voice) tuning -- same
// RMS-over-time-domain-data approach as the VAD constants above, but a
// separate monitor stream/threshold: talking over the tutor's synthesized
// speech means the mic also picks up that speech itself (echoCancellation
// helps but doesn't fully remove it through real speakers), so the bar for
// "the student is actually talking" needs real margin above
// VAD_SILENCE_THRESHOLD's ambient-noise floor.
const BARGE_IN_THRESHOLD = 0.06 // Time-domain RMS (0-1 scale) above this counts as speech.
const BARGE_IN_MIN_MS = 300 // How long speech must persist before triggering an interrupt.
const BARGE_IN_IGNORE_MS = 500 // Ignore this long after the monitor starts -- speaker ramp-up/echo settle.

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
}

interface UseVoiceRecorderResult {
  isRecording: boolean
  startRecording: () => Promise<void>
  stopRecording: () => void
  audioBlob: Blob | null
  error: string | null
  // null = permission not yet requested/known; true/false once it is.
  hasPermission: boolean | null
  // 0 when not in a silence window, ramping to 1 as auto-stop approaches --
  // VoiceButton renders this as a ring around the mic button.
  silenceProgress: number
}

export function useVoiceRecorder(): UseVoiceRecorderResult {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const [silenceProgress, setSilenceProgress] = useState(0)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const isRecordingRef = useRef(false)
  const recordingStartTimeRef = useRef(0)

  const vadFrameRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)

  const releaseStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }, [])

  const stopVAD = useCallback(() => {
    if (vadFrameRef.current !== null) {
      cancelAnimationFrame(vadFrameRef.current)
      vadFrameRef.current = null
    }
    analyserRef.current = null
    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => {})
      audioContextRef.current = null
    }
    setSilenceProgress(0)
  }, [])

  useEffect(
    () => () => {
      stopVAD()
      releaseStream()
    },
    [releaseStream, stopVAD],
  )

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
    isRecordingRef.current = false
    setIsRecording(false)
    stopVAD()
  }, [stopVAD])

  const startVAD = useCallback(
    (stream: MediaStream) => {
      try {
        const AudioContextCtor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
        if (!AudioContextCtor) return // No Web Audio API support -- fall back to manual-only mode.

        const audioContext = new AudioContextCtor()
        const source = audioContext.createMediaStreamSource(stream)
        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 512
        // Read-only monitoring tap: `source` is a node built from the same
        // MediaStream the MediaRecorder above already captures directly, and
        // `analyser` is a dead end (never connected to audioContext.destination,
        // which would otherwise loop the mic back out to speakers). Neither
        // side of this graph touches the MediaRecorder's recording path.
        source.connect(analyser)

        audioContextRef.current = audioContext
        analyserRef.current = analyser

        // Time-domain data (raw waveform amplitude, centered at 128) is a
        // more reliable silence signal than frequency-domain data -- a noise
        // floor spread across FFT bins can look non-trivial even during
        // true silence, causing premature/inconsistent cutoffs.
        const dataArray = new Uint8Array(analyser.fftSize)
        let silenceStart: number | null = null

        const checkVAD = () => {
          if (!isRecordingRef.current) return

          analyser.getByteTimeDomainData(dataArray)
          let sumSquares = 0
          for (let i = 0; i < dataArray.length; i += 1) {
            const normalized = (dataArray[i] - 128) / 128 // -1 to 1
            sumSquares += normalized * normalized
          }
          const rms = Math.sqrt(sumSquares / dataArray.length)

          const elapsed = Date.now() - recordingStartTimeRef.current

          if (elapsed < VAD_INITIAL_GRACE_MS) {
            silenceStart = null
            setSilenceProgress(0)
          } else if (rms < VAD_SILENCE_THRESHOLD) {
            if (silenceStart === null) {
              silenceStart = Date.now()
            }
            const elapsedSilence = Date.now() - silenceStart
            setSilenceProgress(Math.min(elapsedSilence / VAD_SILENCE_DURATION_MS, 1))
            if (elapsedSilence >= VAD_SILENCE_DURATION_MS) {
              stopRecording()
              return
            }
          } else {
            silenceStart = null
            setSilenceProgress(0)
          }

          vadFrameRef.current = requestAnimationFrame(checkVAD)
        }

        vadFrameRef.current = requestAnimationFrame(checkVAD)
      } catch {
        // AudioContext unavailable/blocked -- manual stop still works.
      }
    },
    [stopRecording],
  )

  const startRecording = useCallback(async () => {
    setError(null)
    setAudioBlob(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1, // mono -- better for speech recognition, no benefit from stereo here
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream
      setHasPermission(true)

      const mimeType = pickMimeType()
      const recorderOptions: MediaRecorderOptions = { audioBitsPerSecond: AUDIO_BITS_PER_SECOND }
      if (mimeType) recorderOptions.mimeType = mimeType
      const recorder = new MediaRecorder(stream, recorderOptions)
      chunksRef.current = []
      recordingStartTimeRef.current = Date.now()
      console.log('MediaRecorder using:', recorder.mimeType)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        releaseStream()
        // Too short to be real speech -- avoid sending garbage to ASR.
        if (Date.now() - recordingStartTimeRef.current < MIN_RECORDING_DURATION_MS) return
        setAudioBlob(new Blob(chunksRef.current, { type: recorder.mimeType }))
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      isRecordingRef.current = true
      setIsRecording(true)
      startVAD(stream)
    } catch (caught) {
      setHasPermission(false)
      isRecordingRef.current = false
      setIsRecording(false)
      setError(
        caught instanceof DOMException && caught.name === 'NotAllowedError'
          ? 'Microphone access was denied. Enable it in your browser settings to use voice.'
          : 'Could not access the microphone.',
      )
    }
  }, [releaseStream, startVAD])

  return { isRecording, startRecording, stopRecording, audioBlob, error, hasPermission, silenceProgress }
}

/**
 * While `enabled`, opens a mic stream separate from useVoiceRecorder's own
 * recording stream and watches for sustained speech via RMS -- "Interrupt
 * by speaking" (Settings -> Voice, off by default). Calls `onBargeIn()` at
 * most once per enabled period, then stops monitoring itself; the caller is
 * expected to flip `enabled` off in response (e.g. VoiceButton stops
 * playback and starts a real recording, which moves voiceState off
 * 'playing'). Always closes its stream when `enabled` goes false, so the
 * mic light isn't left on once the tutor stops talking.
 */
export function useBargeInMonitor(enabled: boolean, onBargeIn: () => void): void {
  const onBargeInRef = useRef(onBargeIn)
  useEffect(() => {
    onBargeInRef.current = onBargeIn
  }, [onBargeIn])

  useEffect(() => {
    if (!enabled) return

    let cancelled = false
    let frame: number | null = null
    let audioContext: AudioContext | null = null
    let stream: MediaStream | null = null
    const startTime = Date.now()
    let speechStart: number | null = null

    navigator.mediaDevices
      .getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      .then((mediaStream) => {
        if (cancelled) {
          mediaStream.getTracks().forEach((track) => track.stop())
          return
        }
        stream = mediaStream

        const AudioContextCtor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
        if (!AudioContextCtor) return // No Web Audio API support -- barge-in just doesn't fire.

        audioContext = new AudioContextCtor()
        const source = audioContext.createMediaStreamSource(mediaStream)
        const analyser = audioContext.createAnalyser()
        analyser.fftSize = 512
        // Read-only monitoring tap, same as startVAD's -- never connected to
        // audioContext.destination, so this can't loop the mic back to speakers.
        source.connect(analyser)

        const dataArray = new Uint8Array(analyser.fftSize)

        const check = () => {
          if (cancelled) return

          if (Date.now() - startTime >= BARGE_IN_IGNORE_MS) {
            analyser.getByteTimeDomainData(dataArray)
            let sumSquares = 0
            for (let i = 0; i < dataArray.length; i += 1) {
              const normalized = (dataArray[i] - 128) / 128
              sumSquares += normalized * normalized
            }
            const rms = Math.sqrt(sumSquares / dataArray.length)

            if (rms >= BARGE_IN_THRESHOLD) {
              speechStart ??= Date.now()
              if (Date.now() - speechStart >= BARGE_IN_MIN_MS) {
                onBargeInRef.current()
                return // Stop monitoring -- the caller is expected to flip `enabled` off.
              }
            } else {
              speechStart = null
            }
          }

          frame = requestAnimationFrame(check)
        }

        frame = requestAnimationFrame(check)
      })
      .catch(() => {
        // Mic unavailable/denied for the monitor -- hands-free barge-in
        // just doesn't fire; the explicit mic/Space/Esc controls still work.
      })

    return () => {
      cancelled = true
      if (frame !== null) cancelAnimationFrame(frame)
      if (audioContext) void audioContext.close().catch(() => {})
      if (stream) stream.getTracks().forEach((track) => track.stop())
    }
  }, [enabled])
}
