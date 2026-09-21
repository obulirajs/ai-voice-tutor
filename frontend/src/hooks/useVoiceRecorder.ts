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
