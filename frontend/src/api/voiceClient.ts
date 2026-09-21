// WebSocket client for the voice endpoint (backend/app/api/voice.py).
// Turn-based, not streaming, for v1: connect() -> "ready", then each
// sendAudio() call runs one full audio_start -> binary -> audio_end turn
// and resolves with the transcription/reply/sources/audio for that turn.

import type { VoiceResponse } from './types'

interface PendingReady {
  resolve: () => void
  reject: (error: Error) => void
}

interface PendingTurn {
  resolve: (response: VoiceResponse) => void
  reject: (error: Error) => void
  transcription?: string
  reply?: string
  sources?: VoiceResponse['sources']
  sessionId?: string
  isCommand?: boolean
  commandType?: string | null
  newSubject?: string | null
  visualDirectives?: VoiceResponse['visual_directives']
  audio?: Blob
}

function detectAudioMimeType(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer.slice(0, 4))
  // "RIFF...." container -- Piper's WAV output.
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
    return 'audio/wav'
  }
  // "ID3" tag or an MPEG frame sync -- edge-tts's MP3 output (the default).
  if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) {
    return 'audio/mpeg'
  }
  if (bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) {
    return 'audio/mpeg'
  }
  return 'audio/mpeg'
}

export class VoiceClient {
  private ws: WebSocket | null = null
  private pendingReady: PendingReady | null = null
  private pendingTurn: PendingTurn | null = null
  private lastSessionId: string | null = null
  private lastSubject: string | null = null
  private lastVoice: string | null = null
  private intentionalClose = false
  private reconnectAttempted = false

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  connect(sessionId?: string | null, subject?: string | null, voice?: string | null): Promise<void> {
    this.lastSessionId = sessionId ?? null
    this.lastSubject = subject ?? null
    this.lastVoice = voice ?? null
    this.intentionalClose = false
    this.reconnectAttempted = false
    return this.open()
  }

  disconnect(): void {
    this.intentionalClose = true
    this.pendingReady?.reject(new Error('Voice connection closed.'))
    this.pendingReady = null
    this.pendingTurn?.reject(new Error('Voice connection closed.'))
    this.pendingTurn = null
    this.ws?.close()
    this.ws = null
  }

  sendAudio(audioBlob: Blob): Promise<VoiceResponse> {
    if (!this.isConnected() || this.ws === null) {
      return Promise.reject(new Error('Voice connection is not open.'))
    }
    if (this.pendingTurn !== null) {
      return Promise.reject(new Error('A voice turn is already in progress.'))
    }

    const ws = this.ws
    return new Promise<VoiceResponse>((resolve, reject) => {
      this.pendingTurn = { resolve, reject }
      void audioBlob
        .arrayBuffer()
        .then((buffer) => {
          ws.send(JSON.stringify({ type: 'audio_start' }))
          ws.send(buffer)
          ws.send(JSON.stringify({ type: 'audio_end' }))
        })
        .catch((error: unknown) => {
          this.pendingTurn = null
          reject(error instanceof Error ? error : new Error('Failed to read the recorded audio.'))
        })
    })
  }

  private open(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const url = `ws://${window.location.host}/api/voice`
      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'
      this.ws = ws
      this.pendingReady = { resolve, reject }

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: 'init',
            session_id: this.lastSessionId,
            subject: this.lastSubject,
            voice: this.lastVoice,
          }),
        )
      }

      ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => this.handleMessage(event)

      ws.onclose = () => this.handleClose()
    })
  }

  private handleClose(): void {
    const pendingReady = this.pendingReady
    const pendingTurn = this.pendingTurn
    this.ws = null
    this.pendingReady = null
    this.pendingTurn = null

    if (this.intentionalClose) return

    pendingTurn?.reject(new Error('Voice connection closed unexpectedly while waiting for a response.'))

    // One automatic reconnect before giving up, per design -- a dropped
    // connection is common enough (dev server restart, brief network blip)
    // not to surface as an error on the first occurrence.
    if (!this.reconnectAttempted) {
      this.reconnectAttempted = true
      this.open().then(
        () => pendingReady?.resolve(),
        (error: unknown) =>
          pendingReady?.reject(error instanceof Error ? error : new Error('Voice reconnect failed.')),
      )
    } else {
      pendingReady?.reject(new Error('Voice connection closed unexpectedly.'))
    }
  }

  private handleMessage(event: MessageEvent<string | ArrayBuffer>): void {
    if (typeof event.data === 'string') {
      this.handleTextFrame(event.data)
    } else {
      this.handleBinaryFrame(event.data)
    }
  }

  private handleTextFrame(text: string): void {
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(text) as Record<string, unknown>
    } catch {
      return
    }

    switch (payload.type) {
      case 'ready': {
        this.lastSessionId = (payload.session_id as string) ?? this.lastSessionId
        this.reconnectAttempted = false
        const ready = this.pendingReady
        this.pendingReady = null
        ready?.resolve()
        break
      }
      case 'response': {
        if (this.pendingTurn) {
          this.pendingTurn.transcription = payload.transcription as string
          this.pendingTurn.reply = payload.reply as string
          this.pendingTurn.sources = (payload.sources as VoiceResponse['sources']) ?? []
          this.pendingTurn.sessionId = payload.session_id as string
          this.pendingTurn.isCommand = Boolean(payload.is_command)
          this.pendingTurn.commandType = (payload.command_type as string | null) ?? null
          this.pendingTurn.newSubject = (payload.new_subject as string | null) ?? null
          this.pendingTurn.visualDirectives = (payload.visual_directives as VoiceResponse['visual_directives']) ?? []
        }
        // Remembered for a reconnect (e.g. a switch_subject command changes
        // which session this connection's turns belong to).
        this.lastSessionId = (payload.session_id as string) ?? this.lastSessionId
        break
      }
      case 'turn_complete': {
        this.resolvePendingTurn()
        break
      }
      case 'error': {
        const detail = typeof payload.detail === 'string' ? payload.detail : 'Voice request failed.'
        if (this.pendingReady) {
          const ready = this.pendingReady
          this.pendingReady = null
          ready.reject(new Error(detail))
        } else if (this.pendingTurn) {
          const turn = this.pendingTurn
          this.pendingTurn = null
          turn.reject(new Error(detail))
        } else {
          console.error('Voice: unhandled error frame', detail)
        }
        break
      }
      default:
        break
    }
  }

  private handleBinaryFrame(data: ArrayBuffer): void {
    if (!this.pendingTurn) return
    this.pendingTurn.audio = new Blob([data], { type: detectAudioMimeType(data) })
  }

  private resolvePendingTurn(): void {
    const turn = this.pendingTurn
    if (!turn) return
    this.pendingTurn = null

    if (turn.transcription === undefined || turn.reply === undefined || turn.sessionId === undefined) {
      turn.reject(new Error('Voice turn completed without a response.'))
      return
    }

    turn.resolve({
      transcription: turn.transcription,
      reply: turn.reply,
      sources: turn.sources ?? [],
      session_id: turn.sessionId,
      is_command: turn.isCommand ?? false,
      command_type: turn.commandType ?? null,
      new_subject: turn.newSubject ?? null,
      visual_directives: turn.visualDirectives ?? [],
      audio: turn.audio ?? new Blob([]),
    })
  }
}
