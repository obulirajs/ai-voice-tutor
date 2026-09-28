import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { VoiceResponse } from './api'

const playAudioBlobMock = vi.fn().mockResolvedValue('ended')
vi.mock('./hooks/useAudioPlayer', () => ({
  playAudioBlob: (...args: unknown[]) => playAudioBlobMock(...args),
  stopAudioPlayback: vi.fn(),
}))

let resolveSendAudio: ((response: VoiceResponse) => void) | null = null
const sendAudioMock = vi.fn(
  () =>
    new Promise<VoiceResponse>((resolve) => {
      resolveSendAudio = resolve
    }),
)

vi.mock('./api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api')>()

  class FakeVoiceClient {
    isConnected = () => true
    connect = vi.fn().mockResolvedValue(undefined)
    disconnect = vi.fn()
    sendAudio = sendAudioMock
  }

  return { ...actual, VoiceClient: FakeVoiceClient }
})

// Imported after the mocks above so store.ts's module-level `new VoiceClient()`
// picks up the mocked class.
const { useAppStore } = await import('./store')

describe('cancelVoiceTurn', () => {
  beforeEach(() => {
    playAudioBlobMock.mockClear()
    sendAudioMock.mockClear()
    resolveSendAudio = null
    useAppStore.setState({
      messages: [],
      voiceState: 'idle',
      activeSubjectId: null,
      sessionId: null,
      activePlaybackIndex: null,
    })
  })

  it('appends the late voice-turn response as text without playing its audio', async () => {
    const turnPromise = useAppStore.getState().sendVoiceAudio(new Blob(['fake mic audio']))

    // Give sendVoiceAudio's connect()/isConnected() microtasks a tick to
    // resolve and actually call sendAudio() before we cancel.
    await Promise.resolve()
    await Promise.resolve()
    expect(useAppStore.getState().voiceState).toBe('processing')

    useAppStore.getState().cancelVoiceTurn()
    expect(useAppStore.getState().voiceState).toBe('idle')

    // The backend's response for the cancelled turn arrives late.
    expect(resolveSendAudio).not.toBeNull()
    resolveSendAudio?.({
      transcription: 'What is photosynthesis?',
      reply: 'Photosynthesis is how plants make food from sunlight.',
      session_id: 'session-1',
      is_command: false,
      command_type: null,
      new_subject: null,
      sources: [],
      visual_directives: [],
      audio: new Blob(['fake tts audio']),
    })
    await turnPromise

    const state = useAppStore.getState()
    expect(state.messages.map((m) => m.content)).toEqual([
      'What is photosynthesis?',
      'Photosynthesis is how plants make food from sunlight.',
    ])
    expect(playAudioBlobMock).not.toHaveBeenCalled()
    // cancelVoiceTurn's idle stands -- the late arrival must not resurrect 'playing'.
    expect(state.voiceState).toBe('idle')
  })

  it('a turn that is not cancelled plays its audio and returns to idle normally', async () => {
    const turnPromise = useAppStore.getState().sendVoiceAudio(new Blob(['fake mic audio']))
    await Promise.resolve()
    await Promise.resolve()

    expect(resolveSendAudio).not.toBeNull()
    resolveSendAudio?.({
      transcription: 'Hello',
      reply: 'Hi there!',
      session_id: 'session-1',
      is_command: false,
      command_type: null,
      new_subject: null,
      sources: [],
      visual_directives: [],
      audio: new Blob(['fake tts audio']),
    })
    await turnPromise

    expect(playAudioBlobMock).toHaveBeenCalledTimes(1)
    expect(useAppStore.getState().voiceState).toBe('idle')
  })
})
