import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const startRecordingMock = vi.fn().mockResolvedValue(undefined)
vi.mock('../hooks/useVoiceRecorder', () => ({
  useVoiceRecorder: () => ({
    isRecording: false,
    startRecording: startRecordingMock,
    stopRecording: vi.fn(),
    audioBlob: null,
    error: null,
    hasPermission: true,
    silenceProgress: 0,
  }),
  useBargeInMonitor: () => {},
}))

const stopAudioPlaybackMock = vi.fn()
vi.mock('../hooks/useAudioPlayer', () => ({
  playAudioBlob: vi.fn(),
  stopAudioPlayback: stopAudioPlaybackMock,
}))

const { useAppStore } = await import('../store')
const { VoiceButton } = await import('./VoiceButton')

describe('VoiceButton Space key', () => {
  beforeEach(() => {
    startRecordingMock.mockClear()
    stopAudioPlaybackMock.mockClear()
    useAppStore.setState({
      voiceState: 'playing',
      activePlaybackIndex: 0,
      voiceError: null,
      isUploadPanelOpen: false,
      isSettingsPanelOpen: false,
      isPageLightboxOpen: false,
    })
  })

  afterEach(() => {
    cleanup()
  })

  it('interrupts playback and starts recording, same as clicking the mic', () => {
    render(<VoiceButton />)

    fireEvent.keyDown(document, { code: 'Space' })

    expect(stopAudioPlaybackMock).toHaveBeenCalled()
    expect(startRecordingMock).toHaveBeenCalled()
    expect(useAppStore.getState().voiceState).toBe('recording')
    expect(useAppStore.getState().activePlaybackIndex).toBeNull()
  })

  it('does nothing while typing in a text field', () => {
    render(
      <div>
        <VoiceButton />
        <textarea data-testid="draft" />
      </div>,
    )

    const textarea = document.querySelector('[data-testid="draft"]') as HTMLTextAreaElement
    textarea.focus()
    fireEvent.keyDown(textarea, { code: 'Space' })

    expect(startRecordingMock).not.toHaveBeenCalled()
    expect(useAppStore.getState().voiceState).toBe('playing')
  })
})
