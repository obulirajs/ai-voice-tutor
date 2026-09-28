import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { playAudioBlob, stopAudioPlayback, useAudioPlayer } from './useAudioPlayer'

// jsdom doesn't implement real media playback -- stub play()/pause() so a
// clip's promise stays pending (as it would in a real browser, mid-clip)
// until the test drives it manually via stopAudioPlayback(), rather than
// jsdom's own "not implemented" rejection racing the test.
beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
})

afterEach(() => {
  stopAudioPlayback()
  vi.restoreAllMocks()
})

describe('stopAudioPlayback', () => {
  it('resolves the pending playAudioBlob promise with "interrupted" and sets no error', async () => {
    const { result } = renderHook(() => useAudioPlayer())

    const outcomePromise = playAudioBlob(new Blob(['fake audio'], { type: 'audio/webm' }))
    // Let play()'s mocked promise settle so the clip is genuinely "playing"
    // before we interrupt it, same as a real mid-clip stop.
    await Promise.resolve()

    stopAudioPlayback()

    await expect(outcomePromise).resolves.toBe('interrupted')
    expect(result.current.error).toBeNull()
  })

  it('is a no-op when nothing is playing', () => {
    expect(() => stopAudioPlayback()).not.toThrow()
  })

  it('lets a new playAudioBlob() call interrupt the previous one the same way', async () => {
    const firstOutcome = playAudioBlob(new Blob(['first'], { type: 'audio/webm' }))
    await Promise.resolve()

    const secondOutcome = playAudioBlob(new Blob(['second'], { type: 'audio/webm' }))

    await expect(firstOutcome).resolves.toBe('interrupted')
    void secondOutcome
  })
})
