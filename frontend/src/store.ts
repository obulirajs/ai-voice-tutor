// Global app state (Zustand -- see technical-design.md: deliberately not
// Redux, this is a single-user app that doesn't need the ceremony).

import { create } from 'zustand'

import {
  chat as chatRequest,
  getAvailableVoices,
  getSessionMessages,
  getSubjectDocuments,
  getSubjects,
  getSubjectSessions,
  NetworkError,
  requestDiagram,
  setRole as setApiRole,
  synthesizeSpeech,
  VoiceClient,
} from './api'
import type {
  Document,
  Role,
  SessionPreview,
  SourceCitation,
  Subject,
  TeachingMode,
  VisualDirective,
  VoiceOption,
} from './api'
import { parseMermaidDiagram } from './diagram'
import { playAudioBlob, stopAudioPlayback } from './hooks/useAudioPlayer'
import type { Theme } from './theme'

export interface ChatMessage {
  role: string
  content: string
  sources?: SourceCitation[]
  visual_directives?: VisualDirective[]
  // Cached TTS audio so a voice-turn reply can be replayed without a fresh
  // /api/tts call; a typed-chat reply starts without one and gets it lazily
  // (see playMessageAudio) the first time it's replayed.
  audioBlob?: Blob
}

// A click on a "Page N" source badge (ChatView's sources list, or the
// Visual Companion's Study Notes tab) -- the Visuals tab renders this as a
// "Textbook page" card. requestId is a monotonic counter, not a value the
// badge itself carries: VisualCompanion uses it to detect a genuinely new
// click (vs. a re-render) and switch to the Visuals tab exactly once per
// click, even if the same page is clicked twice in a row.
export interface PageViewRequest {
  requestId: number
  messageIndex: number
  documentId: number
  pageNumber: number
}

// One assistant message's on-demand "Draw diagram" state (Visual
// Companion's Visuals tab, DiagramCard) -- keyed by message index in
// AppState.diagramsByMessageIndex, same pattern as pageView, so switching
// back to an already-drawn message shows it instantly instead of
// re-requesting.
export type DiagramStatus = 'idle' | 'loading' | 'ready' | 'failed' | 'not_applicable'

export interface DiagramState {
  status: DiagramStatus
  mermaid?: string
  error?: string
}

const ROLE_STORAGE_KEY = 'project-tutor:role'
const VOICE_STORAGE_KEY = 'project-tutor:voice'
const INTERRUPT_BY_SPEAKING_STORAGE_KEY = 'project-tutor:interrupt-by-speaking'

function readStoredRole(): Role {
  try {
    return localStorage.getItem(ROLE_STORAGE_KEY) === 'parent' ? 'parent' : 'student'
  } catch {
    return 'student'
  }
}

function readStoredVoice(): string | null {
  try {
    return localStorage.getItem(VOICE_STORAGE_KEY)
  } catch {
    return null
  }
}

function readStoredInterruptBySpeaking(): boolean {
  try {
    return localStorage.getItem(INTERRUPT_BY_SPEAKING_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export type VoiceState = 'idle' | 'recording' | 'processing' | 'playing' | 'error'

interface SendMessageOptions {
  // Used by retryLastMessage -- the user bubble is already in the
  // transcript from the failed attempt, so don't append it again.
  skipAppend?: boolean
}

// One WebSocket connection for the app's lifetime -- outside the store's
// state itself since it's an imperative resource, not serializable data.
const voiceClient = new VoiceClient()
const VOICE_ERROR_RESET_DELAY_MS = 3000

// PageViewRequest.requestId's source -- module-level like voiceClient,
// since it's a monotonic counter, not serializable state.
let pageViewRequestCounter = 0

// diagramVoiceRequest's requestId source -- same pattern as pageViewRequestCounter.
let diagramVoiceRequestCounter = 0

// Each sendVoiceAudio() call captures the current value; cancelVoiceTurn()
// bumps it. When the backend's response for a bumped-past turn finally
// arrives, sendVoiceAudio notices its captured id no longer matches and
// skips playback/voiceState -- the turn was interrupted, but its text
// still gets appended (see cancelVoiceTurn's doc comment below).
let currentVoiceTurnId = 0
// The in-flight replay-synthesis fetch (playMessageAudio's synthesizeSpeech
// call), if any -- cancelVoiceTurn() aborts it. Module-level like
// voiceClient/currentVoiceTurnId: an imperative handle, not serializable state.
let activeReplayAbortController: AbortController | null = null

// The in-flight drawDiagram() request, if any -- module-level like the
// handles above. Aborted when a new drawDiagram() call starts for a
// different message, so switching which message is active before a slow
// diagram call finishes doesn't let a stale response land later.
let activeDiagramRequest: { messageIndex: number; controller: AbortController } | null = null

interface AppState {
  subjects: Subject[]
  activeSubjectId: number | null
  sessionId: string | null
  messages: ChatMessage[]
  isLoading: boolean
  theme: Theme
  role: Role
  // Applies to the NEXT message sent, not retroactively to prior turns.
  teachingMode: TeachingMode
  // null means "follow the latest assistant message" -- the Visual
  // Companion's default; set by clicking an older message in ChatView.
  activeMessageIndex: number | null
  // The active "Page N" badge click, if any -- cleared when the active
  // message changes (see VisualCompanion.tsx). null when no page is
  // currently requested for display.
  pageView: PageViewRequest | null
  // On-demand diagram state per assistant message index (see DiagramState) --
  // absent entries are equivalent to 'idle' (never requested).
  diagramsByMessageIndex: Record<number, DiagramState>
  // Set when a voice "draw a diagram" command resolves against a real prior
  // answer -- VisualCompanion.tsx reacts to a new requestId (same "react
  // once per click" pattern as PageViewRequest) by switching to the Visuals
  // tab and expanding the panel, since the student didn't click anything to
  // open it themselves. null otherwise.
  diagramVoiceRequest: { requestId: number; messageIndex: number } | null
  // Whether the Visual Companion's page-view lightbox is open -- lifted
  // into the store (rather than TextbookPageCard's own local state) so
  // VoiceButton's Esc handler can defer to it ("let the lightbox handle
  // Esc first" instead of also interrupting voice on the same keypress).
  isPageLightboxOpen: boolean

  // Documents fetched lazily as SubjectNav expands each subject.
  documentsBySubjectId: Record<number, Document[]>
  sessions: SessionPreview[]
  sessionsLoading: boolean
  chatError: string | null
  // True when the last send failed because the backend was unreachable
  // (as opposed to a normal HTTP error) -- ChatView shows a retry banner.
  connectionError: boolean

  isUploadPanelOpen: boolean
  uploadPanelSubject: string | null
  isSettingsPanelOpen: boolean

  voiceState: VoiceState
  voiceError: string | null
  // Whether the mic has been granted permission at least once.
  voiceEnabled: boolean

  // Index of the message currently being synthesized/replayed via
  // playMessageAudio, or null when no replay is in flight -- lets ChatView
  // show a per-message spinner instead of a global one.
  activePlaybackIndex: number | null
  // TTS voice id (see AVAILABLE_VOICES) to use for both the voice WebSocket
  // and on-demand replay/preview synthesis; null = provider default.
  selectedVoice: string | null
  availableVoices: VoiceOption[]
  // Settings -> Voice's "Interrupt by speaking" toggle (default off,
  // persisted) -- when on, VoiceButton opens a barge-in mic monitor while
  // voiceState is 'playing'. See VoiceButton.tsx's BARGE_IN_* constants.
  interruptBySpeaking: boolean

  loadSubjects: () => Promise<void>
  setActiveSubject: (subjectId: number | null) => void
  sendMessage: (message: string, options?: SendMessageOptions) => Promise<void>
  retryLastMessage: () => Promise<void>
  setTheme: (theme: Theme) => void
  setRole: (role: Role) => void
  setTeachingMode: (mode: TeachingMode) => void
  setActiveMessageIndex: (index: number | null) => void
  viewTextbookPage: (messageIndex: number, documentId: number, pageNumber: number) => void
  clearPageView: () => void
  setPageLightboxOpen: (open: boolean) => void
  // Draws (or re-draws, with force=true) a Mermaid diagram for one assistant
  // message -- POSTs /api/diagram, validates the result in the browser via
  // mermaid.parse(), and retries once with the parse error if that fails.
  // A no-op if that message already has a non-idle, non-failed result and
  // force isn't set (switching back to an already-drawn message shouldn't
  // re-request it).
  drawDiagram: (messageIndex: number, force?: boolean) => Promise<void>
  loadSubjectDocuments: (subjectId: number) => Promise<void>
  openUploadPanel: (subjectName?: string | null) => void
  closeUploadPanel: () => void
  openSettingsPanel: () => void
  closeSettingsPanel: () => void

  loadSessions: () => Promise<void>
  resumeSession: (sessionId: string) => Promise<void>
  startNewChat: () => void

  startVoiceRecording: () => void
  stopVoiceRecording: () => void
  sendVoiceAudio: (audioBlob: Blob) => Promise<void>
  setVoiceState: (voiceState: VoiceState) => void
  setVoiceEnabled: (voiceEnabled: boolean) => void
  // Stops whatever's currently playing (a voice-turn reply or a 🔊 replay)
  // and goes idle. Safe to call in any state.
  stopSpeaking: () => void
  // Cancels a reply that's still being prepared: a live voice turn
  // (sendVoiceAudio, waiting on ASR+chat+TTS) or a replay's on-demand
  // synthesis (playMessageAudio). Safe to call in any state; a no-op if
  // nothing is processing.
  cancelVoiceTurn: () => void

  playMessageAudio: (messageIndex: number) => Promise<void>
  loadVoices: () => Promise<void>
  setSelectedVoice: (voiceId: string | null) => void
  setInterruptBySpeaking: (value: boolean) => void
  previewVoice: (voiceId: string) => Promise<void>
}

// Runs one drawDiagram() attempt end-to-end: POST /api/diagram, then
// mermaid.parse() the result in the browser, retrying once (with the parse
// error fed back to the model) if parsing fails. Returns null if `signal`
// was aborted partway through -- the caller must not commit stale state in
// that case, since something newer already superseded this request.
async function fetchAndValidateDiagram(
  messageText: string,
  sessionId: string | null,
  signal: AbortSignal,
): Promise<DiagramState | null> {
  let previousError: string | null = null

  for (let attempt = 0; attempt < 2; attempt += 1) {
    let response: Awaited<ReturnType<typeof requestDiagram>>
    try {
      response = await requestDiagram({ sessionId, messageText, previousError }, signal)
    } catch (error) {
      if (signal.aborted) return null
      return {
        status: 'failed',
        error: error instanceof Error ? error.message : 'Something went wrong drawing that diagram.',
      }
    }
    if (signal.aborted) return null

    if (response.mermaid === null) {
      return response.reason === 'not_applicable'
        ? { status: 'not_applicable' }
        : { status: 'failed', error: response.reason ?? undefined }
    }

    const parseError = await parseMermaidDiagram(response.mermaid)
    if (signal.aborted) return null
    if (parseError === null) return { status: 'ready', mermaid: response.mermaid }

    previousError = parseError
  }

  return { status: 'failed', error: previousError ?? undefined }
}

const initialRole = readStoredRole()
setApiRole(initialRole)

export const useAppStore = create<AppState>((set, get) => ({
  subjects: [],
  activeSubjectId: null,
  sessionId: null,
  messages: [],
  isLoading: false,
  theme: 'light',
  role: initialRole,
  teachingMode: 'textbook',
  activeMessageIndex: null,
  pageView: null,
  diagramsByMessageIndex: {},
  diagramVoiceRequest: null,
  isPageLightboxOpen: false,

  documentsBySubjectId: {},
  sessions: [],
  sessionsLoading: false,
  chatError: null,
  connectionError: false,

  isUploadPanelOpen: false,
  uploadPanelSubject: null,
  isSettingsPanelOpen: false,

  voiceState: 'idle',
  voiceError: null,
  voiceEnabled: false,

  activePlaybackIndex: null,
  selectedVoice: readStoredVoice(),
  availableVoices: [],
  interruptBySpeaking: readStoredInterruptBySpeaking(),

  loadSubjects: async () => {
    try {
      const subjects = await getSubjects()
      set({ subjects })
    } catch {
      // Callers use this fire-and-forget (SubjectNav's initial load, the
      // upload panel's post-upload refresh) -- swallow so a backend hiccup
      // doesn't surface as an unhandled promise rejection.
    }
  },

  setActiveSubject: (subjectId) => {
    // Typed-chat subject switches happen out from under any open voice
    // connection (which is still tied to the old session/subject) -- drop
    // it so the next voice turn reconnects fresh under the new subject.
    voiceClient.disconnect()
    set({
      activeSubjectId: subjectId,
      messages: [],
      sessionId: null,
      chatError: null,
      connectionError: false,
      voiceState: 'idle',
      voiceError: null,
      activeMessageIndex: null,
      diagramsByMessageIndex: {},
      sessions: [],
    })
    if (subjectId !== null) void get().loadSessions()
  },

  sendMessage: async (message, options) => {
    const skipAppend = options?.skipAppend ?? false
    const { messages } = get()

    set({
      messages: skipAppend ? messages : [...messages, { role: 'user', content: message }],
      isLoading: true,
      chatError: null,
      connectionError: false,
    })

    const { activeSubjectId, subjects, sessionId, teachingMode } = get()
    const subjectName = subjects.find((subject) => subject.id === activeSubjectId)?.name ?? null

    try {
      const response = await chatRequest(message, sessionId, subjectName, teachingMode)
      set((state) => ({
        messages: [
          ...state.messages,
          {
            role: 'assistant',
            content: response.reply,
            sources: response.sources,
            visual_directives: response.visual_directives,
          },
        ],
        sessionId: response.session_id,
        isLoading: false,
        activeMessageIndex: null,
      }))
      void get().loadSessions()
    } catch (error) {
      if (error instanceof NetworkError) {
        set({ isLoading: false, connectionError: true })
      } else {
        set({
          isLoading: false,
          chatError: error instanceof Error ? error.message : 'Something went wrong sending that message.',
        })
      }
    }
  },

  retryLastMessage: async () => {
    const lastUserMessage = [...get().messages].reverse().find((message) => message.role === 'user')
    if (!lastUserMessage) return
    await get().sendMessage(lastUserMessage.content, { skipAppend: true })
  },

  setTheme: (theme) => set({ theme }),

  setTeachingMode: (mode) => set({ teachingMode: mode }),

  setRole: (role) => {
    setApiRole(role)
    try {
      localStorage.setItem(ROLE_STORAGE_KEY, role)
    } catch {
      // Per-viewer convenience only -- losing the persisted preference isn't fatal.
    }
    set({ role })
  },

  setActiveMessageIndex: (index) => set({ activeMessageIndex: index }),

  viewTextbookPage: (messageIndex, documentId, pageNumber) => {
    pageViewRequestCounter += 1
    set({ pageView: { requestId: pageViewRequestCounter, messageIndex, documentId, pageNumber } })
  },

  clearPageView: () => set({ pageView: null }),

  setPageLightboxOpen: (open) => set({ isPageLightboxOpen: open }),

  drawDiagram: async (messageIndex, force = false) => {
    const state = get()
    const message = state.messages[messageIndex]
    if (!message || message.role !== 'assistant') return

    const existing = state.diagramsByMessageIndex[messageIndex]
    if (!force && existing && existing.status !== 'idle' && existing.status !== 'failed') return

    // A different message's diagram is still in flight -- it's no longer
    // the one the student is looking at, so drop it rather than let a late
    // response land against the wrong message later.
    if (activeDiagramRequest && activeDiagramRequest.messageIndex !== messageIndex) {
      activeDiagramRequest.controller.abort()
    }
    const controller = new AbortController()
    activeDiagramRequest = { messageIndex, controller }

    set((current) => ({
      diagramsByMessageIndex: { ...current.diagramsByMessageIndex, [messageIndex]: { status: 'loading' } },
    }))

    const result = await fetchAndValidateDiagram(message.content, get().sessionId, controller.signal)

    if (activeDiagramRequest?.controller === controller) activeDiagramRequest = null
    if (result === null) return // superseded by a newer request -- don't overwrite its state

    set((current) => ({
      diagramsByMessageIndex: { ...current.diagramsByMessageIndex, [messageIndex]: result },
    }))
  },

  loadSubjectDocuments: async (subjectId) => {
    try {
      const documents = await getSubjectDocuments(subjectId)
      set((state) => ({
        documentsBySubjectId: { ...state.documentsBySubjectId, [subjectId]: documents },
      }))
    } catch {
      // SubjectNav calls this fire-and-forget when expanding a subject --
      // swallow so a backend hiccup doesn't surface as an unhandled
      // promise rejection; the section just stays on its loading/empty state.
    }
  },

  openUploadPanel: (subjectName = null) => set({ isUploadPanelOpen: true, uploadPanelSubject: subjectName }),
  closeUploadPanel: () => set({ isUploadPanelOpen: false }),

  openSettingsPanel: () => set({ isSettingsPanelOpen: true }),
  closeSettingsPanel: () => set({ isSettingsPanelOpen: false }),

  loadSessions: async () => {
    const state = get()
    if (!state.activeSubjectId) return
    set({ sessionsLoading: true })
    try {
      const sessions = await getSubjectSessions(state.activeSubjectId)
      set({ sessions, sessionsLoading: false })
    } catch {
      // Fire-and-forget from setActiveSubject/sendMessage -- swallow so a
      // backend hiccup doesn't surface as an unhandled promise rejection.
      set({ sessionsLoading: false })
    }
  },

  resumeSession: async (sessionId) => {
    set({ isLoading: true })
    try {
      const messages = await getSessionMessages(sessionId)
      set({
        sessionId,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
        isLoading: false,
        activeMessageIndex: null,
        diagramsByMessageIndex: {},
      })
      voiceClient.disconnect()
    } catch {
      set({ isLoading: false })
    }
  },

  startNewChat: () => {
    set({ sessionId: null, messages: [], activeMessageIndex: null, diagramsByMessageIndex: {} })
    voiceClient.disconnect()
  },

  startVoiceRecording: () => set({ voiceState: 'recording', voiceError: null }),

  stopVoiceRecording: () =>
    set((state) => (state.voiceState === 'recording' ? { voiceState: 'processing' } : {})),

  setVoiceState: (voiceState) => set({ voiceState }),

  setVoiceEnabled: (voiceEnabled) => set({ voiceEnabled }),

  stopSpeaking: () => {
    stopAudioPlayback()
    set({ voiceState: 'idle', activePlaybackIndex: null })
  },

  cancelVoiceTurn: () => {
    // Supersedes any in-flight sendVoiceAudio turn -- its late response, if
    // any, will notice currentVoiceTurnId moved past its own captured id.
    currentVoiceTurnId += 1
    if (activeReplayAbortController) {
      activeReplayAbortController.abort()
      activeReplayAbortController = null
    }
    set({ voiceState: 'idle', voiceError: null, activePlaybackIndex: null })
  },

  sendVoiceAudio: async (audioBlob) => {
    const turnId = ++currentVoiceTurnId
    set({ voiceState: 'processing', voiceError: null })

    const { activeSubjectId, subjects, sessionId, selectedVoice } = get()
    const subjectName = subjects.find((subject) => subject.id === activeSubjectId)?.name ?? null

    try {
      if (!voiceClient.isConnected()) {
        await voiceClient.connect(sessionId, subjectName, selectedVoice)
      }

      const response = await voiceClient.sendAudio(audioBlob)
      // cancelVoiceTurn() may have bumped currentVoiceTurnId past ours while
      // we were waiting -- the student already interrupted (started a new
      // recording, or just went idle). The transcription/reply still get
      // appended as text below (nothing lost), but playback and voiceState
      // are skipped: whatever the student did next already owns those.
      const wasCancelled = turnId !== currentVoiceTurnId

      if (response.is_command && response.new_subject) {
        // A resolved subject switch -- fresh conversation under the new
        // subject, same as switching from the sidebar.
        let matched = get().subjects.find(
          (subject) => subject.name.toLowerCase() === response.new_subject?.toLowerCase(),
        )
        if (!matched) {
          await get().loadSubjects()
          matched = get().subjects.find(
            (subject) => subject.name.toLowerCase() === response.new_subject?.toLowerCase(),
          )
        }
        set({
          activeSubjectId: matched?.id ?? get().activeSubjectId,
          sessionId: response.session_id,
          messages: [{ role: 'assistant', content: response.reply, audioBlob: response.audio }],
          activeMessageIndex: null,
        })
      } else {
        // Captured before the command's own reply is appended below -- for
        // a "draw a diagram" command this is the real answer to diagram,
        // not the "Drawing a diagram for you." confirmation that's about to
        // become the new latest assistant message.
        let priorLastAssistantIndex: number | null = null
        for (let i = get().messages.length - 1; i >= 0; i -= 1) {
          if (get().messages[i].role === 'assistant') {
            priorLastAssistantIndex = i
            break
          }
        }

        set((state) => ({
          messages: [
            ...state.messages,
            { role: 'user', content: response.transcription },
            {
              role: 'assistant',
              content: response.reply,
              sources: response.is_command ? undefined : response.sources,
              visual_directives: response.is_command ? undefined : response.visual_directives,
              audioBlob: response.audio,
            },
          ],
          sessionId: response.session_id,
          activeMessageIndex: null,
        }))

        if (response.is_command && response.command_type === 'draw_diagram' && priorLastAssistantIndex !== null) {
          diagramVoiceRequestCounter += 1
          set({
            activeMessageIndex: priorLastAssistantIndex,
            diagramVoiceRequest: { requestId: diagramVoiceRequestCounter, messageIndex: priorLastAssistantIndex },
          })
          void get().drawDiagram(priorLastAssistantIndex)
        }
      }

      if (wasCancelled) return

      set({ voiceState: 'playing' })
      const outcome = await playAudioBlob(response.audio)
      // 'interrupted' means something else already changed voiceState (a
      // new recording, a replay, another cancel) -- setting idle here would
      // race it.
      if (outcome !== 'interrupted') {
        set((state) => (state.voiceState === 'playing' ? { voiceState: 'idle' } : {}))
      }
    } catch (error) {
      if (turnId !== currentVoiceTurnId) return // cancelled -- don't surface a stale error
      set({
        voiceState: 'error',
        voiceError: error instanceof Error ? error.message : 'Something went wrong with voice input.',
      })
      setTimeout(() => {
        set((state) => (state.voiceState === 'error' ? { voiceState: 'idle' } : {}))
      }, VOICE_ERROR_RESET_DELAY_MS)
    }
  },

  playMessageAudio: async (messageIndex) => {
    const state = get()
    const message = state.messages[messageIndex]
    if (!message || message.role !== 'assistant') return
    // A live voice turn (recording, or preparing its own reply) owns
    // voiceState -- don't fight it. Already-playing audio is fine to
    // interrupt: switching which message plays is the point of clicking a
    // different 🔊 button.
    if (state.voiceState === 'recording' || state.voiceState === 'processing') return
    if (state.voiceState === 'playing') get().stopSpeaking()

    set({ activePlaybackIndex: messageIndex })

    try {
      let audioBlob = message.audioBlob

      if (!audioBlob) {
        set({ voiceState: 'processing' })

        const controller = new AbortController()
        activeReplayAbortController = controller
        try {
          audioBlob = await synthesizeSpeech(message.content, get().selectedVoice, controller.signal)
        } finally {
          if (activeReplayAbortController === controller) activeReplayAbortController = null
        }

        set((current) => {
          const currentMessage = current.messages[messageIndex]
          if (!currentMessage || currentMessage.audioBlob) return {}
          const updatedMessages = [...current.messages]
          updatedMessages[messageIndex] = { ...currentMessage, audioBlob }
          return { messages: updatedMessages }
        })
      }

      // cancelVoiceTurn() may have aborted the synthesis above and already
      // moved on (voiceState back to 'idle', or a different message now
      // active) -- don't resurrect playback the student no longer asked for.
      if (get().voiceState !== 'processing' || get().activePlaybackIndex !== messageIndex) return

      set({ voiceState: 'playing' })
      const outcome = await playAudioBlob(audioBlob)
      if (outcome !== 'interrupted') {
        set((current) => (current.voiceState === 'playing' ? { voiceState: 'idle' } : {}))
      }
    } catch {
      set((current) => (current.activePlaybackIndex === messageIndex ? { voiceState: 'idle' } : {}))
    } finally {
      set((current) => (current.activePlaybackIndex === messageIndex ? { activePlaybackIndex: null } : {}))
    }
  },

  loadVoices: async () => {
    try {
      const availableVoices = await getAvailableVoices()
      set({ availableVoices })
    } catch {
      // Settings panel just shows no voice picker if this fails.
    }
  },

  setSelectedVoice: (voiceId) => {
    try {
      if (voiceId === null) {
        localStorage.removeItem(VOICE_STORAGE_KEY)
      } else {
        localStorage.setItem(VOICE_STORAGE_KEY, voiceId)
      }
    } catch {
      // Per-viewer convenience only -- losing the persisted preference isn't fatal.
    }
    set({ selectedVoice: voiceId })
  },

  setInterruptBySpeaking: (value) => {
    try {
      localStorage.setItem(INTERRUPT_BY_SPEAKING_STORAGE_KEY, String(value))
    } catch {
      // Per-viewer convenience only -- losing the persisted preference isn't fatal.
    }
    set({ interruptBySpeaking: value })
  },

  previewVoice: async (voiceId) => {
    try {
      const audioBlob = await synthesizeSpeech("Hello! I'm your tutor. Let's study together.", voiceId)
      await playAudioBlob(audioBlob)
    } catch {
      // A failed preview just stays silent -- not worth surfacing an error for.
    }
  },
}))
