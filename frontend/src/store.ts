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
import { playAudioBlob } from './hooks/useAudioPlayer'
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

const ROLE_STORAGE_KEY = 'project-tutor:role'
const VOICE_STORAGE_KEY = 'project-tutor:voice'

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

  loadSubjects: () => Promise<void>
  setActiveSubject: (subjectId: number | null) => void
  sendMessage: (message: string, options?: SendMessageOptions) => Promise<void>
  retryLastMessage: () => Promise<void>
  setTheme: (theme: Theme) => void
  setRole: (role: Role) => void
  setTeachingMode: (mode: TeachingMode) => void
  setActiveMessageIndex: (index: number | null) => void
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

  playMessageAudio: (messageIndex: number) => Promise<void>
  loadVoices: () => Promise<void>
  setSelectedVoice: (voiceId: string | null) => void
  previewVoice: (voiceId: string) => Promise<void>
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
      })
      voiceClient.disconnect()
    } catch {
      set({ isLoading: false })
    }
  },

  startNewChat: () => {
    set({ sessionId: null, messages: [], activeMessageIndex: null })
    voiceClient.disconnect()
  },

  startVoiceRecording: () => set({ voiceState: 'recording', voiceError: null }),

  stopVoiceRecording: () =>
    set((state) => (state.voiceState === 'recording' ? { voiceState: 'processing' } : {})),

  setVoiceState: (voiceState) => set({ voiceState }),

  setVoiceEnabled: (voiceEnabled) => set({ voiceEnabled }),

  sendVoiceAudio: async (audioBlob) => {
    set({ voiceState: 'processing', voiceError: null })

    const { activeSubjectId, subjects, sessionId, selectedVoice } = get()
    const subjectName = subjects.find((subject) => subject.id === activeSubjectId)?.name ?? null

    try {
      if (!voiceClient.isConnected()) {
        await voiceClient.connect(sessionId, subjectName, selectedVoice)
      }

      const response = await voiceClient.sendAudio(audioBlob)

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
      }

      set({ voiceState: 'playing' })
      await playAudioBlob(response.audio)
      set((state) => (state.voiceState === 'playing' ? { voiceState: 'idle' } : {}))
    } catch (error) {
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
    // Voice recording/playback already own voiceState -- don't fight them.
    if (state.voiceState === 'recording' || state.voiceState === 'processing' || state.voiceState === 'playing') return

    set({ activePlaybackIndex: messageIndex })

    try {
      let audioBlob = message.audioBlob

      if (!audioBlob) {
        set({ voiceState: 'processing' })
        audioBlob = await synthesizeSpeech(message.content, get().selectedVoice)

        set((current) => {
          const currentMessage = current.messages[messageIndex]
          if (!currentMessage || currentMessage.audioBlob) return {}
          const updatedMessages = [...current.messages]
          updatedMessages[messageIndex] = { ...currentMessage, audioBlob }
          return { messages: updatedMessages }
        })
      }

      set({ voiceState: 'playing' })
      await playAudioBlob(audioBlob)
      set((current) => (current.voiceState === 'playing' ? { voiceState: 'idle' } : {}))
    } catch {
      set({ voiceState: 'idle' })
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

  previewVoice: async (voiceId) => {
    try {
      const audioBlob = await synthesizeSpeech("Hello! I'm your tutor. Let's study together.", voiceId)
      await playAudioBlob(audioBlob)
    } catch {
      // A failed preview just stays silent -- not worth surfacing an error for.
    }
  },
}))
