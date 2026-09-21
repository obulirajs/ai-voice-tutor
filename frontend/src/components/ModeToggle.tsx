import { useAppStore } from '../store'
import './ModeToggle.css'

// Segmented toggle between the two teaching modes (prompt_builder.py's
// TEACHING_MODES) -- switching applies to the NEXT message sent, not
// retroactively to messages already in the transcript.
export function ModeToggle() {
  const teachingMode = useAppStore((state) => state.teachingMode)
  const setTeachingMode = useAppStore((state) => state.setTeachingMode)

  return (
    <div className="mode-toggle" role="radiogroup" aria-label="Teaching mode">
      <button
        type="button"
        role="radio"
        aria-checked={teachingMode === 'textbook'}
        className={`mode-toggle__option ${teachingMode === 'textbook' ? 'mode-toggle__option--active' : ''}`}
        onClick={() => setTeachingMode('textbook')}
        title="Quick reference from the textbook"
      >
        <span aria-hidden="true">📖</span> Textbook
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={teachingMode === 'teacher'}
        className={`mode-toggle__option ${teachingMode === 'teacher' ? 'mode-toggle__option--active' : ''}`}
        onClick={() => setTeachingMode('teacher')}
        title="Explain like a teacher, with examples"
      >
        <span aria-hidden="true">👨‍🏫</span> Teacher
      </button>
    </div>
  )
}
