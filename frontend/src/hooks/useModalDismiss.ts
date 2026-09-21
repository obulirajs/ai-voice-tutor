import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Focusable descendants, in real Tab order. Native radio buttons sharing a
 * `name` are collapsed to a single stop (the checked one, or the first if
 * none is checked) -- the browser's actual Tab behavior for a radio group;
 * the others are reached with arrow keys, not Tab.
 */
function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const all = Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
  const handledRadioGroups = new Set<string>()

  return all.filter((el) => {
    if (!(el instanceof HTMLInputElement) || el.type !== 'radio' || !el.name) {
      return true
    }
    if (handledRadioGroups.has(el.name)) return false
    handledRadioGroups.add(el.name)

    const group = Array.from(container.querySelectorAll<HTMLInputElement>(`input[type="radio"][name="${el.name}"]`))
    const representative = group.find((radio) => radio.checked) ?? group[0]
    return el === representative
  })
}

/**
 * Escape-to-close and Tab focus-trapping for a modal/panel, shared by
 * UploadPanel and SettingsPanel. Focuses the panel's first focusable element
 * on open; Tab/Shift+Tab wrap inside the panel instead of escaping to the
 * rest of the page.
 */
export function useModalDismiss(isOpen: boolean, onClose: () => void): RefObject<HTMLDivElement | null> {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isOpen) return

    const container = containerRef.current
    if (container) {
      getFocusableElements(container)[0]?.focus()
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
        return
      }

      if (event.key !== 'Tab' || !container) return

      const focusableEls = getFocusableElements(container)
      if (focusableEls.length === 0) return

      const first = focusableEls[0]
      const last = focusableEls[focusableEls.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  return containerRef
}
