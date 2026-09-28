// Lazy-loaded mermaid singleton -- the Visual Companion's on-demand "Draw
// diagram" flow (store.ts's drawDiagram, VisualsTab.tsx's DiagramCard) is
// the only feature that needs mermaid, so it's dynamically imported on
// first use rather than bundled into the main chunk (a separate chunk per
// vite's default code-splitting for a dynamic import()).

import type { Theme } from './theme'

type MermaidModule = typeof import('mermaid')
type MermaidApi = MermaidModule['default']

let mermaidPromise: Promise<MermaidApi> | null = null
let initializedTheme: 'dark' | 'neutral' | null = null

function mermaidTheme(theme: Theme): 'dark' | 'neutral' {
  return theme === 'dark' ? 'dark' : 'neutral'
}

async function loadMermaid(theme: Theme): Promise<MermaidApi> {
  mermaidPromise ??= import('mermaid').then((module) => module.default)
  const mermaid = await mermaidPromise

  const wantedTheme = mermaidTheme(theme)
  if (initializedTheme !== wantedTheme) {
    mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: wantedTheme })
    initializedTheme = wantedTheme
  }
  return mermaid
}

// Validates Mermaid source without rendering it -- store.ts's drawDiagram
// uses this to decide whether to retry the model once with the parse error
// before giving up. Theme doesn't affect parsing, so this always loads with
// a fixed theme (still needed once, to initialize mermaid at all).
export async function parseMermaidDiagram(source: string): Promise<string | null> {
  try {
    const mermaid = await loadMermaid('light')
    await mermaid.parse(source)
    return null
  } catch (error) {
    return error instanceof Error ? error.message : 'Failed to parse this diagram.'
  }
}

export async function renderMermaidDiagram(id: string, source: string, theme: Theme): Promise<string> {
  const mermaid = await loadMermaid(theme)
  const { svg } = await mermaid.render(id, source)
  return svg
}
