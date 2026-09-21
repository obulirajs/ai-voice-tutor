// Renders inline/display LaTeX embedded in a chat message's plain text
// ($...$ and $$...$$) alongside the surrounding words -- the Visual
// Companion panel shows the same formulae prominently, this keeps them
// legible in the flow of the conversation too.

import katex from 'katex'
import 'katex/dist/katex.min.css'

import './MathText.css'

// Mirrors backend/app/orchestration/visual_directives.py's regexes: display
// math first (so it isn't mistaken for two inline expressions), and the
// negative lookahead after the opening inline "$" excludes a dollar amount
// like "$5.00" (a digit immediately follows) without rejecting math that
// simply ends in one (e.g. "$x^2$").
const MATH_PATTERN = /\$\$([^$]+?)\$\$|\$(?!\d)([^$\n]+?)\$/g

interface Segment {
  type: 'text' | 'math'
  value: string
  displayMode: boolean
}

function splitMathSegments(content: string): Segment[] {
  const segments: Segment[] = []
  let lastIndex = 0

  for (const match of content.matchAll(MATH_PATTERN)) {
    const index = match.index
    if (index > lastIndex) {
      segments.push({ type: 'text', value: content.slice(lastIndex, index), displayMode: false })
    }
    if (match[1] !== undefined) {
      segments.push({ type: 'math', value: match[1], displayMode: true })
    } else {
      segments.push({ type: 'math', value: match[2], displayMode: false })
    }
    lastIndex = index + match[0].length
  }

  if (lastIndex < content.length) {
    segments.push({ type: 'text', value: content.slice(lastIndex), displayMode: false })
  }

  return segments
}

function renderMath(expression: string, displayMode: boolean): string | null {
  try {
    return katex.renderToString(expression, { throwOnError: false, displayMode })
  } catch {
    // Anything KaTeX doesn't degrade gracefully on its own (throwOnError
    // handles most malformed LaTeX) -- fall back to the raw source instead
    // of crashing the chat bubble.
    return null
  }
}

export function MathText({ content }: { content: string }) {
  const segments = splitMathSegments(content)

  if (segments.length <= 1 && segments[0]?.type !== 'math') {
    return <>{content}</>
  }

  return (
    <>
      {segments.map((segment, index) => {
        if (segment.type === 'text') {
          return <span key={index}>{segment.value}</span>
        }

        const html = renderMath(segment.value, segment.displayMode)
        if (html === null) {
          return (
            <code key={index} className="math-text__fallback">
              {segment.displayMode ? `$$${segment.value}$$` : `$${segment.value}$`}
            </code>
          )
        }

        return (
          <span
            key={index}
            className={segment.displayMode ? 'math-text__display' : 'math-text__inline'}
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )
      })}
    </>
  )
}
