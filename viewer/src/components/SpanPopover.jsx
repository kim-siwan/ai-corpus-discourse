import { useEffect, useRef, useState } from 'react'

// Tagging a span rather than the whole segment: for pointing at a particular
// phrase. The whole-segment path is the one that produces training labels, so
// this stays a small floating box rather than anything modal — covering the
// passage while tagging it is how a span gets tagged wrongly.
export default function SpanPopover({ at, quote, registry, onSave, onDismiss }) {
  const [value, setValue] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    ref.current?.focus()
  }, [at])

  const matches = registry.filter(([tag]) => tag.includes(value.toLowerCase())).slice(0, 4)

  return (
    <div className="pop" style={{ left: at.left, top: at.top }} role="dialog" aria-label="Tag selection">
      <p className="pop-quote">{quote.length > 90 ? `${quote.slice(0, 90)}…` : quote}</p>
      <input
        ref={ref}
        type="text"
        value={value}
        placeholder="Tag this passage"
        aria-label="Tag this passage"
        autoComplete="off"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && value.trim()) onSave(value.trim())
          if (e.key === 'Escape') onDismiss()
        }}
      />
      {matches.length > 0 && (
        <ul className="pop-sugg">
          {matches.map(([tag, n]) => (
            <li key={tag}>
              <button type="button" onMouseDown={(e) => { e.preventDefault(); onSave(tag) }}>
                {tag} <i>{n}</i>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
