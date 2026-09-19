import { useEffect, useRef, useState } from 'react'

// The input the keyboard flow runs through. Enter saves and moves to the next
// segment; a labelling pass is meant to happen without touching the mouse.
export default function TagInput({ registry, onSave, autoFocus, hint }) {
  const [value, setValue] = useState('')
  const [highlight, setHighlight] = useState(0)
  const ref = useRef(null)

  useEffect(() => {
    if (autoFocus) ref.current?.focus()
  }, [autoFocus])

  const matches = value
    ? registry.filter(([tag]) => tag.includes(value.toLowerCase())).slice(0, 6)
    : registry.slice(0, 6)

  function commit(tag) {
    const clean = (tag ?? value).trim()
    if (!clean) return
    onSave(clean)
    setValue('')
    setHighlight(0)
    ref.current?.focus()
  }

  function onKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      commit(matches[highlight] && value === '' ? matches[highlight][0] : value)
    } else if (e.key === 'ArrowDown' && matches.length) {
      e.preventDefault()
      setHighlight((h) => (h + 1) % matches.length)
    } else if (e.key === 'ArrowUp' && matches.length) {
      e.preventDefault()
      setHighlight((h) => (h - 1 + matches.length) % matches.length)
    } else if (e.key === 'Tab' && matches.length) {
      e.preventDefault()
      setValue(matches[highlight][0])
    } else if (e.key === 'Escape') {
      ref.current?.blur()
    }
  }

  return (
    <div className="taginput">
      <input
        ref={ref}
        id="tag-input"
        type="text"
        value={value}
        placeholder="Tag this segment, then Enter"
        aria-label="Tag this segment"
        autoComplete="off"
        onChange={(e) => {
          setValue(e.target.value)
          setHighlight(0)
        }}
        onKeyDown={onKeyDown}
      />
      {matches.length > 0 && (
        <ul className="taginput-sugg">
          {matches.map(([tag, n], i) => (
            <li key={tag}>
              <button
                type="button"
                className={i === highlight ? 'on' : undefined}
                onMouseDown={(e) => {
                  e.preventDefault()
                  commit(tag)
                }}
              >
                {tag} <i>{n}</i>
              </button>
            </li>
          ))}
        </ul>
      )}
      {hint && <p className="taginput-hint">{hint}</p>}
    </div>
  )
}
