import { useEffect, useState } from 'react'

// One cache for the whole session. Submissions run to 123,000 characters and a
// reader moves between segments of the same document, so re-fetching on every
// toggle would be wasteful.
const cache = new Map()

// PDFs wrap lines at a fixed column, so a sentence arrives broken across
// several lines. Rejoin them for reading. This is display only: the stored
// text keeps every original newline, because annotation offsets are measured
// against it and rewriting the text would move every offset.
const BULLET = /^\s*(?:[•●○▪◦*-]|o\s|\d+[.)])/

function toBlocks(text) {
  return text.split(/\n{2,}/).map((block) => {
    const lines = block.split('\n')
    const out = []
    for (const line of lines) {
      const prev = out[out.length - 1]
      // Start a new line for a bullet, a heading-ish short line, or the first
      // line; otherwise continue the previous one.
      if (prev === undefined || BULLET.test(line) || /[.:?!]\s*$/.test(prev)) out.push(line.trim())
      else out[out.length - 1] = `${prev} ${line.trim()}`.replace(/\s+/g, ' ')
    }
    return out.filter(Boolean)
  })
}


async function fetchText(docId) {
  const res = await fetch(`/fulltext/${docId}.txt`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  // The dev server answers an unknown path with index.html and a 200, so a
  // broken path would otherwise render as a page of HTML. Check the type.
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('text/plain')) throw new Error(`expected text/plain, got ${type}`)
  return res.text()
}

export default function FullSubmission({ docId, pdfUrl, chars }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState(() => cache.get(docId) ?? null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setOpen(false)
    setText(cache.get(docId) ?? null)
    setError(null)
  }, [docId])

  useEffect(() => {
    if (!open || text !== null || error) return
    let cancelled = false
    fetchText(docId)
      .then((body) => {
        if (cancelled) return
        cache.set(docId, body)
        setText(body)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [open, docId, text, error])

  // undefined: the index did not load, so the length is unknown but the text
  // may well exist. null: the index loaded and this document is not in it —
  // two of the 342 submissions are scanned images with no extractable text.
  if (chars === null) {
    return (
      <p className="readmore-none">
        This submission is a scanned PDF, so there is no extracted text.{' '}
        {pdfUrl && (
          <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
            Open the PDF ↗
          </a>
        )}
      </p>
    )
  }

  if (!open) {
    return (
      <p className="readmore-line">
        <button type="button" className="readmore" onClick={() => setOpen(true)}>
          Read more
        </button>
        <span className="readmore-hint">
          {chars === undefined
            ? 'full submission'
            : `full submission · ${chars.toLocaleString()} characters`}
        </span>
      </p>
    )
  }

  return (
    <div className="fulltext">
      <div className="fulltext-head">
        <span>
          Full submission
          {chars !== undefined && ` · ${chars.toLocaleString()} characters`}
        </span>
        <button type="button" className="readmore" onClick={() => setOpen(false)}>
          Collapse
        </button>
      </div>
      {error ? (
        <p className="readmore-none">
          Could not load the text ({error}).{' '}
          {pdfUrl && (
            <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
              Open the PDF ↗
            </a>
          )}
        </p>
      ) : text === null ? (
        <p className="readmore-none">Loading…</p>
      ) : (
        <div className="fulltext-body">
          {toBlocks(text).map((lines, i) => (
            <p key={i}>
              {lines.map((line, j) => (
                <span key={j}>{line}</span>
              ))}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
