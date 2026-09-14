import { useEffect, useMemo, useState } from 'react'
import './App.css'

const DATA_DIR = '/data'

async function loadJson(name) {
  const res = await fetch(`${DATA_DIR}/${name}`)
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`)
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) throw new Error(`${name}: not JSON (got ${type})`)
  return res.json()
}

export default function App() {
  const [documents, setDocuments] = useState(null)
  const [segments, setSegments] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([loadJson('documents.json'), loadJson('segments.json')])
      .then(([docs, segs]) => {
        setDocuments(docs)
        setSegments(segs)
      })
      .catch((err) => setError(err.message))
  }, [])

  // doc_id -> document, so a segment can show its organization
  const docsById = useMemo(() => {
    if (!documents) return new Map()
    return new Map(documents.map((d) => [d.doc_id, d]))
  }, [documents])

  if (error) {
    return (
      <main className="state state--error">
        <h1>Failed to load corpus</h1>
        <p>{error}</p>
      </main>
    )
  }

  if (!documents || !segments) {
    return <main className="state">Loading corpus…</main>
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI Literacy Discourse</h1>
        <p className="subtitle">
          OSTP AI Action Plan RFI (90 FR 9088) · Talent &amp; Education
        </p>
        <dl className="counts">
          <div>
            <dt>Documents</dt>
            <dd>{documents.length}</dd>
          </div>
          <div>
            <dt>Segments</dt>
            <dd>{segments.length}</dd>
          </div>
          <div>
            <dt>Reviewed</dt>
            <dd>{documents.filter((d) => d.reviewed).length}</dd>
          </div>
        </dl>
      </header>

      <ul className="segments">
        {segments.slice(0, 30).map((seg) => {
          const doc = docsById.get(seg.doc_id)
          return (
            <li key={seg.segment_id} className="segment">
              <div className="segment__meta">
                <span className="segment__id">{seg.segment_id}</span>
                <span className="segment__org">{doc?.org ?? 'Unknown org'}</span>
                <span className="segment__type">{doc?.org_type_ifp ?? '—'}</span>
              </div>
              <h2 className="segment__title">{seg.title}</h2>
              <p className="segment__text">{seg.text.slice(0, 240)}…</p>
            </li>
          )
        })}
      </ul>

      <footer className="footer">
        Showing first 30 of {segments.length} segments.
      </footer>
    </div>
  )
}
