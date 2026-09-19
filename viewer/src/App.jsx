import { useEffect, useMemo, useState } from 'react'
import SegmentList from './components/SegmentList.jsx'
import SegmentDetail from './components/SegmentDetail.jsx'
import SpanPopover from './components/SpanPopover.jsx'
import * as ann from './lib/annotations.js'
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
  const [fullTextIndex, setFullTextIndex] = useState(null)
  const [error, setError] = useState(null)

  const [query, setQuery] = useState('')
  const [activeTypes, setActiveTypes] = useState(() => new Set())
  const [selectedId, setSelectedId] = useState(null)
  const [reviewedIds, setReviewedIds] = useState(() => new Set())
  // 29% of segments cannot be found in the submission they claim to quote.
  // They are hidden rather than dropped: the match may fail because the PDF
  // extracted badly, not because the wording was invented.
  const [showUnverified, setShowUnverified] = useState(false)
  const [untaggedOnly, setUntaggedOnly] = useState(false)

  const [annotations, setAnnotations] = useState(() => ann.load())
  // 'local' until the backend answers; shown in the header so a labelling
  // session never has to guess where its work is going.
  const [store, setStore] = useState('local')
  const [pendingSpan, setPendingSpan] = useState(null)
  // Remounts the tag input after a save, which clears it and returns focus
  // without the component having to reach outside itself.
  const [tagFocusKey, setTagFocusKey] = useState(0)

  // localStorage is written either way, as a local mirror of whatever the
  // backend holds.
  useEffect(() => ann.save(annotations), [annotations])

  useEffect(() => {
    let cancelled = false
    ann.backendUp().then(async (up) => {
      if (cancelled || !up) return
      try {
        const remote = await ann.loadRemote()
        if (cancelled) return
        // Anything made offline is pushed up, then the server's view wins.
        const remoteIds = new Set(remote.map((a) => a.ann_id))
        const unsynced = ann.load().filter((a) => !remoteIds.has(a.ann_id))
        await Promise.all(unsynced.map((a) => ann.putRemote(a)))
        setAnnotations(unsynced.length ? [...remote, ...unsynced] : remote)
        setStore('backend')
      } catch {
        // Keep working against localStorage rather than losing the session.
      }
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    Promise.all([loadJson('documents.json'), loadJson('segments.json')])
      .then(([docs, segs]) => {
        setDocuments(docs)
        setSegments(segs)
      })
      .catch((err) => setError(err.message))

    // Lets the Read more line state the length before fetching. This lives at
    // the site root, not under DATA_DIR, so it does not go through loadJson.
    // On failure the index stays null, which means "length unknown" rather
    // than "no text": Read more is still offered and the fetch decides.
    fetch('/fulltext/index.json')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((ix) => setFullTextIndex(ix.documents ?? {}))
      .catch(() => setFullTextIndex(null))
  }, [])

  const docsById = useMemo(
    () => new Map((documents ?? []).map((d) => [d.doc_id, d])),
    [documents],
  )

  // Segments grouped by document, so the detail panel can offer the siblings.
  // 150 of 342 documents hold more than one segment.
  const segmentsByDoc = useMemo(() => {
    const map = new Map()
    for (const seg of segments ?? []) {
      const list = map.get(seg.doc_id)
      if (list) list.push(seg)
      else map.set(seg.doc_id, [seg])
    }
    return map
  }, [segments])

  const orgTypes = useMemo(() => {
    const counts = new Map()
    for (const doc of documents ?? []) {
      const t = doc.org_type_ifp ?? 'Unknown'
      counts.set(t, (counts.get(t) ?? 0) + 1)
    }
    return [...counts].sort((a, b) => b[1] - a[1])
  }, [documents])

  // Lower-casing 5,166 segments on every keystroke is wasteful, so the
  // searchable form is built once and reused.
  const entries = useMemo(() => {
    if (!segments) return []
    return segments.map((segment) => {
      const doc = docsById.get(segment.doc_id)
      return {
        segment,
        doc,
        haystack: `${segment.title}\n${doc?.org ?? ''}\n${segment.text}`.toLowerCase(),
      }
    })
  }, [segments, docsById])

  const unverifiedCount = useMemo(
    () => entries.filter(({ segment }) => segment.source_match === 'none').length,
    [entries],
  )

  // Annotations keyed by segment, so a row and the detail panel can both ask
  // what is on a segment without walking the whole list.
  const annBySegment = useMemo(() => {
    const map = new Map()
    for (const a of annotations) {
      const list = map.get(a.segment_id)
      if (list) list.push(a)
      else map.set(a.segment_id, [a])
    }
    return map
  }, [annotations])

  const registry = useMemo(() => ann.registry(annotations), [annotations])

  const rows = useMemo(() => {
    const needle = query.toLowerCase()
    return entries.filter(({ segment, doc, haystack }) => {
      if (!showUnverified && segment.source_match === 'none') return false
      if (untaggedOnly && annBySegment.has(segment.segment_id)) return false
      if (activeTypes.size > 0 && !activeTypes.has(doc?.org_type_ifp ?? 'Unknown')) return false
      return needle === '' || haystack.includes(needle)
    })
  }, [entries, activeTypes, query, showUnverified, untaggedOnly, annBySegment])

  // The selection is stored as an id but always read back as a position in
  // `rows`. That keeps a selected segment selected when the filters change,
  // and gives J/K a well-defined next and previous row (added 9/18).
  const selectedIndex = rows.findIndex((r) => r.segment.segment_id === selectedId)
  const selected = selectedIndex === -1 ? null : rows[selectedIndex]

  const siblings = selected
    ? (segmentsByDoc.get(selected.segment.doc_id) ?? []).filter(
        (s) => s.segment_id !== selected.segment.segment_id,
      )
    : []

  function toggleType(type) {
    setActiveTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }

  // Labelling runs on the keyboard: J and K move, Enter in the tag input saves
  // and advances. rfi-corpus-spec.md §5.1 puts the difference at five hours
  // over a thousand segments.
  function step(delta) {
    if (rows.length === 0) return
    const next = selectedIndex === -1 ? 0 : Math.min(Math.max(selectedIndex + delta, 0), rows.length - 1)
    setSelectedId(rows[next].segment.segment_id)
    setPendingSpan(null)
    setTagFocusKey((k) => k + 1)
  }

  useEffect(() => {
    function onKeyDown(e) {
      // Never steal a key from a field the user is typing in, except the
      // navigation the tag input deliberately forwards.
      const tag = e.target.tagName
      const typing = tag === 'INPUT' || tag === 'TEXTAREA'
      if (typing && e.target.id !== 'tag-input') return
      if (e.metaKey || e.ctrlKey || e.altKey) return

      if (!typing && (e.key === 'j' || e.key === 'ArrowDown')) {
        e.preventDefault()
        step(1)
      } else if (!typing && (e.key === 'k' || e.key === 'ArrowUp')) {
        e.preventDefault()
        step(-1)
      } else if (e.key === 'Escape') {
        setPendingSpan(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  function record(annotation) {
    setAnnotations((prev) => [...prev, annotation])
    if (store === 'backend') ann.putRemote(annotation).catch(() => setStore('local'))
  }

  function tagSegment(tag) {
    if (!selected) return
    record(ann.createForSegment(selected.segment, tag))
    step(1)
  }

  function tagSpan(tag) {
    if (!selected || !pendingSpan) return
    record(
      ann.create({
        segment: selected.segment,
        tag,
        start: pendingSpan.start,
        end: pendingSpan.end,
        quote: pendingSpan.quote,
      }),
    )
    setPendingSpan(null)
    window.getSelection()?.removeAllRanges()
  }

  function removeTag(annId) {
    setAnnotations((prev) => prev.filter((a) => a.ann_id !== annId))
    if (store === 'backend') ann.deleteRemote(annId).catch(() => setStore('local'))
  }

  function toggleReviewed() {
    if (!selected) return
    setReviewedIds((prev) => {
      const next = new Set(prev)
      const id = selected.segment.segment_id
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  if (error) {
    return (
      <main className="state state--error">
        <h1>Could not load the corpus</h1>
        <pre>{error}</pre>
        <p>Check that pipeline/data/*.json exist and that the dev server is running.</p>
      </main>
    )
  }

  if (!documents || !segments) return <main className="state">Loading corpus…</main>

  return (
    <div className="app">
      <header className="header">
        <h1>AI Action Plan RFI</h1>
        <p className="docket">
          90 FR 9088 · NITRD · retrieved 2026-09-12 · nfkc-v1
          <span className={`store store--${store}`}>
            {store === 'backend' ? 'annotations → SQLite' : 'annotations → this browser'}
          </span>
        </p>

        <div className="tools">
          <div className="search">
            <input
              id="q"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search titles, organizations and recommendation text"
              aria-label="Search the corpus"
            />
          </div>
          <span className="count">
            {rows.length.toLocaleString()} of {segments.length.toLocaleString()} segments
          </span>
          <button
            type="button"
            className="chip"
            aria-pressed={untaggedOnly}
            onClick={() => setUntaggedOnly((v) => !v)}
            title="Only segments with no tag yet. Labelling happens over several sittings."
          >
            Untagged only
          </button>
          <button
            type="button"
            className="chip"
            aria-pressed={showUnverified}
            onClick={() => setShowUnverified((v) => !v)}
            title="Segments whose wording could not be found in the submission PDF. Hidden by default."
          >
            Unverified <i>{unverifiedCount.toLocaleString()}</i>
          </button>
        </div>

        <div className="chips">
          {orgTypes.map(([type, n]) => (
            <button
              key={type}
              type="button"
              className="chip"
              aria-pressed={activeTypes.has(type)}
              onClick={() => toggleType(type)}
            >
              {type} <i>{n}</i>
            </button>
          ))}
        </div>
      </header>

      <div className="split">
        <div className="panel panel--list">
          <SegmentList
            rows={rows}
            selectedId={selectedId}
            onSelect={setSelectedId}
            query={query}
          />
        </div>
        <div className="panel panel--detail">
          <SegmentDetail
            segment={selected?.segment ?? null}
            doc={selected?.doc ?? null}
            siblings={siblings}
            query={query}
            reviewed={selected ? reviewedIds.has(selected.segment.segment_id) : false}
            onToggleReviewed={toggleReviewed}
            onSelect={setSelectedId}
            fullTextChars={
              fullTextIndex && selected
                ? (fullTextIndex[selected.segment.doc_id]?.chars ?? null)
                : undefined
            }
            annotations={selected ? (annBySegment.get(selected.segment.segment_id) ?? []) : []}
            registry={registry}
            onTagSegment={tagSegment}
            onSelectSpan={setPendingSpan}
            onRemoveTag={removeTag}
            tagFocusKey={tagFocusKey}
          />
        </div>
      </div>

      {pendingSpan && (
        <SpanPopover
          at={pendingSpan.at}
          quote={pendingSpan.quote}
          registry={registry}
          onSave={tagSpan}
          onDismiss={() => setPendingSpan(null)}
        />
      )}

      <p className="footnote">
        <button type="button" className="linkish" onClick={() => ann.download(annotations)}>
          Export {annotations.length.toLocaleString()} annotations
        </button>
        {' · '}
        {documents.length.toLocaleString()} documents · {segments.length.toLocaleString()}{' '}
        segments · segment text comes from the IFP database and is shown only where it
        was found in the submission itself · organization type is an LLM-derived field
        and is not used as an analysis variable
      </p>
    </div>
  )
}
