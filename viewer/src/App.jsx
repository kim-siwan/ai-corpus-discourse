import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import DocumentList from './components/DocumentList.jsx'
import DocumentDetail from './components/DocumentDetail.jsx'
import SpanPopover from './components/SpanPopover.jsx'
import * as ann from './lib/annotations.js'
import { toCodePointOffset } from './lib/annotations.js'
import './App.css'

const DATA = '/data'
const API = import.meta.env.VITE_API ?? 'http://localhost:8000'

async function loadJson(name) {
  const res = await fetch(`${DATA}/${name}`)
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`)
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('json')) throw new Error(`${name}: not JSON (got ${type})`)
  return res.json()
}

const fullTextCache = new Map()

export default function App() {
  const [documents, setDocuments] = useState(null)
  const [paragraphs, setParagraphs] = useState(null)
  const [error, setError] = useState(null)

  const [query, setQuery] = useState('')
  const [bodyHits, setBodyHits] = useState(null) // doc_ids from the backend, or null
  const [activeTypes, setActiveTypes] = useState(() => new Set())
  const [activeTopics, setActiveTopics] = useState(() => new Set())
  // Topics are IFP's labels. They stay off by default so a coder reads the
  // paragraph before reading somebody else's judgement about it.
  const [showTopics, setShowTopics] = useState(false)
  const [sort, setSort] = useState('org')
  const [untaggedOnly, setUntaggedOnly] = useState(false)

  const [selectedDocId, setSelectedDocId] = useState(null)
  const [currentParaId, setCurrentParaId] = useState(null)
  const [text, setText] = useState(null)
  const [textState, setTextState] = useState({ loading: false, error: null })

  const [annotations, setAnnotations] = useState(() => ann.load())
  const [pendingSpan, setPendingSpan] = useState(null)
  const [tagFocusKey, setTagFocusKey] = useState(0)
  const [store, setStore] = useState('local')

  useEffect(() => {
    Promise.all([loadJson('documents.json'), loadJson('segments.json')])
      .then(([docs, paras]) => {
        setDocuments(docs)
        setParagraphs(paras)
      })
      .catch((err) => setError(err.message))
  }, [])

  useEffect(() => ann.save(annotations), [annotations])

  useEffect(() => {
    let cancelled = false
    ann.backendUp().then(async (up) => {
      if (cancelled || !up) return
      try {
        const remote = await ann.loadRemote()
        if (cancelled) return
        const remoteIds = new Set(remote.map((a) => a.ann_id))
        const unsynced = ann.load().filter((a) => !remoteIds.has(a.ann_id))
        await Promise.all(unsynced.map((a) => ann.putRemote(a)))
        setAnnotations(unsynced.length ? [...remote, ...unsynced] : remote)
        setStore('backend')
      } catch {
        /* keep working against localStorage */
      }
    })
    return () => {
      cancelled = true
    }
  }, [])

  const parasByDoc = useMemo(() => {
    const map = new Map()
    for (const p of paragraphs ?? []) {
      const list = map.get(p.doc_id)
      if (list) list.push(p)
      else map.set(p.doc_id, [p])
    }
    return map
  }, [paragraphs])

  const paraCounts = useMemo(
    () => new Map([...parasByDoc].map(([k, v]) => [k, v.length])),
    [parasByDoc],
  )

  // Length is counted in words, not paragraphs: paragraph boundaries come from
  // how the PDF was typeset, so they measure the file as much as the argument.
  const wordCounts = useMemo(
    () =>
      new Map(
        [...parasByDoc].map(([k, v]) => [k, v.reduce((sum, p) => sum + (p.n_words ?? 0), 0)]),
      ),
    [parasByDoc],
  )

  const annBySegment = useMemo(() => {
    const map = new Map()
    for (const a of annotations) {
      const list = map.get(a.segment_id)
      if (list) list.push(a)
      else map.set(a.segment_id, [a])
    }
    return map
  }, [annotations])

  const tagCounts = useMemo(() => {
    const map = new Map()
    for (const a of annotations) map.set(a.doc_id, (map.get(a.doc_id) ?? 0) + 1)
    return map
  }, [annotations])

  const registry = useMemo(() => ann.registry(annotations), [annotations])

  const orgTypes = useMemo(() => {
    const counts = new Map()
    for (const d of documents ?? []) counts.set(d.org_type_official, (counts.get(d.org_type_official) ?? 0) + 1)
    return [...counts].sort((a, b) => b[1] - a[1])
  }, [documents])

  const topics = useMemo(() => {
    const counts = new Map()
    for (const d of documents ?? []) {
      for (const { topic } of d.topics) counts.set(topic, (counts.get(topic) ?? 0) + 1)
    }
    return [...counts].sort((a, b) => b[1] - a[1])
  }, [documents])

  // Body search runs on the backend: the viewer holds offsets, not 14 MB of
  // text, so it can only match what it has — the organization and the topics.
  useEffect(() => {
    const q = query.trim()
    if (store !== 'backend' || q.length < 2) {
      setBodyHits(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      fetch(`${API}/search?q=${encodeURIComponent(q)}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((res) => {
          if (!cancelled) setBodyHits(new Set(res.hits.map((h) => h.doc_id)))
        })
        .catch(() => !cancelled && setBodyHits(null))
    }, 250)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [query, store])

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const matched = (documents ?? []).filter((d) => {
      if (activeTypes.size && !activeTypes.has(d.org_type_official)) return false
      if (activeTopics.size && !d.topics.some(({ topic }) => activeTopics.has(topic))) return false
      if (untaggedOnly && (tagCounts.get(d.doc_id) ?? 0) > 0) return false
      if (!needle) return true
      if (d.org.toLowerCase().includes(needle)) return true
      if (d.topics.some(({ topic }) => topic.toLowerCase().includes(needle))) return true
      return bodyHits ? bodyHits.has(d.doc_id) : false
    })
    return sort === 'words'
      ? [...matched].sort((a, b) => (wordCounts.get(b.doc_id) ?? 0) - (wordCounts.get(a.doc_id) ?? 0))
      : [...matched].sort((a, b) => a.org.localeCompare(b.org, 'en', { sensitivity: 'base' }))
  }, [documents, activeTypes, activeTopics, untaggedOnly, tagCounts, query, bodyHits, sort, wordCounts])

  // One flat sequence of paragraphs across the filtered submissions. A
  // labelling pass runs down it without stopping at document boundaries.
  const paraSeq = useMemo(() => {
    const out = []
    for (const d of rows) for (const p of parasByDoc.get(d.doc_id) ?? []) out.push(p)
    return out
  }, [rows, parasByDoc])

  const seqIndex = useMemo(
    () => paraSeq.findIndex((p) => p.segment_id === currentParaId),
    [paraSeq, currentParaId],
  )

  const selectedDoc = (documents ?? []).find((d) => d.doc_id === selectedDocId) ?? null
  const docParas = parasByDoc.get(selectedDocId) ?? []

  useEffect(() => {
    if (!selectedDocId) return
    if (fullTextCache.has(selectedDocId)) {
      setText(fullTextCache.get(selectedDocId))
      setTextState({ loading: false, error: null })
      return
    }
    let cancelled = false
    setText(null)
    setTextState({ loading: true, error: null })
    fetch(`/fulltext/${selectedDocId}.txt`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        if (!(res.headers.get('content-type') ?? '').includes('text/plain')) {
          throw new Error('unexpected content type')
        }
        return res.text()
      })
      .then((body) => {
        if (cancelled) return
        fullTextCache.set(selectedDocId, body)
        setText(body)
        setTextState({ loading: false, error: null })
      })
      .catch(() => {
        if (cancelled) return
        setTextState({
          loading: false,
          error: 'No extracted text for this submission — it is a scanned PDF or missing from the archive.',
        })
      })
    return () => {
      cancelled = true
    }
  }, [selectedDocId])

  const openDoc = useCallback(
    (docId) => {
      setSelectedDocId(docId)
      setCurrentParaId((parasByDoc.get(docId) ?? [])[0]?.segment_id ?? null)
      setPendingSpan(null)
      setTagFocusKey((k) => k + 1)
    },
    [parasByDoc],
  )

  const step = useCallback(
    (delta) => {
      if (paraSeq.length === 0) return
      const next = seqIndex === -1 ? 0 : Math.min(Math.max(seqIndex + delta, 0), paraSeq.length - 1)
      const p = paraSeq[next]
      setCurrentParaId(p.segment_id)
      if (p.doc_id !== selectedDocId) setSelectedDocId(p.doc_id)
      setPendingSpan(null)
      setTagFocusKey((k) => k + 1)
    },
    [paraSeq, seqIndex, selectedDocId],
  )

  useEffect(() => {
    function onKeyDown(e) {
      const typing = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
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

  function tagParagraph(tag) {
    const p = paraSeq[seqIndex] ?? docParas.find((x) => x.segment_id === currentParaId)
    if (!p) return
    record(ann.createForSegment(p, tag))
    step(1)
  }

  function tagSpan(tag) {
    if (!pendingSpan) return
    record(ann.create({ segment: pendingSpan.para, tag, ...pendingSpan.range }))
    setPendingSpan(null)
    window.getSelection()?.removeAllRanges()
  }

  function removeTag(annId) {
    setAnnotations((prev) => prev.filter((a) => a.ann_id !== annId))
    if (store === 'backend') ann.deleteRemote(annId).catch(() => setStore('local'))
  }

  // A selection inside a paragraph becomes a span annotation. Offsets are
  // measured against the paragraph, so they survive re-rendering.
  function selectSpan(para, node) {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !node.contains(sel.anchorNode)) return
    const quote = sel.toString().trim()
    if (quote.length < 3) return
    const body = node.textContent
    const at = body.indexOf(quote)
    if (at < 0) return
    const rect = sel.getRangeAt(0).getBoundingClientRect()
    setCurrentParaId(para.segment_id)
    setPendingSpan({
      para,
      quote,
      range: {
        start: toCodePointOffset(body, at),
        end: toCodePointOffset(body, at + quote.length),
        quote,
      },
      at: { left: Math.min(rect.left, window.innerWidth - 270), top: rect.bottom + 6 },
    })
  }

  function toggle(setter, value) {
    setter((prev) => {
      const next = new Set(prev)
      if (next.has(value)) next.delete(value)
      else next.add(value)
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
  if (!documents || !paragraphs) return <main className="state">Loading corpus…</main>

  const bodySearchOn = store === 'backend'

  return (
    <div className="app">
      <header className="header">
        <h1>AI Action Plan RFI</h1>
        <p className="docket">
          90 FR 9088 · NITRD · organizational submissions · retrieved 2026-09-12 · nfkc-v1
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
              placeholder={
                bodySearchOn
                  ? 'Search organizations, topics and submission text'
                  : 'Search organizations and topics — start the backend for body search'
              }
              aria-label="Search"
            />
          </div>
          <span className="count">
            {rows.length.toLocaleString()} of {documents.length.toLocaleString()} submissions
          </span>
          <label className="sort">
            Sort
            <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort submissions">
              <option value="org">Organization A–Z</option>
              <option value="words">Longest first</option>
            </select>
          </label>
          <button type="button" className="chip" aria-pressed={untaggedOnly} onClick={() => setUntaggedOnly((v) => !v)}>
            Untagged only
          </button>
          <button
            type="button"
            className="chip"
            aria-pressed={showTopics}
            onClick={() => setShowTopics((v) => !v)}
            title="IFP's topic labels. Turn them off while labelling so they do not steer a judgement about the paragraph in front of you."
          >
            Topics
          </button>
        </div>

        <div className="chips">
          {orgTypes.map(([type, n]) => (
            <button
              key={type}
              type="button"
              className="chip"
              aria-pressed={activeTypes.has(type)}
              onClick={() => toggle(setActiveTypes, type)}
            >
              {type} <i>{n}</i>
            </button>
          ))}
        </div>

        {showTopics && (
          <div className="chips chips--topic">
            {topics.map(([topic, n]) => (
              <button
                key={topic}
                type="button"
                className="chip chip--topic"
                aria-pressed={activeTopics.has(topic)}
                onClick={() => toggle(setActiveTopics, topic)}
              >
                {topic} <i>{n}</i>
              </button>
            ))}
          </div>
        )}
      </header>

      <div className="split">
        <div className="panel panel--list">
          <DocumentList
            rows={rows}
            selectedId={selectedDocId}
            onSelect={openDoc}
            tagCounts={tagCounts}
            paraCounts={paraCounts}
            wordCounts={wordCounts}
            showTopics={showTopics}
          />
        </div>
        <div className="panel panel--detail">
          <DocumentDetail
            doc={selectedDoc}
            paragraphs={docParas}
            text={text}
            loading={textState.loading}
            error={textState.error}
            currentId={currentParaId}
            onSelectParagraph={setCurrentParaId}
            annotationsBySegment={annBySegment}
            registry={registry}
            onTag={tagParagraph}
            onRemoveTag={removeTag}
            onSelectSpan={selectSpan}
            tagFocusKey={tagFocusKey}
            showTopics={showTopics}
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
        {documents.length} submissions · {paragraphs.length.toLocaleString()} paragraphs · respondent
        type is as recorded in the docket; topics are LLM-derived and used for navigation only
      </p>
    </div>
  )
}
