// Stand-off annotations, in the shape rfi-corpus-spec.md fixes for them. They
// live in localStorage until the backend lands; the export below is what moves
// them across, and is also what a labelling session hands to training code.
//
// Offsets are code-point offsets into segment.text, measured the way Python's
// len() counts. JavaScript strings are UTF-16, so a character outside the BMP
// counts as two there and one here. Array.from() splits by code point, which is
// why every offset below goes through it.

const KEY = 'rfi-annotations-v1'
const SCHEME = 'freeform-v0'
const API = import.meta.env.VITE_API ?? 'http://localhost:8000'

// The store is the backend when it answers and localStorage when it does not.
// Writes always reach localStorage as well, so a backend that goes down
// mid-session cannot take the work with it.
export async function backendUp() {
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(1500) })
    return res.ok
  } catch {
    return false
  }
}

export async function loadRemote() {
  const res = await fetch(`${API}/annotations`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function putRemote(annotation) {
  const res = await fetch(`${API}/annotations/${annotation.ann_id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotation),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}

export async function deleteRemote(annId) {
  const res = await fetch(`${API}/annotations/${annId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}

export function load() {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function save(annotations) {
  try {
    localStorage.setItem(KEY, JSON.stringify(annotations))
  } catch {
    // A full or blocked store must not lose the annotation being written, so
    // the caller keeps it in memory either way.
  }
}

export function codePointLength(text) {
  return Array.from(text).length
}

/** Convert a UTF-16 index from the DOM into a code-point offset. */
export function toCodePointOffset(text, utf16Index) {
  return Array.from(text.slice(0, utf16Index)).length
}

export function create({ segment, tag, start, end, quote, coder = 'kim' }) {
  return {
    ann_id: `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    segment_id: segment.segment_id,
    doc_id: segment.doc_id,
    start,
    end,
    quote,
    tag,
    scheme: SCHEME,
    coder,
    created: new Date().toISOString(),
    note: '',
  }
}

/** A tag on the whole segment: start 0, end the segment's length. */
export function createForSegment(segment, tag, coder) {
  return create({
    segment,
    tag,
    start: 0,
    end: codePointLength(segment.text),
    quote: '',
    coder,
  })
}

/** Tags already used, most frequent first, for the autocomplete. */
export function registry(annotations) {
  const counts = new Map()
  for (const a of annotations) counts.set(a.tag, (counts.get(a.tag) ?? 0) + 1)
  return [...counts].sort((a, b) => b[1] - a[1])
}

export function download(annotations) {
  const body = JSON.stringify(
    { scheme: SCHEME, exported: new Date().toISOString(), annotations },
    null,
    1,
  )
  const url = URL.createObjectURL(new Blob([body], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `annotations-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

export async function loadReviewed() {
  const res = await fetch(`${API}/reviewed`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function setReviewed(segmentId, on, coder = 'kim') {
  const res = await fetch(`${API}/reviewed/${segmentId}`, {
    method: on ? 'PUT' : 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: on
      ? JSON.stringify({ segment_id: segmentId, coder, at: new Date().toISOString() })
      : undefined,
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
}
