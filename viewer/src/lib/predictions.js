// Sentence-type predictions (HW2): fact / policy / testimony / value.
//
// Two sources, one shape. The live endpoints on the backend (POST /predict,
// GET /predict/document/:id) run the trained network on request; the files in
// public/predictions/ hold the same network's output for every paragraph,
// computed once by backend/predict_corpus.py. The files feed the list
// filters, and stand in for the live call when the backend is off.
//
// The server counts offsets in code points (Python len()); the DOM counts
// UTF-16 units. Everything that reaches a component goes through toUtf16().

const API = import.meta.env.VITE_API ?? 'http://localhost:8000'

export const LABELS = ['fact', 'policy', 'testimony', 'value']

export const UNVALIDATED =
  'Model trained on CDCP citizen comments (CFPB rule), not validated on these submissions.'

// Whether predictions are shown at all. Off while hand-coding, so the model's
// guess is not on screen when the coder makes theirs. Remembered per browser.
const SHOW_KEY = 'rfi-show-predictions'

export function loadShow() {
  try {
    return localStorage.getItem(SHOW_KEY) !== 'off'
  } catch {
    return true
  }
}

export function saveShow(on) {
  try {
    localStorage.setItem(SHOW_KEY, on ? 'on' : 'off')
  } catch {
    /* the toggle still works for this session */
  }
}

/** Can the backend predict? /health answers from either Python environment;
 *  /predict/status says whether the model is actually loaded. */
export async function status() {
  try {
    const res = await fetch(`${API}/predict/status`, { signal: AbortSignal.timeout(1500) })
    if (!res.ok) return { available: false, reason: 'backend has no /predict/status — restart it' }
    return await res.json()
  } catch {
    return { available: false, reason: 'backend is not running' }
  }
}

async function call(url, init) {
  const t0 = performance.now()
  const res = await fetch(url, init)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail ?? `HTTP ${res.status}`)
  return { ...body, ms: Math.round(performance.now() - t0) }
}

/** Live: any text. */
export function predictText(text) {
  return call(`${API}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
}

/** Live: every paragraph of one submission, in one request.
 *  -> { n_sentences, ms, segments: { [segment_id]: { sentences, labels, contains_demand } } } */
export function predictDocument(docId, signal) {
  return call(`${API}/predict/document/${encodeURIComponent(docId)}`, { signal })
}

/** Stored: the same network's output, precomputed. Same shape as predictDocument. */
export async function storedDocument(docId, signal) {
  const res = await fetch(`/predictions/docs/${docId}.json`, { signal })
  if (!res.ok) throw new Error(`no stored predictions (HTTP ${res.status})`)
  const file = await res.json()
  const segments = {}
  let n = 0
  for (const [segmentId, rows] of Object.entries(file.segments)) {
    const sentences = rows.map(([start, end, label, p]) => ({
      start,
      end,
      label,
      probs: Object.fromEntries(LABELS.map((l, i) => [l, p[i]])),
    }))
    n += sentences.length
    segments[segmentId] = { sentences, labels: [...new Set(sentences.map((s) => s.label))] }
  }
  return { n_sentences: n, segments }
}

/** Convert code-point offsets to UTF-16 offsets into `text`. */
export function toUtf16(text, sentences) {
  const at = [0] // at[codePoint] = UTF-16 index
  let u = 0
  for (const ch of text) {
    u += ch.length
    at.push(u)
  }
  return sentences.map((s) => ({ ...s, start: at[s.start] ?? u, end: at[s.end] ?? u }))
}

// The same rule as reflow.js: a hyphen that exists only because a word was
// broken across two lines. Kept in the DOM, hidden from the eye.
const SOFTBREAK = /(?<=\p{Ll})-\n(?=\p{Ll})/gu

/** Cut `body` into pieces, each inside at most one sentence and either fully
 *  a soft break or not. Concatenating the pieces gives `body` back exactly,
 *  so textContent — and every annotation offset measured against it — is
 *  unchanged by the tinting. */
export function pieces(body, sentences) {
  const cuts = new Set([0, body.length])
  for (const s of sentences) {
    cuts.add(s.start)
    cuts.add(s.end)
  }
  const soft = []
  for (const m of body.matchAll(SOFTBREAK)) {
    soft.push([m.index, m.index + m[0].length])
    cuts.add(m.index)
    cuts.add(m.index + m[0].length)
  }
  const points = [...cuts].filter((x) => x >= 0 && x <= body.length).sort((a, b) => a - b)
  const out = []
  for (let i = 0; i < points.length - 1; i++) {
    const [a, b] = [points[i], points[i + 1]]
    if (a === b) continue
    out.push({
      text: body.slice(a, b),
      sentence: sentences.find((s) => s.start <= a && b <= s.end) ?? null,
      hidden: soft.some(([x, y]) => x <= a && b <= y),
    })
  }
  return out
}

/** "policy 0.71 · value 0.22 · fact 0.05 · testimony 0.02", highest first. */
export function describe(probs) {
  return Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([label, p]) => `${label} ${p.toFixed(2)}`)
    .join(' · ')
}
