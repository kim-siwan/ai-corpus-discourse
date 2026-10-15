// The Model tab's training monitor and interactive training (HW2 optional
// parts). Everything goes through the backend's /train endpoints
// (backend/train_api.py); the training loop itself is training/train.py.

const API = import.meta.env.VITE_API ?? 'http://localhost:8000'

async function call(path, init) {
  const res = await fetch(`${API}${path}`, init)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail
    throw new Error(detail ?? `HTTP ${res.status}`)
  }
  return body
}

/** The served model's record: { config, history, metrics }. */
export function served() {
  return call('/train/served')
}

/** Start a run: { lr, max_epochs, class_weight_power } -> { run_id }. */
export function start(settings) {
  return call('/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
}

/** A run's status and history so far. */
export function run(runId) {
  return call(`/train/${encodeURIComponent(runId)}`)
}

/** The latest run of this server session, or null — lets the panel pick up a run after a reload. */
export function current() {
  return call('/train/current')
}

export function stop(runId) {
  return call(`/train/${encodeURIComponent(runId)}/stop`, { method: 'POST' })
}

/** [[epoch, value], …] for one history key. */
export function points(history, key) {
  return (history ?? []).filter((r) => Number.isFinite(r[key])).map((r) => [r.epoch, r[key]])
}
