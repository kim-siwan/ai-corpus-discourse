import { useEffect, useState } from 'react'
import LossChart from './LossChart.jsx'
import { current, points, run, served, start, stop } from '../lib/training.js'

// Interactive training: pick a learning rate, a number of epochs and a class
// weight, press Train, and watch train and dev loss move epoch by epoch. The
// backend runs training/train.py's loop — the same one that trained the
// served model — on cached CDCP embeddings. Each run is saved to its own
// folder under training/runs/; the served model is never replaced, and the
// test split is never scored here.
const LRS = [0.01, 0.03, 0.1, 0.3, 1, 3]
const POWERS = [
  [0, 'none'],
  [0.5, '√ balanced'],
  [1, 'balanced'],
]
const POLL_MS = 400

const f3 = (x) => (x == null ? '—' : x.toFixed(3))
const f2 = (x) => (x == null ? '—' : x.toFixed(2))

export default function TrainPanel({ status }) {
  const [lr, setLr] = useState(0.1)
  const [epochs, setEpochs] = useState(100)
  const [power, setPower] = useState(0)
  const [state, setState] = useState(null) // the run as the backend reports it
  const [error, setError] = useState(null)
  const [reference, setReference] = useState(null) // the served model's dev error

  const ready = status?.available
  const running = state?.status === 'running'

  // Pick up the latest run of this server session (e.g. after a page reload),
  // and the served model's dev result to compare against.
  useEffect(() => {
    if (!ready) return
    current()
      .then((r) => r && setState(r))
      .catch(() => {})
    served()
      .then((d) => setReference(d.metrics.dev))
      .catch(() => {})
  }, [ready])

  // While a run is going, ask for its progress a few times a second.
  useEffect(() => {
    if (!running) return
    const id = state.run_id
    const timer = setInterval(() => {
      run(id)
        .then(setState)
        .catch((e) => setError(e.message))
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [running, state?.run_id])

  async function onTrain() {
    setError(null)
    try {
      const settings = { lr, max_epochs: epochs, class_weight_power: power }
      const { run_id } = await start(settings)
      setState({ run_id, status: 'running', config: settings, history: [], epoch: 0 })
    } catch (e) {
      setError(e.message)
    }
  }

  const history = state?.history ?? []
  const last = history.at(-1)
  const result = state?.result
  const best = result?.best_epoch ?? state?.best_epoch_so_far

  return (
    <div className="trainpanel">
      <p className="paste-note">
        Train a new network with your own settings and watch it learn. The backend runs the same seven-step loop
        that trained the served model (<code>training/train.py</code>), on the cached CDCP embeddings, with early
        stopping on dev loss. Runs are saved under <code>training/runs/</code>; the served model is not replaced,
        and the test split is not used here.
      </p>

      <div className="train-form">
        <label className="train-field">
          <span>learning rate</span>
          <select value={lr} onChange={(e) => setLr(Number(e.target.value))} disabled={running}>
            {LRS.map((v) => (
              <option key={v} value={v}>
                {v}
                {v === 0.1 ? ' (served)' : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="train-field">
          <span>max epochs</span>
          <input
            type="number"
            min={1}
            max={300}
            value={epochs}
            disabled={running}
            onChange={(e) => setEpochs(Math.min(300, Math.max(1, Number(e.target.value) || 1)))}
          />
        </label>
        <div className="train-field">
          <span title="w_c = (N / (4 · n_c))^p — how much more a rare class's mistakes cost">class weight p</span>
          <div className="seg" role="radiogroup" aria-label="Class weight power">
            {POWERS.map(([p, label]) => (
              <button
                key={p}
                type="button"
                role="radio"
                aria-checked={power === p}
                disabled={running}
                onClick={() => setPower(p)}
                title={label}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
        <div className="train-actions">
          <button type="button" className="btn btn--primary" onClick={onTrain} disabled={!ready || running}>
            {running ? 'Training…' : 'Train'}
          </button>
          {running && (
            <button type="button" className="btn" onClick={() => stop(state.run_id).catch((e) => setError(e.message))}>
              Stop
            </button>
          )}
        </div>
      </div>
      {!ready && status && <p className="backend-off">Training is off: {status.reason}</p>}
      {error && <p className="state--error paste-error">{error}</p>}

      {state && (
        <div className="train-run">
          <LossChart
            title={state.config?.class_weight_power ? 'Weighted cross-entropy loss' : 'Cross-entropy loss'}
            series={[
              { key: 'train', label: 'train', points: points(history, 'train_loss') },
              { key: 'dev', label: 'dev', points: points(history, 'dev_loss') },
            ]}
            best={best ?? null}
            xMax={Math.max(10, last?.epoch ?? 0)}
          />
          <p className="train-status">
            <RunStatus state={state} />
            {last && (
              <>
                {' '}
                · epoch <b>{last.epoch}</b> · train loss {f3(last.train_loss)} · dev loss {f3(last.dev_loss)} · dev
                error {f3(last.dev_error)} · dev policy F1 {f2(last.dev_policy_f1)}
                {best != null && <> · best so far: epoch {best}</>}
              </>
            )}
          </p>
          {result && (
            <p className="train-result">
              Kept epoch <b>{result.best_epoch}</b> of {result.last_epoch} (
              {result.stop_reason === 'early_stop'
                ? 'early-stopped: dev loss stopped improving'
                : result.stop_reason === 'stopped'
                  ? 'stopped by you'
                  : result.stop_reason === 'diverged'
                    ? 'diverged: the learning rate is too large'
                    : 'reached max epochs — still improving, under-trained'}
              ) → dev error <b>{f3(result.dev_error)}</b>, dev policy P/R/F1 {f2(result.dev_policy_precision)} /{' '}
              {f2(result.dev_policy_recall)} / {f2(result.dev_policy_f1)}
              {reference && (
                <>
                  {' '}
                  — served model: dev error {f3(reference.error)}, policy F1 {f2(reference.per_class.policy.f1)}
                </>
              )}
              .<br />
              Saved to <code>{state.out_dir}</code>. The served model (<code>training/artifacts/</code>) is unchanged.
            </p>
          )}
          {state.error && <p className="state--error paste-error">{state.error}</p>}
        </div>
      )}
    </div>
  )
}

function RunStatus({ state }) {
  const c = state.config ?? {}
  const what = `lr ${c.lr} · p ${c.class_weight_power} · up to ${c.max_epochs} epochs`
  if (state.status === 'running') return <span className="live">training · {what}</span>
  if (state.status === 'error') return <span className="state--error">failed · {what}</span>
  return (
    <span className="stored">
      {state.status} · {what}
    </span>
  )
}
