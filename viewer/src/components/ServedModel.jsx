import { useEffect, useState } from 'react'
import LossChart from './LossChart.jsx'
import { points, served } from '../lib/training.js'

// Training monitor for the model the app serves: how its loss and error
// moved per epoch, and how it scored on train, dev and the set-aside test
// split. Everything comes from training/artifacts/ through GET /train/served.

// From the Day 1 comparison (training/results/day1_minilm.json): the shallow
// benchmark the network has to beat on CDCP test error.
const BENCHMARK = { name: 'TF-IDF + LogReg (shallow benchmark)', error: 0.327, macro_f1: 0.615, policy_f1: 0.71 }

const f3 = (x) => (x == null ? '—' : x.toFixed(3))
const f2 = (x) => (x == null ? '—' : x.toFixed(2))

export default function ServedModel({ status }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (status === null) return
    let cancelled = false
    served()
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [status])

  if (error) return <p className="backend-off">Training record unavailable: {error}</p>
  if (!data) return <p className="backend-off">Loading the training record…</p>

  const { config: c, history, metrics: m } = data
  const seeds = m.final_setting_3_seeds
  const ci = m.test.policy_f1_ci
  const cm = m.test.confusion_matrix
  const rows = [
    ['train', m.train],
    ['dev', m.dev],
    ['test', m.test],
  ]

  return (
    <div className="served">
      <p className="served-config">
        MiniLM (frozen) → Linear {c.hidden.join('→')} → ReLU → Dropout {c.dropout} → Linear 4 ·{' '}
        {c.n_parameters.toLocaleString()} weights · plain SGD, lr {c.lr} · class weight p {c.class_weight_power} ·
        batch {c.batch_size} · seed {c.seed} · best epoch <b>{m.best_epoch}</b> (
        {m.stop_reason === 'early_stop' ? `early-stopped at ${m.last_epoch}` : m.stop_reason})
      </p>

      <div className="served-charts">
        <LossChart
          title={c.class_weight_power ? 'Weighted cross-entropy loss' : 'Cross-entropy loss'}
          series={[
            { key: 'train', label: 'train', points: points(history, 'train_loss') },
            { key: 'dev', label: 'dev', points: points(history, 'dev_loss') },
          ]}
          best={m.best_epoch}
        />
        <LossChart
          title="Error (1 − accuracy)"
          series={[
            { key: 'train', label: 'train', points: points(history, 'train_error') },
            { key: 'dev', label: 'dev', points: points(history, 'dev_error') },
          ]}
          best={m.best_epoch}
        />
      </div>
      <p className="served-read">
        Dev loss stops falling around epoch 20 while train loss keeps going down: the network starts fitting the
        training set rather than the pattern. Early stopping keeps the weights from the epoch with the lowest dev
        loss.
      </p>

      <table className="results">
        <thead>
          <tr>
            <th>CDCP split</th>
            <th>sentences</th>
            <th>error</th>
            <th>macro-F1</th>
            <th>policy P</th>
            <th>policy R</th>
            <th>policy F1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name} className={name === 'test' ? 'results-test' : undefined}>
              <td>
                {name}
                {name === 'test' && <span className="results-note"> scored once</span>}
              </td>
              <td className="num">
                {Object.values(r.per_class)
                  .reduce((n, x) => n + x.support, 0)
                  .toLocaleString()}
              </td>
              <td className="num">{f3(r.error)}</td>
              <td className="num">{f3(r.macro_f1)}</td>
              <td className="num">{f2(r.per_class.policy.precision)}</td>
              <td className="num">{f2(r.per_class.policy.recall)}</td>
              <td className="num">
                {f2(r.per_class.policy.f1)}
                {name === 'test' && (
                  <span className="results-note">
                    {' '}
                    [{f2(ci.ci95[0])}, {f2(ci.ci95[1])}]
                  </span>
                )}
              </td>
            </tr>
          ))}
          <tr className="results-bench">
            <td>{BENCHMARK.name}, test</td>
            <td className="num">836</td>
            <td className="num">{f3(BENCHMARK.error)}</td>
            <td className="num">{f3(BENCHMARK.macro_f1)}</td>
            <td className="num">—</td>
            <td className="num">—</td>
            <td className="num">{f2(BENCHMARK.policy_f1)}</td>
          </tr>
        </tbody>
      </table>
      <p className="served-read">
        Test error over three seeds: <b>{f3(seeds.test_error_mean)}</b> ± {f3(seeds.test_error_sd)} (range{' '}
        {f3(seeds.test_error_range[0])}–{f3(seeds.test_error_range[1])}), below the benchmark's {f3(BENCHMARK.error)}.
        The policy interval is a bootstrap over {ci.n_units} comments.
      </p>

      <details className="served-cm">
        <summary>Test confusion matrix — where the errors go</summary>
        <table className="results">
          <thead>
            <tr>
              <th>true ↓ / predicted →</th>
              {cm.labels.map((l) => (
                <th key={l}>{l}</th>
              ))}
              <th>recall</th>
            </tr>
          </thead>
          <tbody>
            {cm.rows_true_cols_pred.map((row, i) => (
              <tr key={cm.labels[i]}>
                <td>{cm.labels[i]}</td>
                {row.map((n, j) => (
                  <td key={j} className={`num${i === j ? ' cm-diag' : ''}`}>
                    {n}
                  </td>
                ))}
                <td className="num">{f2(row[i] / row.reduce((a, b) => a + b, 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="served-read">
          Most errors land in <code>value</code>: missed policy and fact sentences are read as evaluations.
        </p>
      </details>
    </div>
  )
}
