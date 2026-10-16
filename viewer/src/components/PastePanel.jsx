import { useState } from 'react'
import SentenceText from './SentenceText.jsx'
import { LABELS, predictText, toUtf16 } from '../lib/predictions.js'

// Paste any text and run the trained network on it, live. Every press of
// Predict is a request to the backend's POST /predict — nothing here is
// precomputed. One section of the Model tab (ModelView).

export default function PastePanel({ status, text, onText, result, onResult }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const ready = status?.available

  async function run() {
    if (!text.trim() || !ready) return
    setBusy(true)
    setError(null)
    try {
      const res = await predictText(text)
      onResult({ ...res, text, sentences: toUtf16(text, res.sentences) })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="paste">
      <p className="paste-note">
        Each sentence is labelled by the network, called live through <code>POST /predict</code>. Policy
        is shaded; hover any sentence for its type.
      </p>
      <div>
        <textarea
          className="paste-input"
          value={text}
          onChange={(e) => onText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              run()
            }
          }}
          placeholder="Paste a paragraph from a submission, or any text"
          rows={6}
          aria-label="Text to classify"
        />
        <div className="paste-actions">
          <button type="button" className="btn btn--primary" onClick={run} disabled={!ready || busy || !text.trim()}>
            {busy ? 'Predicting…' : 'Predict'}
          </button>
          <span className="paste-hint">⌘/Ctrl + Enter</span>
          {!ready && (
            <span className="backend-off">
              Prediction is off: {status?.reason ?? 'checking…'}. Start the backend with{' '}
              <code>training/.venv/bin/uvicorn app:app --app-dir backend --port 8000</code>
            </span>
          )}
        </div>
        {error && <p className="state--error paste-error">{error}</p>}

        {result && (
          <div className="paste-result">
            <p className="paste-summary">
              <b>{result.sentences.length}</b> sentences · predicted types{' '}
              {result.labels.map((l) => (
                <span key={l} className={`pred-chip pred-chip--${l}`}>
                  {l}
                </span>
              ))}{' '}
              · {result.contains_demand ? 'contains a demand (≥ 1 policy sentence)' : 'no policy sentence'} ·{' '}
              <span className="live">live · {result.ms} ms</span>
            </p>
            <div className="para-body paste-text">
              <SentenceText body={result.text} sentences={result.sentences} />
            </div>
            <table className="probs">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Sentence</th>
                  <th>Predicted</th>
                  {LABELS.map((l) => (
                    <th key={l}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.sentences.map((s, i) => (
                  <tr key={s.start}>
                    <td className="num">{i + 1}</td>
                    <td className="probs-text">{s.text.replace(/\s+/g, ' ')}</td>
                    <td>
                      <span className={`pred-chip pred-chip--${s.label}`}>{s.label}</span>
                    </td>
                    {LABELS.map((l) => (
                      <td key={l} className="probs-cell" title={`${l} ${s.probs[l].toFixed(4)}`}>
                        <span className={`bar bar--${l}`} style={{ width: `${Math.round(s.probs[l] * 100)}%` }} />
                        <span className="num">{s.probs[l].toFixed(2)}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
