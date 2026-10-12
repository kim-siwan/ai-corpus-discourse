import { describe, pieces, UNVALIDATED } from '../lib/predictions.js'

// A paragraph with each sentence marked by its predicted type. Hovering a
// sentence shows the four probabilities.
//
// Only spans are added around the text, never characters, so the element's
// textContent is the paragraph exactly as before — the span-tagging code
// measures its offsets against that. `sentences` must already be in UTF-16
// offsets (predictions.toUtf16).
export default function SentenceText({ body, sentences }) {
  return pieces(body, sentences ?? []).map((piece, i) => {
    if (piece.hidden) {
      return (
        <span className="softbreak" key={i}>
          {piece.text}
        </span>
      )
    }
    const s = piece.sentence
    if (!s) return piece.text
    return (
      <span
        key={i}
        className={`sent sent--${s.label}`}
        title={`Predicted ${s.label} — ${describe(s.probs)}\n${UNVALIDATED}`}
      >
        {piece.text}
      </span>
    )
  })
}
