// Anything other than an exact match is stated on the segment itself, so a
// reader never mistakes a passage for a quotation it could not be verified as.
const MATCH_NOTE = {
  partial: ['part of this text was found in the submission', 'Some lines appear in the source PDF and some do not.'],
  none: ['not found in the submission', 'The wording could not be located in the source PDF. That may mean the IFP database rewrote it, or that the PDF extracted badly.'],
  unavailable: ['source text unavailable', 'No text could be extracted from this submission, so the passage could not be checked.'],
}

import FullSubmission from './FullSubmission.jsx'
import { splitOnQuery } from '../lib/search.js'

function Highlighted({ text, query }) {
  return splitOnQuery(text, query).map((part, i) =>
    part.hit ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
  )
}

export default function SegmentDetail({
  segment,
  doc,
  siblings,
  query,
  reviewed,
  onToggleReviewed,
  onSelect,
  fullTextChars,
}) {
  if (!segment) {
    return <p className="empty">Select a segment to read it.</p>
  }

  return (
    <div className="detail">
      <h2>{segment.title}</h2>

      {MATCH_NOTE[segment.source_match] && (
        <p className={`match match--${segment.source_match}`} title={MATCH_NOTE[segment.source_match][1]}>
          {MATCH_NOTE[segment.source_match][0]}
        </p>
      )}

      <div className="dmeta">
        <span className="org">{doc?.org ?? 'Unknown organization'}</span>
        <span>
          {doc?.org_type_ifp ?? '—'}
          <span
            className="sup"
            title="LLM-derived field from the IFP database. Not used as an analysis variable."
          >
            †
          </span>
        </span>
        <span>{doc?.date ?? '—'}</span>
        {doc?.url && (
          <a href={doc.url} target="_blank" rel="noopener noreferrer">
            Source PDF ↗
          </a>
        )}
      </div>

      {/* Decision 4 puts the tag list here, above the text. Added 9/18. */}

      <div className="seg-text">
        <Highlighted text={segment.text} query={query} />
      </div>

      {doc?.doc_id && (
        <FullSubmission docId={doc.doc_id} pdfUrl={doc.url} chars={fullTextChars} />
      )}

      {siblings.length > 0 && (
        <div className="sibs">
          <h3>Other segments in this document · {siblings.length}</h3>
          {siblings.map((sib) => (
            <button
              key={sib.segment_id}
              type="button"
              className="sib"
              onClick={() => onSelect(sib.segment_id)}
            >
              <code>{sib.segment_id}</code>
              {sib.title}
            </button>
          ))}
        </div>
      )}

      <label className="revrow">
        <input type="checkbox" checked={reviewed} onChange={onToggleReviewed} />
        Reviewed <code>reviewed</code>
        <span style={{ color: 'var(--ink-3)' }}>· not saved yet (backend lands 9/19)</span>
      </label>
    </div>
  )
}
