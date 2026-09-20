import { useRef, useState } from 'react'
import FullSubmission from './FullSubmission.jsx'
import TagInput from './TagInput.jsx'
import { splitOnQuery } from '../lib/search.js'
import { toCodePointOffset } from '../lib/annotations.js'

// Anything other than an exact match is stated on the segment itself, so a
// reader never mistakes a passage for a quotation it could not be verified as.
const MATCH_NOTE = {
  partial: [
    'part of this text was found in the submission',
    'Some lines appear in the source PDF and some do not.',
  ],
  none: [
    'not found in the submission',
    'The wording could not be located in the source PDF. That may mean the IFP database rewrote it, or that the PDF extracted badly.',
  ],
  unavailable: [
    'source text unavailable',
    'No text could be extracted from this submission, so the passage could not be checked.',
  ],
}

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
  annotations,
  registry,
  onTagSegment,
  onSelectSpan,
  onRemoveTag,
  tagFocusKey,
}) {
  const textRef = useRef(null)
  // One submission holds 56 segments. Listing every sibling pushes the tag
  // field and the reviewed toggle far down the panel, which is the opposite of
  // what a keyboard pass needs.
  const [allSiblings, setAllSiblings] = useState(false)

  if (!segment) {
    return <p className="empty">Select a segment to read it. Use J and K to move through the list.</p>
  }

  // A selection inside the passage becomes a span annotation. The offsets are
  // measured against segment.text, not the rendered DOM, so highlighting or
  // re-rendering cannot shift them.
  function onMouseUp() {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !textRef.current?.contains(sel.anchorNode)) return
    const quote = sel.toString().trim()
    if (quote.length < 3) return

    const utf16 = segment.text.indexOf(quote)
    if (utf16 < 0) return // a selection spanning the highlight marks; ignore it
    const rect = sel.getRangeAt(0).getBoundingClientRect()
    onSelectSpan({
      quote,
      start: toCodePointOffset(segment.text, utf16),
      end: toCodePointOffset(segment.text, utf16 + quote.length),
      at: {
        left: Math.min(rect.left, window.innerWidth - 270),
        top: rect.bottom + 6,
      },
    })
  }

  const note = MATCH_NOTE[segment.source_match]

  return (
    <div className="detail">
      <h2>{segment.title}</h2>

      <div className="dmeta">
        <span className="org">{doc?.org ?? 'Unknown organization'}</span>
        <span>
          {doc?.org_type_ifp ?? '—'}
          <span className="sup" title="LLM-derived field from the IFP database. Not used as an analysis variable.">
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

      {note && (
        <p className={`match match--${segment.source_match}`} title={note[1]}>
          {note[0]}
        </p>
      )}

      {/* Decision 4: tags sit above the passage, so what is already applied
          stays visible without scrolling during a labelling pass. */}
      <div className="tagbox">
        <h3>Tags · {annotations.length}</h3>
        {annotations.length === 0 ? (
          <p className="none">No tags yet. Type below, or select a phrase in the passage.</p>
        ) : (
          annotations.map((a) => (
            <span className="tag" key={a.ann_id}>
              <b>{a.tag}</b>
              {a.quote && <q>{a.quote}</q>}
              <button type="button" className="tag-x" onClick={() => onRemoveTag(a.ann_id)} aria-label={`Remove ${a.tag}`}>
                ×
              </button>
            </span>
          ))
        )}
      </div>

      <div className="seg-text" ref={textRef} onMouseUp={onMouseUp}>
        <Highlighted text={segment.text} query={query} />
      </div>

      {doc?.doc_id && <FullSubmission docId={doc.doc_id} pdfUrl={doc.url} chars={fullTextChars} />}

      {siblings.length > 0 && (
        <div className="sibs">
          <h3>Other segments in this document · {siblings.length}</h3>
          {(allSiblings ? siblings : siblings.slice(0, 6)).map((sib) => (
            <button key={sib.segment_id} type="button" className="sib" onClick={() => onSelect(sib.segment_id)}>
              <code>{sib.segment_id}</code>
              {sib.title}
            </button>
          ))}
          {siblings.length > 6 && (
            <button type="button" className="readmore" onClick={() => setAllSiblings((v) => !v)}>
              {allSiblings ? 'Show fewer' : `Show all ${siblings.length}`}
            </button>
          )}
        </div>
      )}

      <label className="revrow">
        <input type="checkbox" checked={reviewed} onChange={onToggleReviewed} />
        Reviewed <code>reviewed</code>
      </label>

      {/* Decision 6: the input stays at the foot of the panel, in view however
          far the passage is scrolled. */}
      <div className="tagbar">
        <TagInput
          key={tagFocusKey}
          registry={registry}
          onSave={onTagSegment}
          autoFocus
          hint="Enter saves and moves to the next segment · J and K move without tagging"
        />
      </div>
    </div>
  )
}
