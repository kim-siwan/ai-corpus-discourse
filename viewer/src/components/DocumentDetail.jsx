import { useEffect, useMemo, useRef } from 'react'
import TagInput from './TagInput.jsx'
import SentenceText from './SentenceText.jsx'
import { runs } from '../lib/reflow.js'
import { toUtf16, UNVALIDATED } from '../lib/predictions.js'

// The submission itself, paragraph by paragraph. Each paragraph is the unit a
// tag attaches to: a sentence is too small to carry a policy position and a
// whole submission is too large to label as one thing.
//
// Who wrote it stays at the top and the tag box stays at the foot; only the
// submission scrolls between them, so the coder never loses the box they are
// typing into or the name of what they are reading.
export default function DocumentDetail({
  doc,
  paragraphs,
  text,
  loading,
  error,
  currentId,
  onSelectParagraph,
  annotationsBySegment,
  registry,
  onTag,
  onRemoveTag,
  onSelectSpan,
  tagFocusKey,
  showPreds,
  onToggleShow,
  docPreds,
}) {
  const currentRef = useRef(null)

  // Predicted sentences per paragraph, with offsets converted to UTF-16
  // against the exact paragraph text rendered below.
  const predBySegment = useMemo(() => {
    const map = new Map()
    if (!docPreds?.segments || text === null) return map
    for (const p of paragraphs) {
      const seg = docPreds.segments[p.segment_id]
      if (!seg) continue
      const body = text.slice(p.char_offset_in_doc, p.char_end_in_doc)
      map.set(p.segment_id, toUtf16(body, seg.sentences))
    }
    return map
  }, [docPreds, text, paragraphs])

  // Keep the paragraph being tagged in view as J and K move through them.
  useEffect(() => {
    currentRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [currentId])

  if (!doc) return <p className="empty">Select a submission to read it.</p>

  return (
    <div className="detail">
      <div className="dhead">
        <div className="dtype">{doc.org_type_official}</div>
        <div className="dmeta">
          <span className="org">{doc.org}</span>
          <a className="pdf" href={doc.url} target="_blank" rel="noopener noreferrer">
            Source PDF ↗
          </a>
        </div>
        {doc.topics.length > 0 && (
          <div className="dtopics">
            {doc.topics.map(({ topic }) => (
              <span
                key={topic}
                className="topic"
                title={`IFP tagged something in this submission "${topic}". Which paragraph is not recorded: IFP published its own segmentation, not offsets into the PDF.`}
              >
                {topic}
              </span>
            ))}
          </div>
        )}
        <PredictHeader show={showPreds} onToggle={onToggleShow} preds={docPreds} />
      </div>

      <div className="dbody">
        {error && <p className="readmore-none">{error}</p>}
        {loading && <p className="readmore-none">Loading the submission…</p>}

        {text !== null && (
          <div className="paras">
            {paragraphs.map((p) => {
              const tags = annotationsBySegment.get(p.segment_id) ?? []
              const isCurrent = p.segment_id === currentId
              const body = text.slice(p.char_offset_in_doc, p.char_end_in_doc)
              const sentences = predBySegment.get(p.segment_id)
              const nPolicy = sentences?.filter((s) => s.label === 'policy').length ?? 0
              return (
                <div
                  key={p.segment_id}
                  ref={isCurrent ? currentRef : null}
                  className={`para${isCurrent ? ' para--current' : ''}${tags.length ? ' para--tagged' : ''}`}
                  onMouseDown={() => onSelectParagraph(p.segment_id)}
                >
                  <span className="para-no">
                    {p.para_index}
                    {tags.length > 0 && <i className="para-dot" aria-label="tagged" />}
                  </span>
                  <div className="para-body" onMouseUp={(e) => onSelectSpan(p, e.currentTarget)}>
                    {sentences ? (
                      <SentenceText body={body} sentences={sentences} />
                    ) : (
                      runs(body).map((run, i) =>
                        run.hidden ? (
                          <span className="softbreak" key={i}>
                            {run.text}
                          </span>
                        ) : (
                          run.text
                        ),
                      )
                    )}
                  </div>
                  {nPolicy > 0 && (
                    <p className="para-pred">
                      {nPolicy} of {sentences.length} sentence{sentences.length === 1 ? '' : 's'} predicted policy
                    </p>
                  )}
                  {tags.length > 0 && (
                    <div className="para-tags">
                      {tags.map((a) => (
                        <span className="chip chip--tag" key={a.ann_id}>
                          {a.tag}
                          {a.quote && <q>{a.quote.slice(0, 40)}</q>}
                          <button type="button" onClick={() => onRemoveTag(a.ann_id)} aria-label={`Remove ${a.tag}`}>
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="tagbar">
        <TagInput
          key={tagFocusKey}
          registry={registry}
          onSave={onTag}
          autoFocus
          hint={
            currentId
              ? `Tagging paragraph ${paragraphs.find((p) => p.segment_id === currentId)?.para_index ?? ''} · Enter saves it and moves to the next`
              : 'Select a paragraph to tag it'
          }
        />
      </div>
    </div>
  )
}

// Above the submission: whether predictions are shown, what the shading
// means, and where the predictions came from — the network called live, or
// the stored copy when the backend is off.
function PredictHeader({ show, onToggle, preds }) {
  let source = null
  if (show && preds?.status === 'loading') {
    source = <span className="pred-source">Predicting…</span>
  } else if (show && preds?.status === 'error') {
    source = <span className="pred-source state--error">No predictions: {preds.error}</span>
  } else if (show && preds?.status === 'ready') {
    source =
      preds.source === 'live' ? (
        <span className="pred-source">
          <span className="live">live</span> · {preds.n_sentences.toLocaleString()} sentences ·{' '}
          {(preds.ms / 1000).toFixed(2)} s · not validated on these submissions
        </span>
      ) : (
        <span className="pred-source" title={preds.reason}>
          <span className="stored">stored predictions</span> — backend off · not validated on these submissions
        </span>
      )
  }
  return (
    <div className="predhead">
      <label className="predtoggle">
        <input type="checkbox" checked={show} onChange={(e) => onToggle(e.target.checked)} />
        Show predicted types
      </label>
      {show && (
        <span className="pred-legend" title={UNVALIDATED}>
          <span className="sent sent--policy">policy</span> shaded · hover any sentence for its type
        </span>
      )}
      {source}
    </div>
  )
}
