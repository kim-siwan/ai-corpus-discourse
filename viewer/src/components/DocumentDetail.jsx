import { useEffect, useRef } from 'react'
import TagInput from './TagInput.jsx'

// The submission itself, paragraph by paragraph. Each paragraph is the unit a
// tag attaches to: a sentence is too small to carry a policy position and a
// whole submission is too large to label as one thing.
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
  showTopics,
}) {
  const currentRef = useRef(null)

  // Keep the paragraph being tagged in view as J and K move through them.
  useEffect(() => {
    currentRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [currentId])

  if (!doc) return <p className="empty">Select a submission. J and K move between paragraphs.</p>

  return (
    <div className="detail">
      <div className="dmeta">
        <span className="org">{doc.org}</span>
        <span className="badge">{doc.org_type_official}</span>
        <span>{doc.date}</span>
        <a className="pdf" href={doc.url} target="_blank" rel="noopener noreferrer">
          Source PDF ↗
        </a>
      </div>

      {showTopics && doc.topics.length > 0 && (
        <div className="dtopics">
          {doc.topics.map(({ topic, n }) => (
            <span
              key={topic}
              className="topic"
              title={`IFP tagged ${n} recommendation${n === 1 ? '' : 's'} in this submission with "${topic}". A count reflects how finely IFP split the submission, not how much it stressed the topic.`}
            >
              {topic} <i>{n}</i>
            </span>
          ))}
        </div>
      )}

      {error && <p className="readmore-none">{error}</p>}
      {loading && <p className="readmore-none">Loading the submission…</p>}

      {text !== null && (
        <div className="paras">
          {paragraphs.map((p) => {
            const tags = annotationsBySegment.get(p.segment_id) ?? []
            const isCurrent = p.segment_id === currentId
            return (
              <div
                key={p.segment_id}
                ref={isCurrent ? currentRef : null}
                className={`para${isCurrent ? ' para--current' : ''}${tags.length ? ' para--tagged' : ''}`}
                onMouseDown={() => onSelectParagraph(p.segment_id)}
              >
                <span className="para-no">{p.para_index}</span>
                <div
                  className="para-body"
                  onMouseUp={(e) => onSelectSpan(p, e.currentTarget)}
                >
                  {text.slice(p.char_offset_in_doc, p.char_end_in_doc)}
                </div>
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

      <div className="tagbar">
        <TagInput
          key={tagFocusKey}
          registry={registry}
          onSave={onTag}
          autoFocus
          hint={
            currentId
              ? `Tagging paragraph ${paragraphs.find((p) => p.segment_id === currentId)?.para_index ?? ''} · Enter saves and moves to the next · J and K move without tagging`
              : 'Select a paragraph to tag it'
          }
        />
      </div>
    </div>
  )
}
