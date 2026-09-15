import { excerpt, splitOnQuery } from '../lib/search.js'

function Highlighted({ text, query }) {
  return splitOnQuery(text, query).map((part, i) =>
    part.hit ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
  )
}

export default function SegmentList({ rows, selectedId, onSelect, query }) {
  if (rows.length === 0) {
    return <p className="empty">No segments match the current search and filters.</p>
  }

  return (
    <>
      {rows.map(({ segment, doc }) => {
        // Only show a body excerpt when the match is not already visible in
        // the title, so rows stay scannable (decision 2).
        const inTitle = query && segment.title.toLowerCase().includes(query.toLowerCase())
        const snippet = inTitle ? null : excerpt(segment.text, query)

        return (
          <button
            key={segment.segment_id}
            type="button"
            className="row"
            aria-current={segment.segment_id === selectedId}
            onClick={() => onSelect(segment.segment_id)}
          >
            <div className="row-id">{segment.segment_id}</div>
            <div className="row-org">
              {doc?.org ?? 'Unknown organization'} · {doc?.org_type_ifp ?? '—'}
            </div>
            <div className="row-title">
              <Highlighted text={segment.title} query={query} />
            </div>
            {snippet && (
              <div className="row-snippet">
                {snippet.lead}
                <mark>{snippet.hit}</mark>
                {snippet.tail}
              </div>
            )}
          </button>
        )
      })}
    </>
  )
}
