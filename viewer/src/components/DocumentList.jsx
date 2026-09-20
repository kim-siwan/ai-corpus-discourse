// Submissions, not recommendations. A row identifies who wrote it and what the
// docket calls them — there is no title to show, because a submission does not
// have one. The 43% that carry a "RE:" line all say the same thing: the name
// of the RFI.
//
// The row reports paragraphs and words together because they do not stand in
// for one another: across the corpus they correlate at r = 0.74, so a high
// paragraph count often means the PDF broke its lines, not that the submission
// is long. Paragraphs say how much there is to label; words say how much there
// is to read.
export default function DocumentList({
  rows,
  selectedId,
  onSelect,
  tagCounts,
  paraCounts,
  wordCounts,
  showTopics,
}) {
  if (rows.length === 0) {
    return <p className="empty">No submissions match the current search and filters.</p>
  }

  return (
    <>
      {rows.map((doc) => {
        const tagged = tagCounts.get(doc.doc_id) ?? 0
        const paras = paraCounts.get(doc.doc_id) ?? 0
        const words = wordCounts.get(doc.doc_id) ?? 0
        return (
          <button
            key={doc.doc_id}
            type="button"
            className="row"
            aria-current={doc.doc_id === selectedId}
            onClick={() => onSelect(doc.doc_id)}
          >
            <div className="row-type">{doc.org_type_official}</div>
            <div className="row-org">{doc.org}</div>
            <div className="row-meta">
              {paras.toLocaleString()} paragraphs · {words.toLocaleString()} words
              {tagged > 0 && <span className="row-tagged"> · {tagged} tagged</span>}
            </div>
            {showTopics && (
              <div className="row-topics">
                {doc.topics.slice(0, 3).map(({ topic, n }) => (
                  <span key={topic} className="topic">
                    {topic} <i>{n}</i>
                  </span>
                ))}
                {doc.topics.length === 0 && (
                  <span className="topic topic--none">not indexed by IFP</span>
                )}
              </div>
            )}
          </button>
        )
      })}
    </>
  )
}
