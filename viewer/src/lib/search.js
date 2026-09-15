// Search over segments. Decision 5: the query is matched against the title,
// the organization name and the segment text. `title` is an LLM-written
// summary from the IFP database, so the words a researcher looks for
// (e.g. "apprenticeship") are often only in the text.

/** Split `text` into alternating plain/hit parts for rendering. */
export function splitOnQuery(text, query) {
  if (!query) return [{ text, hit: false }]
  const parts = []
  const haystack = text.toLowerCase()
  const needle = query.toLowerCase()
  let from = 0
  for (;;) {
    const at = haystack.indexOf(needle, from)
    if (at === -1) break
    if (at > from) parts.push({ text: text.slice(from, at), hit: false })
    parts.push({ text: text.slice(at, at + needle.length), hit: true })
    from = at + needle.length
  }
  if (from < text.length) parts.push({ text: text.slice(from), hit: false })
  return parts.length ? parts : [{ text, hit: false }]
}

/** Text around the first match, for a list row. Null when there is no match. */
export function excerpt(text, query, before = 40, after = 60) {
  if (!query) return null
  const at = text.toLowerCase().indexOf(query.toLowerCase())
  if (at === -1) return null
  const start = Math.max(0, at - before)
  const end = Math.min(text.length, at + query.length + after)
  return {
    lead: (start > 0 ? '…' : '') + text.slice(start, at),
    hit: text.slice(at, at + query.length),
    tail: text.slice(at + query.length, end) + (end < text.length ? '…' : ''),
  }
}
