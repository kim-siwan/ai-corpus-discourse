// A submission's text comes out of pdftotext with the PDF's own line breaks in
// it, so a sentence arrives split across four lines at whatever width the
// original page happened to be. Rendering those breaks makes the text ragged
// and hard to read next to a list of tags.
//
// The fix is a CSS one: .para-body collapses whitespace, so the browser
// reflows the text to the panel. Nothing in the DOM text node changes, which
// matters because a span annotation's offsets are measured against it.
//
// Two consequences are handled here.

const SPECIAL = /[.*+?^${}()|[\]\\]/g

/** Locate `quote`, as the browser rendered it, in the source `text`.
 *
 *  A selection that crosses a line break comes back with a space where the
 *  source has a newline, so an exact indexOf misses. Matching every run of
 *  whitespace against every other run finds it and returns the offsets the
 *  annotation should carry — offsets into the source, not into the rendering.
 */
export function locate(text, quote) {
  const words = quote.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return null
  const pattern = words.map((w) => w.replace(SPECIAL, '\\$&')).join('\\s+')
  const match = new RegExp(pattern).exec(text)
  return match ? { at: match.index, matched: match[0] } : null
}

/** Split a paragraph into runs, marking the hyphens that only exist because a
 *  word was broken across two lines. The caller hides those runs; they stay in
 *  the DOM text so the offsets above keep counting them.
 */
export function runs(body) {
  const out = []
  let last = 0
  for (const m of body.matchAll(/(?<=\p{Ll})-\n(?=\p{Ll})/gu)) {
    if (m.index > last) out.push({ text: body.slice(last, m.index) })
    out.push({ text: m[0], hidden: true })
    last = m.index + m[0].length
  }
  out.push({ text: body.slice(last) })
  return out
}
