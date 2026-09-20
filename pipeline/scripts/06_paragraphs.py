"""Split each submission into paragraphs. These are the corpus's segments.

The unit of analysis has to be one this project owns. A sentence is too small
to carry a policy position and a submission is too large to label as one
thing; a paragraph is what an organization actually writes a recommendation
in. The alternative was to inherit IFP's extraction, which sets boundaries
with an unnamed LLM and, for 29% of its records, does not reproduce wording
found in the submission at all.

    python3 scripts/06_paragraphs.py     # run after 04

Reads  data/fulltext/*.txt
Writes data/segments.json

Paragraphs are separated by a blank line in the extracted text, which is what
pdftotext emits between them. Runs shorter than MIN_CHARS are dropped: page
numbers, letterhead fragments, stray headers.

Offsets are code-point offsets into the document's extracted text, so an
annotation on a paragraph can be located in the submission it came from.
Re-running extraction with a different pdftotext changes those offsets, which
is why the text and its hash are stored with them.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULLTEXT = ROOT / "data" / "fulltext"
OUT = ROOT / "data" / "segments.json"
LAYER = "fulltext-paragraph"
MIN_CHARS = 80


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paragraphs(text: str):
    """Yield (start, end, paragraph) with offsets into `text`."""
    for match in re.finditer(r"[^\n](?:.|\n(?!\n))*", text):
        body = match.group(0).strip()
        if len(body) < MIN_CHARS:
            continue
        # strip() moved the edges; find where the kept text actually sits
        start = match.start() + match.group(0).index(body[:40])
        yield start, start + len(body), body


def main() -> None:
    if not FULLTEXT.exists():
        sys.exit("run scripts/04_extract_fulltext.py first")

    documents = json.loads((ROOT / "data" / "documents.json").read_text())
    segments, empty = [], []

    for doc in documents:
        path = FULLTEXT / f"{doc['doc_id']}.txt"
        if not path.exists():
            empty.append(doc["doc_id"])
            continue
        text = path.read_text()
        found = 0
        for i, (start, end, body) in enumerate(paragraphs(text), start=1):
            segments.append({
                "segment_id": f"{doc['doc_id']}#p{i:04d}",
                "doc_id": doc["doc_id"],
                "layer": LAYER,
                "text": body,
                "text_sha256": sha256(body),
                "char_offset_in_doc": start,
                "char_end_in_doc": end,
                "para_index": i,
            })
            found += 1
        if not found:
            empty.append(doc["doc_id"])

    OUT.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")

    lengths = sorted(len(s["text"]) for s in segments)
    per_doc = {}
    for s in segments:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    counts = sorted(per_doc.values())

    print(f"{len(segments):,} paragraphs from {len(per_doc)} documents")
    print(f"  {len(empty)} documents yielded none (no extracted text)")
    print(f"\nParagraph length  median {lengths[len(lengths)//2]:,} "
          f"/ 95th {lengths[int(len(lengths)*0.95)]:,} / max {lengths[-1]:,} chars")
    print(f"Per document      median {counts[len(counts)//2]} "
          f"/ 95th {counts[int(len(counts)*0.95)]} / max {counts[-1]}")

    over = [x for x in lengths if x > 2000]
    print(f"\nOver 2,000 chars  {len(over):,} ({len(over)/len(lengths):.1%}), "
          f"holding {sum(over)/sum(lengths):.1%} of all text")
    print("  These need chunking before they are embedded; a 512-token encoder")
    print("  would silently truncate them.")

    assert all(s["text_sha256"] == sha256(s["text"]) for s in segments), "hash mismatch"
    assert len({s["segment_id"] for s in segments}) == len(segments), "duplicate segment_id"
    print("\nChecks passed.")


if __name__ == "__main__":
    main()
