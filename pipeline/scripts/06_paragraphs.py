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

A paragraph is stored as a pair of offsets into its document's extracted
text, not as a copy of it. The text is already on disk in data/fulltext, and
duplicating 14 MB of it would put the same words in two places that could
drift apart. Readers slice: text[char_offset_in_doc:char_end_in_doc].

Offsets are code-point offsets into that file. Re-running extraction with a
different pdftotext moves them, so index.json records a SHA-256 of each
document's text; a mismatch means every offset for that document is stale.
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
# Some submissions arrive with no blank lines at all, so splitting on those
# alone hands back one 2,892-word "paragraph" that cannot be labelled as a
# single thing. Blocks longer than this are cut again at sentence ends.
MAX_CHARS = 1500
TARGET_CHARS = 700

# Python requires a fixed-width look-behind, so the optional closing quote or
# bracket is spelled as a second alternative rather than with `?`.
_AFTER = r"(?=[A-Z0-9\u2022\u25cf\-])"
SENTENCE_END = re.compile(
    r"(?<=[.!?][\"'\)\]])\s+" + _AFTER + r"|(?<=[.!?])\s+" + _AFTER
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_long(start: int, body: str):
    """Cut an oversized block at sentence ends, keeping offsets exact."""
    cuts = [0] + [m.end() for m in SENTENCE_END.finditer(body)] + [len(body)]
    piece_start = 0
    for i, cut in enumerate(cuts[1:], start=1):
        if cut - piece_start < TARGET_CHARS and i < len(cuts) - 1:
            continue
        piece = body[piece_start:cut]
        stripped = piece.strip()
        if len(stripped) >= MIN_CHARS:
            lead = len(piece) - len(piece.lstrip())
            yield start + piece_start + lead, start + piece_start + lead + len(stripped), stripped
        piece_start = cut


def paragraphs(text: str):
    """Yield (start, end, paragraph) with offsets into `text`."""
    for match in re.finditer(r"[^\n](?:.|\n(?!\n))*", text):
        body = match.group(0).strip()
        if len(body) < MIN_CHARS:
            continue
        # strip() moved the edges; find where the kept text actually sits
        start = match.start() + match.group(0).index(body[:40])
        if len(body) <= MAX_CHARS:
            yield start, start + len(body), body
        else:
            yield from split_long(start, body)


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
                "char_offset_in_doc": start,
                "char_end_in_doc": end,
                "n_chars": len(body),
                "n_words": len(body.split()),
                "para_index": i,
            })
            found += 1
        if not found:
            empty.append(doc["doc_id"])

    OUT.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")

    # Record what the offsets were measured against.
    index_path = FULLTEXT / "index.json"
    index = json.loads(index_path.read_text())
    for doc_id, entry in index["documents"].items():
        path = FULLTEXT / f"{doc_id}.txt"
        if path.exists():
            entry["sha256"] = sha256(path.read_text())
    index["paragraphs"] = {
        "layer": LAYER,
        "min_chars": MIN_CHARS,
        "note": "segments.json holds offsets into these files, not copies of the text",
    }
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")

    lengths = sorted(s["n_chars"] for s in segments)
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

    assert len({s["segment_id"] for s in segments}) == len(segments), "duplicate segment_id"
    # Every offset pair must still cut the paragraph it was made from.
    checked = 0
    for doc_id in list(per_doc)[:40]:
        text = (FULLTEXT / f"{doc_id}.txt").read_text()
        for s in (x for x in segments if x["doc_id"] == doc_id):
            cut = text[s["char_offset_in_doc"]:s["char_end_in_doc"]]
            assert len(cut) == s["n_chars"] and cut == cut.strip(), f"bad offsets {s['segment_id']}"
            checked += 1
    print(f"Verified {checked:,} offset pairs against the source text.")
    print("\nChecks passed.")


if __name__ == "__main__":
    main()
