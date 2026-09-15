"""Check each segment against the submission it came from.

The IFP `Full text` field is not consistently a quotation. Measuring 788
segments against their source PDFs, 62% appeared verbatim, 32% shared almost
no wording with the submission, and the rest were partial. Nothing in the
record distinguishes the two, so this step adds the distinction.

    python3 scripts/05_match_fulltext.py        # run after 02 and 04

Adds to each segment:

    source_match         exact | partial | none | unavailable
    char_offset_in_doc   set when source_match is exact

`none` means NOT FOUND, which is not the same as "invented": a verbatim
quotation can go undetected when the PDF extracts badly (broken ligatures,
two-column layout, tables). Treat it as unverified, not as fabricated.

Comparison ignores case, whitespace and punctuation, so it is insensitive to
how pdftotext lays out a line but still requires the same words in the same
order.
"""

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FULLTEXT = DATA / "fulltext"
MIN_LINE = 25  # a line shorter than this carries too little signal to match on


def reduce_text(text: str) -> tuple[str, list[int]]:
    """Keep only alphanumerics, and map each kept character back to its index."""
    folded = unicodedata.normalize("NFKC", text).lower()
    kept, index = [], []
    for i, ch in enumerate(folded):
        if ch.isalnum():
            kept.append(ch)
            index.append(i)
    return "".join(kept), index


def classify(segment_text: str, haystack: str) -> tuple[str, int | None]:
    reduced, _ = reduce_text(segment_text)
    if not reduced:
        return "none", None

    at = haystack.find(reduced)
    if at >= 0:
        return "exact", at

    lines = [ln for ln in segment_text.split("\n") if len(reduce_text(ln)[0]) > MIN_LINE]
    if not lines:
        return "none", None
    hits = sum(1 for ln in lines if reduce_text(ln)[0] in haystack)
    return ("partial", None) if hits / len(lines) >= 0.3 else ("none", None)


def main() -> None:
    segments = json.loads((DATA / "segments.json").read_text())
    by_doc: dict[str, list[dict]] = {}
    for seg in segments:
        by_doc.setdefault(seg["doc_id"], []).append(seg)

    counts = {"exact": 0, "partial": 0, "none": 0, "unavailable": 0}

    for doc_id, segs in by_doc.items():
        path = FULLTEXT / f"{doc_id}.txt"
        if not path.exists():
            for seg in segs:
                seg["source_match"] = "unavailable"
                counts["unavailable"] += 1
            continue

        full = path.read_text()
        haystack, index = reduce_text(full)
        for seg in segs:
            verdict, at = classify(seg["text"], haystack)
            seg["source_match"] = verdict
            seg["char_offset_in_doc"] = index[at] if at is not None else None
            counts[verdict] += 1

    (DATA / "segments.json").write_text(
        json.dumps(segments, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    total = len(segments)
    print(f"{total:,} segments")
    for key in ("exact", "partial", "none", "unavailable"):
        print(f"  {counts[key]:5,}  {counts[key] / total:5.1%}  {key}")
    offsets = sum(1 for s in segments if s.get("char_offset_in_doc") is not None)
    print(f"\nchar_offset_in_doc filled for {offsets:,} segments")


if __name__ == "__main__":
    main()
