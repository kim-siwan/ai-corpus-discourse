"""Adapt the IFP snapshot into the shared corpus schema.

Adapter contract: adapter(raw) -> {documents, segments}
Downstream code (viewer, annotation UI, embeddings, models) reads only the
shared schema and does not know the source.

    python scripts/02_adapter_ifp.py              # Talent & Education only
    python scripts/02_adapter_ifp.py --topic ""   # all records

Output: data/corpus.json, data/documents.json, data/segments.json
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

CORPUS_ID = "ostp-ai-action-plan-rfi-2025"
LAYER = "ifp-recommendation"
NORMALIZATION = "nfkc-v1"
DEFAULT_TOPIC = "Talent & Education"
RFI_CLOSING_DATE = "2025-03-15"


def normalize(text: str) -> str:
    """Apply nfkc-v1. Annotation offsets are measured against this output.

    Changing this function invalidates every existing offset. If it must
    change, give it a new name (nfkc-v2) and write a remapping step.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)  # folds PDF ligatures: ﬁ -> fi
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id_from_url(url: str) -> str:
    """Derive doc_id from the NITRD filename, which is more stable than org name."""
    stem = Path(unquote(urlparse(url).path)).stem if url else "unknown"
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "unknown"
    return f"nitrd-90fr9088-{slug}"


def first(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def split_topics(value) -> list:
    return [t.strip() for t in str(value or "").split(",") if t.strip()]


def load_records(snapshot: Path) -> list:
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("records", "data", "results", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    raise ValueError("No record array found in snapshot.")


def adapt(records: list, topic_filter: str) -> dict:
    # Sort by source id so ids stay stable across reruns, then number the whole
    # set BEFORE filtering. Numbering the filtered set instead would make
    # segment_id depend on --topic: ostp25-0004 would name a different
    # recommendation in every subset, and annotations would not survive a
    # change of scope.
    records = sorted(records, key=lambda r: str(r.get("Record ID") or r.get("id") or ""))
    numbered = list(enumerate(records, start=1))

    if topic_filter:
        numbered = [(i, r) for i, r in numbered if topic_filter in str(r.get("Topic") or "")]

    documents, segments, skipped = {}, [], 0

    for i, r in numbered:
        text = normalize(first(r.get("Full text")))
        if not text:
            skipped += 1
            continue

        url = first(r.get("URL"))
        doc_id = doc_id_from_url(url)

        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": doc_id,
                "corpus": CORPUS_ID,
                "org": first(r.get("Organization")),
                "org_type_official": None,   # fill from the NITRD metadata table
                "org_type_ifp": first(r.get("Org Type")) or None,  # LLM label, unvalidated
                "date": RFI_CLOSING_DATE,
                "url": url,
                "retrieved": None,
                "reviewed": False,
            }

        segments.append({
            "segment_id": f"ostp25-{i:04d}",
            "doc_id": doc_id,
            "layer": LAYER,
            "title": normalize(first(r.get("Recommendation"))) or None,
            "text": text,
            "text_sha256": sha256(text),
            "char_offset_in_doc": None,  # fill when NITRD full text is added
            "topics": split_topics(r.get("Topic")),
            "source_ids": {"ifp_record_id": first(r.get("Record ID") or r.get("id")) or None},
        })

    if skipped:
        print(f"  Skipped {skipped} records with empty text")

    return {"documents": list(documents.values()), "segments": segments}


def build_corpus_meta(retrieved: str, topic_filter: str) -> dict:
    return {
        "corpus_id": CORPUS_ID,
        "source": "NITRD 90 FR 9088",
        "docket_url": "https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/",
        "normalization": NORMALIZATION,
        "retrieved": retrieved,
        "topic_filter": topic_filter or None,
        "layers": {
            LAYER: {
                "extractor": "IFP AI Action Plan Database",
                "extractor_model": "claude-3-7-sonnet-latest",
                "extractor_url": "https://www.aiactionplan.org",
                "extractor_methodology": "https://www.aiactionplan.org/methodology",
                "extracted_field": "Full text",
                "note": (
                    "Text and derived labels in this layer were extracted by IFP using an "
                    "LLM. Use Topic only as a sampling filter; validate Org Type before "
                    "using it as an analytic variable."
                ),
            }
        },
    }


def report(documents: list, segments: list) -> None:
    lengths = sorted(len(s["text"]) for s in segments)
    types = {}
    for d in documents:
        types[d["org_type_ifp"]] = types.get(d["org_type_ifp"], 0) + 1

    print(f"\n{len(documents):,} documents / {len(segments):,} segments")
    if lengths:
        print(f"Segment length: median {lengths[len(lengths)//2]:,} "
              f"/ min {lengths[0]:,} / max {lengths[-1]:,} chars")
    print("Documents by org type (IFP label, unvalidated):")
    for t, n in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {t}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--topic", default=DEFAULT_TOPIC, help='substring filter; "" for all')
    args = ap.parse_args()

    snapshot = args.snapshot
    if snapshot is None:
        candidates = sorted(RAW_DIR.glob("ifp_airtable_*.json"))
        if not candidates:
            print("No snapshot found. Run 01_snapshot_ifp.py first.", file=sys.stderr)
            return 1
        snapshot = candidates[-1]

    match = re.search(r"(\d{4}-\d{2}-\d{2})", snapshot.name)
    retrieved = match.group(1) if match else None

    print(f"Snapshot: {snapshot.name}")
    records = load_records(snapshot)
    print(f"  {len(records):,} records total")

    result = adapt(records, args.topic)
    for d in result["documents"]:
        d["retrieved"] = retrieved

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("corpus", build_corpus_meta(retrieved, args.topic)),
        ("documents", result["documents"]),
        ("segments", result["segments"]),
    ]
    for name, data in outputs:
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  Wrote {path.name} ({len(data) if isinstance(data, list) else 1:,})")

    report(result["documents"], result["segments"])

    segments, documents = result["segments"], result["documents"]
    assert all(s["text_sha256"] == sha256(s["text"]) for s in segments), "hash mismatch"
    assert len({s["segment_id"] for s in segments}) == len(segments), "duplicate segment_id"
    assert len({d["doc_id"] for d in documents}) == len(documents), "duplicate doc_id"
    print("\nChecks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
