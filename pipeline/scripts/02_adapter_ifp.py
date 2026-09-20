"""Adapt the IFP snapshot into the shared corpus schema.

Adapter contract: adapter(raw) -> {documents, segments}
Downstream code (viewer, annotation UI, embeddings, models) reads only the
shared schema and does not know the source.

    python3 scripts/02_adapter_ifp.py                 # the whole corpus
    python3 scripts/02_adapter_ifp.py --topic "Security"   # IFP records on one topic

Documents come from the NITRD respondent listing (01b), not from IFP: every
organizational submission gets one, whether or not IFP indexed it. IFP
contributes two things to each, its topic labels and its own recommendation
records, and neither is used as an analysis variable.

--topic narrows the IFP records and therefore the topics each document
carries. It does not narrow the documents. Leave it empty to build the
corpus the viewer reads.

Output: data/corpus.json
        data/documents.json            755 submissions, with IFP topics
        data/ifp_recommendations.json  IFP's own segmentation, kept for reference

The paragraphs the viewer labels are built separately, by 06.
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
DEFAULT_TOPIC = ""
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


# IFP writes one topic two ways. Folding them keeps a filter from splitting
# 203 documents into two chips that mean the same thing.
TOPIC_ALIASES = {"global engagement": "Global Engagement"}


def split_topics(value) -> list:
    out = []
    for raw in str(value or "").split(","):
        name = raw.strip()
        if name:
            out.append(TOPIC_ALIASES.get(name.lower(), name))
    return out


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


def load_official() -> dict:
    """The NITRD respondent listing, keyed by file name.

    This is the document spine. Which submissions are organizational is the
    agency's determination, recorded in the docket, and not something to infer
    from whether a third party chose to index a submission.
    """
    snapshots = sorted((ROOT / "data" / "raw").glob("nitrd_responses_*.json"))
    if not snapshots:
        sys.exit("run scripts/01b_snapshot_nitrd.py first")
    rows = json.loads(snapshots[-1].read_text())
    return {
        r["FileName"]: r
        for r in rows
        if r.get("RespondentType") and r["RespondentType"] != "Individual"
    }


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

    official = load_official()

    # Every organizational submission gets a document, whether or not IFP
    # indexed it. Twelve have no IFP recommendations at all; dropping them
    # would make "what did IFP miss?" unanswerable.
    documents = {}
    for name, row in official.items():
        doc_id = doc_id_from_url(row["FileURL"])
        documents[doc_id] = {
            "doc_id": doc_id,
            "corpus": CORPUS_ID,
            "org": row["Respondent"],
            "org_type_official": row["RespondentType"],
            "org_type_ifp": None,        # filled below where IFP indexed it
            "date": RFI_CLOSING_DATE,
            "url": row["FileURL"],
            "retrieved": None,
            "reviewed": False,
            # Topic counts, not a flat set. A submission with fifty
            # recommendations touches most topics, so presence alone says
            # little; how often IFP tagged a topic here says more.
            "topics": {},
        }

    segments, skipped, outside = [], 0, 0

    for i, r in numbered:
        text = normalize(first(r.get("Full text")))
        if not text:
            skipped += 1
            continue

        url = first(r.get("URL"))
        doc_id = doc_id_from_url(url)

        # IFP indexed a few submissions the docket lists as Individual, and a
        # few it does not list at all. The official listing decides.
        if doc_id not in documents:
            outside += 1
            continue

        documents[doc_id]["org_type_ifp"] = first(r.get("Org Type")) or None
        for topic in split_topics(r.get("Topic")):
            documents[doc_id]["topics"][topic] = documents[doc_id]["topics"].get(topic, 0) + 1

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
    if outside:
        print(f"  Skipped {outside} records outside the official organizational listing")

    # Sort each document's topics by how often they occur in it.
    for d in documents.values():
        d["topics"] = [
            {"topic": k, "n": v}
            for k, v in sorted(d["topics"].items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    with_segments = {s["doc_id"] for s in segments}
    print(f"  {len(documents) - len(with_segments)} documents have no IFP recommendation")

    return {"documents": list(documents.values()), "segments": segments}


def build_corpus_meta(retrieved: str, topic_filter: str) -> dict:
    return {
        "corpus_id": CORPUS_ID,
        "source": "NITRD 90 FR 9088",
        "docket_url": "https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/",
        "normalization": NORMALIZATION,
        "retrieved": retrieved,
        "topic_filter": topic_filter or None,
        "document_spine": "NITRD respondent listing, RespondentType != Individual",
        "layers": {
            LAYER: {
                "extractor": "IFP AI Action Plan Database",
                "extractor_model": None,  # IFP says an LLM was used; the model is not published
                "extractor_url": "https://www.aiactionplan.org",
                "extractor_methodology": "https://www.aiactionplan.org/methodology",
                "extracted_field": "Full text",
                "note": (
                    "Text and derived labels in this layer were extracted by IFP using an "
                    "LLM, which IFP states but does not name. Topic and Org Type are "
                    "navigation aids only: org_type_official carries the respondent type "
                    "the agency recorded, and the two are incompatible for 79 of the 743 "
                    "documents both cover."
                ),
            }
        },
    }


def report(documents: list, segments: list) -> None:
    lengths = sorted(len(s["text"]) for s in segments)
    types = {}
    for d in documents:
        key = d["org_type_official"]
        types[key] = types.get(key, 0) + 1

    print(f"\n{len(documents):,} documents / {len(segments):,} segments")
    if lengths:
        print(f"Segment length: median {lengths[len(lengths)//2]:,} "
              f"/ min {lengths[0]:,} / max {lengths[-1]:,} chars")
    print("Documents by respondent type (NITRD, as declared in the docket):")
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
        ("ifp_recommendations", result["segments"]),
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
