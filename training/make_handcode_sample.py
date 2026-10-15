"""Pick the RFI paragraphs Siwan hand-codes (Day C), blind to the model.

Stratified random: for each respondent type (org_type_official, from the
docket — never IFP's), 3 paragraphs of at least 15 words, drawn with seed 7
from the paragraphs sorted by segment_id, so the same seed always gives the
same sample. Purposive: paragraphs chosen on purpose (PURPOSIVE below),
marked "purposive": true. The random draw ignores the purposive list, so
choosing purposive paragraphs never changes the random ones; a paragraph in
both carries both marks.

Paragraphs whose sentences appear with model predictions in
results/rfi_transfer_tfidf.json are excluded, so no sampled paragraph has
been seen next to a prediction. The output carries no predictions either.

    python training/make_handcode_sample.py               # writes the sample
    python training/make_handcode_sample.py --candidates  # also prints purposive candidates

Writes training/results/handcode_sample.json.
"""

import argparse
import json
import pathlib
import random
import re
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results" / "handcode_sample.json"
SEEN = HERE / "results" / "rfi_transfer_tfidf.json"

SEED = 7
PER_TYPE = 3
MIN_WORDS = 15

# Chosen on purpose (Siwan). Anthropic and OpenAI picks pending: paragraph 7
# of each opens with a stray footnote number and is not a demand.
PURPOSIVE = [
    "nitrd-90fr9088-accolade-ai-rfi-2025#p0007",
]
# Where purposive candidates come from, and what makes one a candidate.
CANDIDATE_DOCS = ["nitrd-90fr9088-anthropic-ai-rfi-2025", "nitrd-90fr9088-openai-rfi-2025"]
CANDIDATE_PATTERN = re.compile(r"\b(should|must|recommend\w*)\b", re.IGNORECASE)

CODING = {
    "tags": ["hw2:fact", "hw2:policy", "hw2:testimony", "hw2:value", "hw2:other"],
    "why_prefix": "hand codes live in annotations.db beside HW1 tags; the prefix keeps them separable",
    "before_coding": [
        "turn 'Show predicted types' OFF (the viewer remembers it per browser)",
        "code in your own Chrome, with the real backend (training/.venv, port 8000)",
        "note the footnote's 'Export N annotations' count before and after opening the viewer",
    ],
}


def load():
    docs = {d["doc_id"]: d for d in json.loads((ROOT / "pipeline/data/documents.json").read_text())}
    segs = json.loads((ROOT / "pipeline/data/segments.json").read_text())
    texts = {}

    def text(seg):
        if seg["doc_id"] not in texts:
            texts[seg["doc_id"]] = (ROOT / "pipeline/data/fulltext" / f"{seg['doc_id']}.txt").read_text(encoding="utf-8")
        return texts[seg["doc_id"]][seg["char_offset_in_doc"]:seg["char_end_in_doc"]]

    return docs, segs, text


def record(seg, doc, body, stratum, purposive):
    return {"segment_id": seg["segment_id"], "doc_id": seg["doc_id"], "org": doc["org"],
            "org_type_official": doc["org_type_official"], "para_index": seg["para_index"],
            "n_words": seg["n_words"], "text": " ".join(body.split()),
            "stratum": stratum, "purposive": purposive}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true", help="print purposive candidates (text only)")
    args = ap.parse_args()

    docs, segs, text = load()
    seen = set()
    if SEEN.exists():
        for group in json.loads(SEEN.read_text())["examples_deontic"].values():
            seen.update(e["segment_id"] for e in group)

    # ---- stratified random
    by_type = defaultdict(list)
    for s in segs:
        if s["n_words"] >= MIN_WORDS and s["segment_id"] not in seen:
            by_type[docs[s["doc_id"]]["org_type_official"]].append(s)
    rng = random.Random(SEED)
    picked = {}
    for t in sorted(by_type):  # fixed order, so the seed's stream is reproducible
        for s in rng.sample(sorted(by_type[t], key=lambda s: s["segment_id"]), PER_TYPE):
            picked[s["segment_id"]] = record(s, docs[s["doc_id"]], text(s), stratum=t, purposive=False)

    # ---- purposive
    seg_by_id = {s["segment_id"]: s for s in segs}
    for sid in PURPOSIVE:
        if sid in picked:
            picked[sid]["purposive"] = True
        else:
            s = seg_by_id[sid]
            picked[sid] = record(s, docs[s["doc_id"]], text(s), stratum=None, purposive=True)

    sample = sorted(picked.values(), key=lambda r: (r["purposive"] and r["stratum"] is None, r["org_type_official"], r["segment_id"]))
    OUT.write_text(json.dumps({
        "design": {"seed": SEED, "per_type": PER_TYPE, "min_words": MIN_WORDS,
                   "strata": "org_type_official (docket)", "excluded_seen_with_predictions": len(seen),
                   "purposive": PURPOSIVE},
        "coding": CODING,
        "n": len(sample),
        "paragraphs": sample,
    }, indent=2, ensure_ascii=False))
    print(f"{len(sample)} paragraphs ({sum(r['stratum'] is not None for r in sample)} random, "
          f"{sum(r['purposive'] for r in sample)} purposive) -> {OUT.relative_to(ROOT)}")

    if args.candidates:
        rng = random.Random(SEED)
        for doc_id in CANDIDATE_DOCS:
            pool = sorted((s for s in segs if s["doc_id"] == doc_id and s["n_words"] >= MIN_WORDS
                           and s["segment_id"] not in seen and CANDIDATE_PATTERN.search(text(s))),
                          key=lambda s: s["segment_id"])
            print(f"\n== {docs[doc_id]['org']}: {len(pool)} paragraphs match; 3 candidates")
            for s in sorted(rng.sample(pool, min(3, len(pool))), key=lambda s: s["para_index"]):
                print(f"\n[{s['segment_id']}]  paragraph {s['para_index']}, {s['n_words']} words")
                print(" ".join(text(s).split()))


if __name__ == "__main__":
    main()
