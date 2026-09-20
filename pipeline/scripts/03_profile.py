"""Profile the corpus to size the length-bias problem.

Reports how unevenly text mass is distributed across documents and respondent
types. These are the denominators for lift normalization in the alignment
analysis: a type that wrote 40% of the words will surface in 40% of any
unweighted result whether or not the policy listened to it.

    python3 scripts/03_profile.py        # run after 06

Reads the paragraphs in segments.json, whose lengths are stored as n_chars.
Respondent type is the docket's own (org_type_official), not IFP's.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EMBED_CHAR_LIMIT = 2000  # ~512 tokens for most sentence encoders


def percentiles(values: list, points=(10, 25, 50, 75, 90, 99)) -> dict:
    ordered = sorted(values)
    return {p: ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))] for p in points}


def main() -> int:
    documents = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
    segments = json.loads((DATA / "segments.json").read_text(encoding="utf-8"))
    by_id = {d["doc_id"]: d for d in documents}

    seg_count = Counter(s["doc_id"] for s in segments)
    char_count = defaultdict(int)
    for s in segments:
        char_count[s["doc_id"]] += s["n_chars"]

    print(f"{len(documents):,} documents / {len(segments):,} paragraphs")

    print("\n--- paragraphs per document ---")
    counts = [seg_count[d["doc_id"]] for d in documents]
    for p, v in percentiles(counts).items():
        print(f"  p{p:<3} {v}")
    print(f"  mean {sum(counts)/len(counts):.1f}   max {max(counts)}")
    print("  distribution:", dict(sorted(Counter(counts).items())))

    print("\n--- characters per document ---")
    chars = [char_count[d["doc_id"]] for d in documents]
    present = sorted(c for c in chars if c)
    for p, v in percentiles(chars).items():
        print(f"  p{p:<3} {v:,}")
    print(f"  mean {sum(chars)/len(chars):,.0f}   max {max(chars):,}")
    top, bottom = max(present), present[len(present) // 2]
    print(f"  largest document is {top / bottom:.0f}x the median")

    print("\n--- top 10 documents by text mass ---")
    for doc_id, n in sorted(char_count.items(), key=lambda kv: -kv[1])[:10]:
        d = by_id[doc_id]
        share = 100 * n / sum(chars)
        print(f"  {n:9,} chars ({share:4.1f}%)  {seg_count[doc_id]:3d} para  "
              f"{d['org'][:44]}  [{d['org_type_official']}]")

    print("\n--- respondent type: documents / paragraphs / chars ---")
    rows = defaultdict(lambda: [0, 0, 0])
    for d in documents:
        t = d["org_type_official"]
        rows[t][0] += 1
        rows[t][1] += seg_count[d["doc_id"]]
        rows[t][2] += char_count[d["doc_id"]]
    total_chars = sum(r[2] for r in rows.values())
    total_segs = sum(r[1] for r in rows.values())
    print(f"  {'type':<46} {'docs':>5} {'paras':>6} {'para%':>6} {'char%':>6}")
    for t, (nd, ns, nc) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
        print(f"  {t:<46} {nd:5d} {ns:6d} {100*ns/total_segs:5.1f}% {100*nc/total_chars:5.1f}%")

    print("\n--- paragraphs needing a split before embedding ---")
    long_segments = [s for s in segments if s["n_chars"] > EMBED_CHAR_LIMIT]
    print(f"  {len(long_segments)} of {len(segments):,} paragraphs exceed {EMBED_CHAR_LIMIT:,} chars "
          f"({100*len(long_segments)/len(segments):.2f}%)")
    print(f"  they hold {100*sum(s['n_chars'] for s in long_segments)/sum(chars):.2f}% "
          f"of all text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
