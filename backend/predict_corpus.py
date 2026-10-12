"""Label every sentence of the RFI corpus once (HW2, Day B step 2).

Uses the same code path as the live endpoints in predict.py — same sentence
splitter, same encoder, same network — so what the app shows from these files
is what POST /predict would return.

    training/.venv/bin/python backend/predict_corpus.py      # ~4 min, CPU

Writes three things:

  A. training/predictions/cdcp-pred-v1.jsonl   (gitignored, ~35 MB)
     One stand-off annotation per sentence, in the same fields as the
     annotations table, plus `probs`. Compare with hand codes (coder "kim")
     using the same code. Not written to annotations.db.
  B. viewer/public/predictions/                (committed, small)
     index.json             segment_id -> predicted label codes ("pv" = policy + value)
     docs/<doc_id>.json     segment_id -> [[start, end, label, [p_fact, p_policy, p_testimony, p_value]], ...]
  C. training/results/rfi_policy_share.json    how much of the RFI is predicted policy

Offsets are code points into the paragraph (segment) text, as everywhere else.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import predict
from predict import FULLTEXT, MODEL_INFO, ROOT, SEGMENTS, model_text, split_sentences

OUT_FULL = ROOT / "training" / "predictions" / "cdcp-pred-v1.jsonl"
OUT_VIEWER = ROOT / "viewer" / "public" / "predictions"
OUT_REPORT = ROOT / "training" / "results" / "rfi_policy_share.json"

CODES = {"fact": "f", "policy": "p", "testimony": "t", "value": "v"}
THRESHOLDS = [0.4, 0.3, 0.2]
SHORT = 4  # pieces under 4 words are mostly headings, page numbers, list markers
# A heuristic, not a label: "should" can appear outside a demand, and a
# demand can be phrased without any of these words.
DEONTIC = re.compile(r"\b(should|must|ought to|needs? to|requires?|required)\b", re.IGNORECASE)


def share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def main() -> None:
    p = predict.predictor
    if p is None:
        sys.exit(f"cannot predict: {predict.unavailable}")
    labels = p.labels
    POLICY = labels.index("policy")

    # ---- 1. every paragraph -> sentences (offsets into the paragraph)
    segments = json.loads(SEGMENTS.read_text())
    docs = {}
    rows = []  # (segment, start, end, quote, model_text)
    for seg in segments:
        if seg["doc_id"] not in docs:
            docs[seg["doc_id"]] = (FULLTEXT / f"{seg['doc_id']}.txt").read_text(encoding="utf-8")
        para = docs[seg["doc_id"]][seg["char_offset_in_doc"]:seg["char_end_in_doc"]]
        for start, end in split_sentences(para):
            rows.append((seg, start, end, para[start:end], model_text(para[start:end])))
    print(f"{len(segments):,} paragraphs in {len(docs)} documents -> {len(rows):,} sentences")

    # ---- 2. encode and classify, all at once
    X = p.encoder.encode([r[4] for r in rows], batch_size=128, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)
    probs = p.model.predict_proba(p.torch.from_numpy(X)).numpy()
    pred = probs.argmax(axis=1)

    # ---- A. full stand-off annotations
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FULL.open("w", encoding="utf-8") as f:
        for (seg, start, end, quote, _), pr, k in zip(rows, probs, pred):
            f.write(json.dumps({
                "ann_id": f"{MODEL_INFO['scheme']}:{seg['segment_id']}:{start}",
                "segment_id": seg["segment_id"], "doc_id": seg["doc_id"],
                "start": start, "end": end, "quote": quote, "tag": labels[k],
                "scheme": MODEL_INFO["scheme"], "coder": MODEL_INFO["coder"],
                "created": created, "note": "",
                "probs": {l: round(float(v), 4) for l, v in zip(labels, pr)},
            }, ensure_ascii=False) + "\n")

    # ---- B. viewer files
    by_doc = defaultdict(lambda: defaultdict(list))
    seg_codes = {s["segment_id"]: set() for s in segments}
    for (seg, start, end, _, _), pr, k in zip(rows, probs, pred):
        by_doc[seg["doc_id"]][seg["segment_id"]].append(
            [start, end, labels[k], [round(float(v), 3) for v in pr]])
        seg_codes[seg["segment_id"]].add(CODES[labels[k]])
    (OUT_VIEWER / "docs").mkdir(parents=True, exist_ok=True)
    for doc_id, segs in by_doc.items():
        (OUT_VIEWER / "docs" / f"{doc_id}.json").write_text(
            json.dumps({"scheme": MODEL_INFO["scheme"], "segments": segs}, separators=(",", ":")))
    order = "".join(CODES[l] for l in labels)
    index = {
        "model": MODEL_INFO, "created": created, "labels": labels,
        "codes": {c: l for l, c in CODES.items()},
        "n_sentences": len(rows),
        "segments": {sid: "".join(c for c in order if c in cs) for sid, cs in seg_codes.items()},
    }
    (OUT_VIEWER / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    # ---- C. how big is the problem?
    n = len(rows)
    words = np.array([len(r[4].split()) for r in rows])
    long_ = words >= SHORT
    counts = Counter(labels[k] for k in pred)
    counts_long = Counter(labels[k] for k in pred[long_])
    seg_of = [r[0]["segment_id"] for r in rows]
    doc_of = [r[0]["doc_id"] for r in rows]
    n_paras = len({s for s in seg_of})

    def demand_units(is_policy: np.ndarray, unit: list[str]) -> int:
        return len({u for u, flag in zip(unit, is_policy) if flag})

    is_pol = pred == POLICY
    per_doc = defaultdict(lambda: [0, 0])
    for d, flag in zip(doc_of, is_pol):
        per_doc[d][0] += int(flag)
        per_doc[d][1] += 1
    doc_shares = np.array([a / b for a, b in per_doc.values()])

    by_threshold = [{"rule": "argmax", "policy_sentences": int(is_pol.sum()),
                     "share": share(int(is_pol.sum()), n),
                     "paragraphs_with_demand": share(demand_units(is_pol, seg_of), n_paras)}]
    for t in THRESHOLDS:
        flag = probs[:, POLICY] >= t
        by_threshold.append({"rule": f"p_policy >= {t}", "policy_sentences": int(flag.sum()),
                             "share": share(int(flag.sum()), n),
                             "paragraphs_with_demand": share(demand_units(flag, seg_of), n_paras)})

    deontic = np.array([bool(DEONTIC.search(r[4])) for r in rows])
    cdcp_ref = cdcp_deontic_reference(p, POLICY)

    report = {
        "model": MODEL_INFO, "created": created,
        "corpus": {"documents": len(docs), "paragraphs": len(segments),
                   "paragraphs_with_sentences": n_paras, "sentences": n},
        "cdcp_policy_share": {"train": round(342 / 2742, 4), "test": round(93 / 836, 4)},
        "predicted_labels": {l: {"n": counts[l], "share": share(counts[l], n)} for l in labels},
        "predicted_labels_without_short": {
            "min_words": SHORT, "short_pieces": int((~long_).sum()), "sentences": int(long_.sum()),
            **{l: {"n": counts_long[l], "share": share(counts_long[l], int(long_.sum()))} for l in labels}},
        "demand": {
            "paragraphs_with_policy": demand_units(is_pol, seg_of),
            "paragraphs_with_policy_share": share(demand_units(is_pol, seg_of), n_paras),
            "documents_with_policy": int((doc_shares > 0).sum()),
            "documents_without_policy": int((doc_shares == 0).sum()),
            "per_document_policy_share": {q: round(float(np.percentile(doc_shares, v)), 4)
                                          for q, v in [("p25", 25), ("median", 50), ("p75", 75)]},
        },
        "by_threshold": by_threshold,
        "deontic_heuristic": {
            "pattern": DEONTIC.pattern, "note": "heuristic, not ground truth",
            "rfi_sentences_with_pattern": int(deontic.sum()),
            "rfi_share_with_pattern": share(int(deontic.sum()), n),
            "rfi_predicted_policy_among_them": share(int((is_pol & deontic).sum()), int(deontic.sum())),
            "rfi_mean_p_policy_among_them": round(float(probs[deontic, POLICY].mean()), 4),
            "cdcp_dev": cdcp_ref,
        },
    }
    OUT_REPORT.parent.mkdir(exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nwrote {OUT_FULL.relative_to(ROOT)}, {OUT_VIEWER.relative_to(ROOT)}/ "
          f"({len(by_doc)} docs), {OUT_REPORT.relative_to(ROOT)}")


def cdcp_deontic_reference(p, POLICY: int) -> dict:
    """The same heuristic on CDCP dev, where the true labels are known."""
    import cdcp  # training/ is on sys.path via predict.Predictor
    from embed import embed

    dev = cdcp.load("dev")
    probs = p.model.predict_proba(p.torch.from_numpy(embed([e.text for e in dev], "minilm"))).numpy()
    pred = probs.argmax(axis=1)
    true = np.array([e.label == "policy" for e in dev])
    has = np.array([bool(DEONTIC.search(e.text)) for e in dev])
    return {"sentences": len(dev), "with_pattern": int(has.sum()),
            "share_with_pattern": share(int(has.sum()), len(dev)),
            "true_policy_among_them": share(int((true & has).sum()), int(has.sum())),
            "predicted_policy_among_them": share(int(((pred == POLICY) & has).sum()), int(has.sum())),
            "pattern_recall_of_true_policy": share(int((true & has).sum()), int(true.sum()))}


def print_report(r: dict) -> None:
    c = r["corpus"]
    print(f"\n== RFI: {c['sentences']:,} sentences, {c['paragraphs_with_sentences']:,} paragraphs, "
          f"{c['documents']} documents")
    print("predicted label shares (all / without pieces under 4 words):")
    for l in r["predicted_labels"]:
        a, b = r["predicted_labels"][l], r["predicted_labels_without_short"][l]
        print(f"  {l:10s} {a['share']:6.1%} ({a['n']:,})   {b['share']:6.1%}")
    print(f"  CDCP policy share: train {r['cdcp_policy_share']['train']:.1%}, test {r['cdcp_policy_share']['test']:.1%}")
    d = r["demand"]
    print(f"paragraphs with >= 1 policy sentence: {d['paragraphs_with_policy']:,} ({d['paragraphs_with_policy_share']:.1%})")
    print(f"documents with / without any policy sentence: {d['documents_with_policy']} / {d['documents_without_policy']}")
    q = d["per_document_policy_share"]
    print(f"per-document policy share: p25 {q['p25']:.1%}  median {q['median']:.1%}  p75 {q['p75']:.1%}")
    print("by threshold:")
    for t in r["by_threshold"]:
        print(f"  {t['rule']:16s} sentences {t['share']:6.1%} ({t['policy_sentences']:,})   "
              f"paragraphs with demand {t['paragraphs_with_demand']:6.1%}")
    h = r["deontic_heuristic"]
    cd = h["cdcp_dev"]
    print(f"sentences with should/must/need to/require (heuristic): RFI {h['rfi_share_with_pattern']:.1%}, "
          f"CDCP dev {cd['share_with_pattern']:.1%}")
    print(f"  of those, predicted policy: RFI {h['rfi_predicted_policy_among_them']:.1%} "
          f"(mean p_policy {h['rfi_mean_p_policy_among_them']:.2f}); CDCP dev {cd['predicted_policy_among_them']:.1%} "
          f"(truly policy there: {cd['true_policy_among_them']:.1%})")


if __name__ == "__main__":
    main()
