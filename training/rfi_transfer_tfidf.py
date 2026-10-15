"""Does the shallow benchmark transfer to the RFI better than MiniLM does?

MiniLM + MLP calls 66% of CDCP dev sentences with should / must / need to /
require "policy", but only 19% of such RFI sentences (CLAUDE.md, Decision
log 2026-10-11). If TF-IDF + LogReg — lexical, so it sees "should" directly —
holds up on the RFI, the drop is the encoder's. If it drops as far, it is
the domain.

Same benchmark as day1_compare.py (CDCP train only; test is not touched),
same RFI sentences as the served model (training/predictions/cdcp-pred-v1.jsonl,
built by backend/predict_corpus.py), same deontic pattern as that report.

    python training/rfi_transfer_tfidf.py      # seconds

Writes training/results/rfi_transfer_tfidf.json.
"""

import json
import pathlib
import random
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import cdcp

HERE = pathlib.Path(__file__).resolve().parent
PREDICTIONS = HERE / "predictions" / "cdcp-pred-v1.jsonl"
MINILM_REPORT = HERE / "results" / "rfi_policy_share.json"
OUT = HERE / "results" / "rfi_transfer_tfidf.json"

# Identical to backend/predict_corpus.py, so the 19% is comparable.
DEONTIC = re.compile(r"\b(should|must|ought to|needs? to|requires?|required)\b", re.IGNORECASE)
# The narrower reading of "should/must sentences".
SHOULD_MUST = re.compile(r"\b(should|must)\b", re.IGNORECASE)


def model_text(sentence: str) -> str:
    """Same cleanup the served model applies (backend/predict.py, model_text)."""
    sentence = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", sentence)
    return " ".join(sentence.split())


def share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def among(pattern: re.Pattern, texts: list[str], policy: np.ndarray) -> dict:
    has = np.array([bool(pattern.search(t)) for t in texts])
    return {"sentences_with_pattern": int(has.sum()), "share_with_pattern": share(int(has.sum()), len(texts)),
            "predicted_policy_among_them": share(int((policy & has).sum()), int(has.sum()))}


def main() -> None:
    if not PREDICTIONS.exists():
        raise SystemExit(f"{PREDICTIONS} missing — run backend/predict_corpus.py first")

    # ---- the benchmark, exactly as in day1_compare.py
    train = cdcp.load("train")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=3000, C=5).fit(vec.fit_transform([e.text for e in train]),
                                                    [e.label for e in train])

    # ---- CDCP dev reference (true labels known)
    dev = cdcp.load("dev")
    dev_texts = [e.text for e in dev]
    dev_pred = lr.predict(vec.transform(dev_texts))
    dev_true = np.array([e.label == "policy" for e in dev])
    has = np.array([bool(DEONTIC.search(t)) for t in dev_texts])
    cdcp_dev = {**among(DEONTIC, dev_texts, dev_pred == "policy"),
                "true_policy_among_them": share(int((dev_true & has).sum()), int(has.sum())),
                "dev_error": round(float((dev_pred != np.array([e.label for e in dev])).mean()), 4)}

    # ---- RFI: the served model's sentences and labels
    rows = [json.loads(line) for line in PREDICTIONS.open(encoding="utf-8")]
    texts = [model_text(r["quote"]) for r in rows]
    minilm = np.array([r["tag"] for r in rows])
    tfidf = lr.predict(vec.transform(texts))
    t_pol, m_pol = tfidf == "policy", minilm == "policy"

    minilm_report = json.loads(MINILM_REPORT.read_text())
    counts = Counter(tfidf)
    deontic = np.array([bool(DEONTIC.search(t)) for t in texts])

    def examples(mask: np.ndarray, k: int = 10) -> list[dict]:
        idx = sorted(np.flatnonzero(mask).tolist())
        pick = random.Random(7).sample(idx, min(k, len(idx)))
        return [{"segment_id": rows[i]["segment_id"], "start": rows[i]["start"], "text": texts[i],
                 "tfidf": str(tfidf[i]), "minilm": str(minilm[i]),
                 "minilm_p_policy": rows[i]["probs"]["policy"]} for i in sorted(pick)]

    report = {
        "benchmark": "TF-IDF (1-2 grams, min_df 2, sublinear) + LogReg (C=5), trained on CDCP train",
        "rfi_sentences": len(rows),
        "rfi_predicted_labels": {
            "tfidf": {l: share(counts[l], len(rows)) for l in cdcp.LABELS},
            "minilm": {l: v["share"] for l, v in minilm_report["predicted_labels"].items()},
        },
        "deontic_pattern": {
            "pattern": DEONTIC.pattern,
            "rfi": {"tfidf": among(DEONTIC, texts, t_pol), "minilm": among(DEONTIC, texts, m_pol)},
            "cdcp_dev": {"tfidf": cdcp_dev,
                         "minilm": {k: minilm_report["deontic_heuristic"]["cdcp_dev"][k]
                                    for k in ("predicted_policy_among_them", "true_policy_among_them")}},
        },
        "should_must_only": {
            "pattern": SHOULD_MUST.pattern,
            "rfi": {"tfidf": among(SHOULD_MUST, texts, t_pol), "minilm": among(SHOULD_MUST, texts, m_pol)},
        },
        "agreement": {
            "same_label": share(int((tfidf == minilm).sum()), len(rows)),
            "policy_both": int((t_pol & m_pol).sum()),
            "policy_tfidf_only": int((t_pol & ~m_pol).sum()),
            "policy_minilm_only": int((~t_pol & m_pol).sum()),
        },
        "examples_deontic": {
            "tfidf_policy_minilm_value": examples(deontic & t_pol & (minilm == "value")),
            "minilm_policy_tfidf_not": examples(deontic & m_pol & ~t_pol),
        },
        "note": "Heuristic comparison, not accuracy: no RFI sentence has a true label yet.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    d, s = report["deontic_pattern"], report["should_must_only"]
    print(f"RFI sentences: {len(rows):,}")
    print(f"predicted policy share   TF-IDF {report['rfi_predicted_labels']['tfidf']['policy']:.1%}   "
          f"MiniLM {report['rfi_predicted_labels']['minilm']['policy']:.1%}")
    print(f"should/must/need to/require sentences ({d['rfi']['tfidf']['share_with_pattern']:.1%} of RFI), "
          f"predicted policy:")
    print(f"  RFI       TF-IDF {d['rfi']['tfidf']['predicted_policy_among_them']:.1%}   "
          f"MiniLM {d['rfi']['minilm']['predicted_policy_among_them']:.1%}")
    print(f"  CDCP dev  TF-IDF {d['cdcp_dev']['tfidf']['predicted_policy_among_them']:.1%}   "
          f"MiniLM {d['cdcp_dev']['minilm']['predicted_policy_among_them']:.1%}   "
          f"(truly policy {d['cdcp_dev']['tfidf']['true_policy_among_them']:.1%})")
    print(f"should/must only, RFI   TF-IDF {s['rfi']['tfidf']['predicted_policy_among_them']:.1%}   "
          f"MiniLM {s['rfi']['minilm']['predicted_policy_among_them']:.1%}")
    a = report["agreement"]
    print(f"same label {a['same_label']:.1%}; policy: both {a['policy_both']:,}, "
          f"TF-IDF only {a['policy_tfidf_only']:,}, MiniLM only {a['policy_minilm_only']:,}")
    print(f"wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
