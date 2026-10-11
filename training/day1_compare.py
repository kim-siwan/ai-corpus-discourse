"""Day 1 go/no-go: can a network on sentence embeddings beat the shallow baseline?

This is a scouting run with off-the-shelf scikit-learn models, NOT the HW2
network. The HW2 network (own PyTorch training loop) is Day 2.

Three candidates, same CDCP train split, scored once on the CDCP test split:

  1. TF-IDF + logistic regression   — the shallow benchmark (rubric: +20 needs a lower error)
  2. embedding + logistic regression — a network with no hidden layer on the same input
  3. embedding + MLP (1 hidden layer) — the shape of the HW2 model

(2) vs (3) isolates what the hidden layer adds; (1) vs (2) what the encoder adds.

Known from the cloud probe (2026-10-05): on TF-IDF and on LSA the MLP does NOT
beat logistic regression. The hope is that a semantic embedding changes that.

    python training/day1_compare.py                  # minilm
    python training/day1_compare.py --encoder mpnet
    python training/day1_compare.py --encoder lsa    # offline pipeline check

Writes training/results/day1_<encoder>.json.
"""

import argparse
import json
import pathlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

import cdcp
from embed import embed

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def score(y_true, y_pred) -> dict:
    per = f1_score(y_true, y_pred, labels=cdcp.LABELS, average=None, zero_division=0)
    return {
        "error": round(1 - accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=cdcp.LABELS, average="macro", zero_division=0), 4),
        "per_class_f1": {l: round(float(v), 3) for l, v in zip(cdcp.LABELS, per)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    train, test = cdcp.load("train"), cdcp.load("test")
    Xtr_text, ytr = [e.text for e in train], [e.label for e in train]
    Xte_text, yte = [e.text for e in test], [e.label for e in test]
    print(f"CDCP train {len(train)} / test {len(test)} propositions, labels {cdcp.LABELS}")

    rows = []

    # 1. shallow benchmark
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=3000, C=5).fit(vec.fit_transform(Xtr_text), ytr)
    rows.append({"model": "TF-IDF + LogReg (shallow benchmark)",
                 **score(yte, lr.predict(vec.transform(Xte_text)))})

    # 2 & 3. on the frozen embedding
    Etr = embed(Xtr_text, args.encoder, fit_texts=Xtr_text)
    Ete = embed(Xte_text, args.encoder, fit_texts=Xtr_text)
    print(f"embedding: {args.encoder}, dim {Etr.shape[1]}")

    lr2 = LogisticRegression(max_iter=3000, C=1).fit(Etr, ytr)
    rows.append({"model": f"{args.encoder} + LogReg (no hidden layer)", **score(yte, lr2.predict(Ete))})

    mlp_runs = []
    for seed in range(args.seeds):
        mlp = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
                            early_stopping=True, validation_fraction=0.15, random_state=seed)
        mlp.fit(Etr, ytr)
        mlp_runs.append(score(yte, mlp.predict(Ete)))
    rows.append({
        "model": f"{args.encoder} + MLP (1 hidden, 128)",
        "error": round(float(np.mean([r["error"] for r in mlp_runs])), 4),
        "macro_f1": round(float(np.mean([r["macro_f1"] for r in mlp_runs])), 4),
        "per_class_f1": {l: round(float(np.mean([r["per_class_f1"][l] for r in mlp_runs])), 3)
                         for l in cdcp.LABELS},
        "runs": mlp_runs,
    })

    print(f"\n{'model':42s} {'error':>7s} {'macroF1':>8s}  " + "  ".join(f"{l[:6]:>6s}" for l in cdcp.LABELS))
    for r in rows:
        print(f"{r['model']:42s} {r['error']:7.3f} {r['macro_f1']:8.3f}  "
              + "  ".join(f"{r['per_class_f1'][l]:6.2f}" for l in cdcp.LABELS))

    base, mlp_row = rows[0], rows[2]
    beats = mlp_row["error"] < base["error"]
    verdict = ("GO — the MLP has a lower error than the shallow benchmark."
               if beats else
               "NO-GO — try --encoder mpnet; then a 2nd hidden layer + dropout in the Day 2 loop.")
    print(f"\n{verdict}")
    if beats and mlp_row["macro_f1"] < base["macro_f1"]:
        print("Note: lower error but lower macro-F1 — it is winning on the big classes only. "
              "Decide which metric you report before tuning.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"day1_{args.encoder}.json"
    out.write_text(json.dumps({"encoder": args.encoder, "n_train": len(train), "n_test": len(test),
                               "rows": rows, "mlp_beats_shallow_on_error": beats}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
