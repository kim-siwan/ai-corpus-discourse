"""FT-all seed 0 on CDCP dev -> blank error coding sheet (groups A, B, C).

Loads only the dev split (cdcp.load("dev")); the test split is never read.
"""
import csv
import json
import pathlib
import random
import re
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]  # repo root (hw3/diagnostics/ -> repo)
sys.path.insert(0, str(REPO / "training"))
import cdcp  # noqa: E402
import finetune as ft  # noqa: E402

RUN = REPO / "training/runs/hw3/ft_all_seed0"
OUT = REPO / "training/results/hw3_diag/hw3_error_coding_blank.csv"  # kept apart from the hand-coded sheet so a rerun can't overwrite it
MODAL = re.compile(r"\b(should|must|need to|has to)\b", re.I)  # same as hw3_overnight.py
SEED = 7

cfg = json.loads((RUN / "config.json").read_text())
model = ft.FineTuneModel(cfg)
model.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu", weights_only=True))
model.eval()

ex = cdcp.load("dev")
texts = [e.text for e in ex]
gold = np.array(cdcp.label_index([e.label for e in ex]))
P = model.predict_proba(texts)
pred = P.argmax(1)

# sanity check against the saved dev metrics
saved = json.loads((REPO / "training/results/hw3/ft_all_seed0/metrics.json").read_text())["dev"]
cm = np.zeros((4, 4), int)
for g, p in zip(gold, pred):
    cm[g, p] += 1
print("dev error", round(float((pred != gold).mean()), 4), "saved", round(saved["error"], 4))
print("confusion matches saved:", cm.tolist() == saved["confusion_matrix"]["rows_true_cols_pred"])

# id = <comment_id>.<proposition index within that comment, after dropping `reference`>
ids, k, prev = [], 0, None
for e in ex:
    k = k + 1 if e.comment_id == prev else 0
    prev = e.comment_id
    ids.append(f"{e.comment_id}.{k}")

POL = cdcp.LABELS.index("policy")
A = [i for i in range(len(ex)) if gold[i] == POL and pred[i] != POL]
B = [i for i in range(len(ex)) if pred[i] == POL and gold[i] != POL]
correct = [i for i in range(len(ex)) if gold[i] == POL and pred[i] == POL]
rng = random.Random(SEED)
C = rng.sample(correct, 10)
for g in (A, B, C):
    rng.shuffle(g)


def modal_info(text):
    toks = text.split()  # CDCP text is already tokenized; tokens = whitespace pieces
    m = MODAL.search(text)
    if not m:
        return len(toks), "n", ""
    start_tok = len(text[:m.start()].split())
    pos = start_tok / (len(toks) - 1) if len(toks) > 1 else 0.0
    return len(toks), "y", f"{pos:.2f}"


cols = ["id", "group", "text", "gold", "pred", "P(fact)", "P(policy)", "P(testimony)", "P(value)",
        "n_tokens", "has_modal", "modal_relative_position", "why_wrong", "ambiguous_to_me", "notes"]
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for name, group in (("A", A), ("B", B), ("C", C)):
        for i in group:
            n, has, pos = modal_info(texts[i])
            w.writerow([ids[i], name, texts[i], cdcp.LABELS[gold[i]], cdcp.LABELS[pred[i]],
                        *[f"{x:.3f}" for x in P[i]], n, has, pos, "", "", ""])

print(f"A (gold policy, missed): {len(A)}")
print(f"B (pred policy, gold not): {len(B)}")
print(f"C (gold policy, correct, random 10 of {len(correct)}): {len(C)}")
print("wrote", OUT)
