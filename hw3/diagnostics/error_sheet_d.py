"""FT-all seed 0 on CDCP dev -> append group D to the blank sheet. Run after error_sheet.py.

D = 25 random (gold fact|testimony, pred value) + 10 random (gold value, pred fact|testimony), seed 7.
Only the dev split is loaded; test is never read.
"""
import csv
import json
import pathlib
import random
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]  # repo root (hw3/diagnostics/ -> repo)
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cdcp  # noqa: E402
import finetune as ft  # noqa: E402

RUN = REPO / "training/runs/hw3/ft_all_seed0"
OUT = REPO / "training/results/hw3_diag/hw3_error_coding_blank.csv"  # kept apart from the hand-coded sheet so a rerun can't overwrite it
SEED = 7
L = cdcp.LABELS

cfg = json.loads((RUN / "config.json").read_text())
model = ft.FineTuneModel(cfg)
model.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu", weights_only=True))
model.eval()

ex = cdcp.load("dev")
texts = [e.text for e in ex]
gold = np.array(cdcp.label_index([e.label for e in ex]))
P = model.predict_proba(texts)
pred = P.argmax(1)

cm = np.zeros((4, 4), int)
for g, p in zip(gold, pred):
    cm[g, p] += 1
saved = json.loads((REPO / "training/results/hw3/ft_all_seed0/metrics.json").read_text())["dev"]
print("confusion matches saved metrics.json:", cm.tolist() == saved["confusion_matrix"]["rows_true_cols_pred"])
print(f"\nFT-all seed 0, CDCP dev ({len(ex)} sentences), rows = gold, cols = predicted")
print(f"{'':>10}" + "".join(f"{l:>11}" for l in L) + f"{'n':>7}{'recall':>8}")
for i, l in enumerate(L):
    print(f"{l:>10}" + "".join(f"{v:>11}" for v in cm[i]) + f"{cm[i].sum():>7}{cm[i, i] / cm[i].sum():>8.2f}")
print(f"{'precision':>10}" + "".join(f"{cm[j, j] / cm[:, j].sum():>11.2f}" for j in range(4)))
print(f"error {(pred != gold).mean():.4f}\n")

ids, k, prev = [], 0, None
for e in ex:
    k = k + 1 if e.comment_id == prev else 0
    prev = e.comment_id
    ids.append(f"{e.comment_id}.{k}")

F, T, V = L.index("fact"), L.index("testimony"), L.index("value")
to_value = [i for i in range(len(ex)) if gold[i] in (F, T) and pred[i] == V]
from_value = [i for i in range(len(ex)) if gold[i] == V and pred[i] in (F, T)]
rng = random.Random(SEED)
d1 = rng.sample(to_value, 25)
d2 = rng.sample(from_value, 10)
D = d1 + d2
rng.shuffle(D)

from error_sheet_helpers import modal_info  # noqa: E402

with OUT.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for i in D:
        n, has, pos = modal_info(texts[i])
        w.writerow([ids[i], "D", texts[i], L[gold[i]], L[pred[i]],
                    *[f"{x:.3f}" for x in P[i]], n, has, pos, "", "", ""])

print(f"D1 (gold fact/testimony -> pred value): 25 of {len(to_value)}  "
      f"[fact {sum(gold[i] == F for i in d1)}, testimony {sum(gold[i] == T for i in d1)}]")
print(f"D2 (gold value -> pred fact/testimony): 10 of {len(from_value)}  "
      f"[fact {sum(pred[i] == F for i in d2)}, testimony {sum(pred[i] == T for i in d2)}]")
print(f"appended {len(D)} rows to {OUT}")
