"""Frame-PREPEND diagnostic. Reads the repo only; writes training/results/hw3_diag/frames.json."""
import json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parents[2]  # repo root (hw3/diagnostics/ -> repo)
TR = REPO / "training"
sys.path.insert(0, str(TR))
import cdcp, finetune as ft
from model import SentenceTypeMLP
from sentence_transformers import SentenceTransformer
L = cdcp.LABELS; PI = L.index("policy")

base = json.loads((TR / "results/hw3/partC_ft_all_seed0.json").read_text())["hw2_set"]["texts"]
assert len(base) == 10

# HW2: frozen MiniLM + saved head
cfg = json.loads((TR / "artifacts/config.json").read_text())
head = SentenceTypeMLP(384, cfg["hidden"], 4, cfg["activation"], cfg["dropout"])
head.load_state_dict(torch.load(TR / "artifacts/model.pt", map_location="cpu", weights_only=True)); head.eval()
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
def hw2(texts):
    X = enc.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return head.predict_proba(torch.from_numpy(np.asarray(X, np.float32))).numpy()

# FT-all seed 0
run = TR / "runs/hw3/ft_all_seed0"
m = ft.FineTuneModel(json.loads((run / "config.json").read_text()))
m.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True)); m.eval()
def ftall(texts):
    with torch.no_grad():
        return m.predict_proba(texts)

FR = {"S1": "I am writing this comment in response to the proposed rule",
      "S2": "We are writing this comment in response to the proposed rule",
      "S3": "This comment is written in response to the proposed rule",
      "S4": "I have been dealing with collectors for years",
      "S5": "Our organization has followed this issue for years"}
lower1 = lambda s: s[:1].lower() + s[1:]
variants = {"S0": list(base)} | {k: [f"{f}, and {lower1(s)}" for s in base] for k, f in FR.items()}

out = {"base": base, "example": {k: v[0] for k, v in variants.items()}}
for name, fn in [("hw2", hw2), ("ft_all_seed0", ftall)]:
    res = {}
    for k, texts in variants.items():
        P = np.asarray(fn(texts))
        res[k] = {**{f"p_{l}": float(P[:, j].mean()) for j, l in enumerate(L)},
                  "n_policy": int((P.argmax(1) == PI).sum()),
                  "dist": {l: int((P.argmax(1) == j).sum()) for j, l in enumerate(L)},
                  "per_sentence_p_policy": P[:, PI].round(3).tolist()}
    Pf = np.asarray(fn(list(FR.values())))
    res["frame_alone"] = {k: {l: float(Pf[i, j]) for j, l in enumerate(L)} for i, k in enumerate(FR)}
    out[name] = res
json.dump(out, open(REPO / "training/results/hw3_diag" / "frames.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()}, indent=1))
