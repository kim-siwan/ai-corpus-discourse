"""HW3 diagnostic of the HW2 model. Reads training/ artifacts; writes only training/results/hw3_diag/out.json.

Run from the repo root:
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 training/.venv/bin/python hw3/diagnostics/diag.py
Writes training/results/hw3_diag/out.json.
"""
import hashlib, json, pathlib, re, sys
from collections import Counter

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]  # repo root (hw3/diagnostics/ -> repo)
TR = REPO / "training"
sys.path.insert(0, str(TR))
import cdcp  # noqa: E402
from model import SentenceTypeMLP  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

L = cdcp.LABELS
PI, VI = L.index("policy"), L.index("value")
out = {}

# ---------- embeddings: read the existing cache only; never write to it
def cache_path(texts):
    h = hashlib.sha256(); h.update(b"minilm")
    for t in texts: h.update(b"\0" + t.encode())
    return TR / "cache" / f"minilm-{h.hexdigest()[:16]}.npy"

_enc = None
def encode_new(texts):
    global _enc
    if _enc is None:
        from sentence_transformers import SentenceTransformer
        _enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(_enc.encode(texts, batch_size=64, normalize_embeddings=True), dtype=np.float32)

splits = {}
for s in ["train", "dev", "test"]:
    ex = cdcp.load(s)
    texts = [e.text for e in ex]
    p = cache_path(texts)
    X = np.load(p) if p.exists() else encode_new(texts)
    splits[s] = dict(texts=texts, y=np.array(cdcp.label_index([e.label for e in ex])), X=X,
                     cached=p.exists())

# ---------- model
cfg = json.loads((TR / "artifacts/config.json").read_text())
model = SentenceTypeMLP(384, cfg["hidden"], 4, cfg["activation"], cfg["dropout"])
model.load_state_dict(torch.load(TR / "artifacts/model.pt", map_location="cpu"))
model.eval()
def proba(X):
    return model.predict_proba(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

# ================= confusion matrix and errors (dev)
out["cache_hit"] = {s: d["cached"] for s, d in splits.items()}
out["counts"] = {s: {l: int((d["y"] == i).sum()) for i, l in enumerate(L)} | {"total": len(d["y"])}
                 for s, d in splits.items()}
dev = splits["dev"]
P = proba(dev["X"]); pred = P.argmax(1); y = dev["y"]
out["dev_error"] = float((pred != y).mean())
out["cm"] = confusion_matrix(y, pred, labels=range(4)).tolist()
p, r, f, sup = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
out["prf"] = {l: dict(p=float(p[i]), r=float(r[i]), f1=float(f[i]), n=int(sup[i])) for i, l in enumerate(L)}
out["macro_f1"] = float(f.mean())

# wrongly predicted as value -> max softmax (= P(value))
wv = {}
for i, l in enumerate(L):
    if l == "value": continue
    m = (pred == VI) & (y == i)
    mx = P[m].max(1)
    wv[l] = dict(n=int(m.sum()), mean=float(mx.mean()) if m.any() else None,
                 median=float(np.median(mx)) if m.any() else None,
                 bins=[int((mx < .5).sum()), int(((mx >= .5) & (mx < .7)).sum()),
                       int(((mx >= .7) & (mx < .9)).sum()), int((mx >= .9).sum())])
m = (pred == VI) & (y != VI); mx = P[m].max(1)
wv["ALL"] = dict(n=int(m.sum()), mean=float(mx.mean()), median=float(np.median(mx)),
                 bins=[int((mx < .5).sum()), int(((mx >= .5) & (mx < .7)).sum()),
                       int(((mx >= .7) & (mx < .9)).sum()), int((mx >= .9).sum())])
# reference: correctly predicted value
mc = (pred == VI) & (y == VI); mxc = P[mc].max(1)
wv["value (correct, reference)"] = dict(n=int(mc.sum()), mean=float(mxc.mean()), median=float(np.median(mxc)),
    bins=[int((mxc < .5).sum()), int(((mxc >= .5) & (mxc < .7)).sum()),
          int(((mxc >= .7) & (mxc < .9)).sum()), int((mxc >= .9).sum())])
out["wrong_value_conf"] = wv

# 8 + 8 policy examples, random with seed 0
rng = np.random.default_rng(0)
ok = np.where((y == PI) & (pred == PI))[0]; miss = np.where((y == PI) & (pred == VI))[0]
pick = lambda idx: sorted(rng.choice(idx, size=min(8, len(idx)), replace=False).tolist())
out["policy_ok_n"], out["policy_to_value_n"] = len(ok), len(miss)
out["policy_ok"] = [dict(i=i, text=dev["texts"][i], pp=float(P[i, PI]), pv=float(P[i, VI])) for i in pick(ok)]
out["policy_miss"] = [dict(i=i, text=dev["texts"][i], pp=float(P[i, PI]), pv=float(P[i, VI])) for i in pick(miss)]

# ================= minimal pairs: should/must/need to/has to -> can / will (dev policy)
MODAL = re.compile(r"\b(should|must|need to|has to)\b", re.I)
pol = [i for i in np.where(y == PI)[0] if MODAL.search(dev["texts"][i])]
orig = [dev["texts"][i] for i in pol]
var_can = [MODAL.sub("can", t) for t in orig]
var_will = [MODAL.sub("will", t) for t in orig]
Po = P[pol]; Pc = proba(encode_new(var_can)); Pw = proba(encode_new(var_will))
def summ(Pv):
    a = Pv.argmax(1)
    return dict(mean_pp=float(Pv[:, PI].mean()), median_pp=float(np.median(Pv[:, PI])),
                n_policy=int((a == PI).sum()), dist={L[k]: int((a == k).sum()) for k in range(4)})
o_pol = Po.argmax(1) == PI
out["mp"] = dict(n=len(pol), n_policy_total=int((y == PI).sum()),
                 modal_counts=dict(Counter(MODAL.search(t).group(1).lower() for t in orig)),
                 orig=summ(Po), can=summ(Pc), will=summ(Pw),
                 flip_can=int((o_pol & (Pc.argmax(1) != PI)).sum()),
                 flip_will=int((o_pol & (Pw.argmax(1) != PI)).sum()),
                 gain_can=int((~o_pol & (Pc.argmax(1) == PI)).sum()),
                 gain_will=int((~o_pol & (Pw.argmax(1) == PI)).sum()),
                 delta_can=float((Pc[:, PI] - Po[:, PI]).mean()),
                 delta_will=float((Pw[:, PI] - Po[:, PI]).mean()),
                 n_pp_drop_can=int((Pc[:, PI] < Po[:, PI]).sum()),
                 n_pp_drop_will=int((Pw[:, PI] < Po[:, PI]).sum()),
                 cos_can=float(np.mean(np.sum(dev["X"][pol] * encode_new(var_can), 1))),
                 cos_will=float(np.mean(np.sum(dev["X"][pol] * encode_new(var_will), 1))))
# biggest drops for illustration
drop = Po[:, PI] - Pc[:, PI]
out["mp_examples"] = [dict(o=orig[k], c=var_can[k], po=float(Po[k, PI]), pc=float(Pc[k, PI]), pw=float(Pw[k, PI]))
                      for k in np.argsort(-drop)[:4]] + \
                     [dict(o=orig[k], c=var_can[k], po=float(Po[k, PI]), pc=float(Pc[k, PI]), pw=float(Pw[k, PI]))
                      for k in np.argsort(drop)[:2]]

# three probe sentences: should / can / requires
s7 = ["The agency should require collectors to verify the debt.",
      "The agency can require collectors to verify the debt.",
      "The agency requires collectors to verify the debt."]
P7 = proba(encode_new(s7))
out["q7"] = [dict(text=t, **{L[k]: float(P7[j, k]) for k in range(4)}) for j, t in enumerate(s7)]

# ================= TF-IDF + LogReg refit (HW2 saved no TF-IDF model)
tr, te = splits["train"], splits["test"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(tr["texts"])
lr = LogisticRegression(max_iter=3000, C=5).fit(Xtr, tr["y"])
out["tfidf_check"] = dict(dev_err=float((lr.predict(vec.transform(dev["texts"])) != y).mean()),
                          test_err=float((lr.predict(vec.transform(te["texts"])) != te["y"]).mean()))
vocab = np.array(vec.get_feature_names_out())
for name, k in [("policy", PI), ("value", VI)]:
    w = lr.coef_[k]; top = np.argsort(-w)[:20]
    out[f"top_{name}"] = [(vocab[j], float(w[j])) for j in top]

# ================= modal probe: can a linear model read "contains should/must/need to/has to"?
def lab(texts): return np.array([1 if MODAL.search(t) else 0 for t in texts])
ytr4, ydv4 = lab(tr["texts"]), lab(dev["texts"])
out["probe_rate"] = dict(train=float(ytr4.mean()), dev=float(ydv4.mean()), n_dev_pos=int(ydv4.sum()))
maj = int(np.bincount(ytr4).argmax())
pm = np.full_like(ydv4, maj)
out["probe_majority"] = dict(cls=maj, f1_pos=float(f1_score(ydv4, pm, zero_division=0)),
                             macro=float(f1_score(ydv4, pm, average="macro", zero_division=0)))
res = {}
m4 = LogisticRegression(max_iter=3000, C=1.0).fit(tr["X"], ytr4); pd4 = m4.predict(dev["X"])
res["minilm_C1"] = dict(f1_pos=float(f1_score(ydv4, pd4)), macro=float(f1_score(ydv4, pd4, average="macro")),
                        p=float(precision_recall_fscore_support(ydv4, pd4, average="binary")[0]),
                        r=float(precision_recall_fscore_support(ydv4, pd4, average="binary")[1]))
m4b = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(tr["X"], ytr4); pd4b = m4b.predict(dev["X"])
prb = precision_recall_fscore_support(ydv4, pd4b, average="binary")
res["minilm_C1_balanced"] = dict(f1_pos=float(prb[2]), macro=float(f1_score(ydv4, pd4b, average="macro")),
                                 p=float(prb[0]), r=float(prb[1]))
v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
m4t = LogisticRegression(max_iter=3000, C=5).fit(v2.fit_transform(tr["texts"]), ytr4)
pd4t = m4t.predict(v2.transform(dev["texts"]))
prt = precision_recall_fscore_support(ydv4, pd4t, average="binary")
res["tfidf_C5"] = dict(f1_pos=float(prt[2]), macro=float(f1_score(ydv4, pd4t, average="macro")),
                       p=float(prt[0]), r=float(prt[1]))
# TF-IDF with the four cue words removed from the vocabulary (is the label just a lookup?)
v3 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                     stop_words=["should", "must", "need", "has", "to"])
m4s = LogisticRegression(max_iter=3000, C=5).fit(v3.fit_transform(tr["texts"]), ytr4)
pd4s = m4s.predict(v3.transform(dev["texts"]))
prs = precision_recall_fscore_support(ydv4, pd4s, average="binary", zero_division=0)
res["tfidf_cues_removed"] = dict(f1_pos=float(prs[2]), macro=float(f1_score(ydv4, pd4s, average="macro")),
                                 p=float(prs[0]), r=float(prs[1]))
out["probe"] = res

json.dump(out, open(REPO / "training/results/hw3_diag" / "out.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
