"""HW3 diagnostic: is the modal lost to truncation or diluted by sentence length?
Writes training/results/hw3_diag/out2.json."""
import hashlib, json, pathlib, re, sys
import numpy as np, torch
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[2]  # repo root (hw3/diagnostics/ -> repo)
TR = REPO / "training"
sys.path.insert(0, str(TR))
import cdcp
from model import SentenceTypeMLP
from sentence_transformers import SentenceTransformer

L = cdcp.LABELS; PI, TI, VI = L.index("policy"), L.index("testimony"), L.index("value")
MODAL = re.compile(r"\b(should|must|need to|has to)\b", re.I)
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tok = enc.tokenizer
out = {}

# ---------- encoder limits: max length, pooling, CDCP token lengths
out["max_seq_length"] = enc.max_seq_length
out["tokenizer_model_max_length"] = tok.model_max_length
out["modules"] = [type(m).__name__ for m in enc]
pool = enc[1]
out["pooling"] = {k: v for k, v in vars(pool).items() if "pool" in k.lower() or "mode" in k.lower()}
out["pooling_config"] = pool.get_config_dict()

def cache_path(texts):
    h = hashlib.sha256(); h.update(b"minilm")
    for t in texts: h.update(b"\0" + t.encode())
    return TR / "cache" / f"minilm-{h.hexdigest()[:16]}.npy"

def ntok_full(texts):  # incl. [CLS]/[SEP], untruncated
    return [len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts]

S = {}
for s in ["train", "dev", "test"]:
    ex = cdcp.load(s); texts = [e.text for e in ex]
    S[s] = dict(texts=texts, y=np.array(cdcp.label_index([e.label for e in ex])), X=np.load(cache_path(texts)))
    n = np.array(ntok_full(texts))
    S[s]["ntok"] = n
    out.setdefault("over_max", {})[s] = dict(n=len(n), over=int((n > enc.max_seq_length).sum()),
                                             max_tokens=int(n.max()), p99=float(np.percentile(n, 99)),
                                             median=float(np.median(n)))

cfg = json.loads((TR / "artifacts/config.json").read_text())
model = SentenceTypeMLP(384, cfg["hidden"], 4, cfg["activation"], cfg["dropout"])
model.load_state_dict(torch.load(TR / "artifacts/model.pt", map_location="cpu")); model.eval()
proba = lambda X: model.predict_proba(torch.from_numpy(np.asarray(X, np.float32))).numpy()
emb = lambda texts: np.asarray(enc.encode(texts, batch_size=64, normalize_embeddings=True), np.float32)

# check: cached == fresh encoding for dev (same served path)
out["cache_vs_fresh_maxabs"] = float(np.abs(emb(S["dev"]["texts"]) - S["dev"]["X"]).max())

# ---------- modal position and sentence length vs P(policy) (gold policy with a modal, dev + test)
def modal_info(t):
    """content-token length (no specials) and index of first token of first modal."""
    enc_ = tok(t, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = enc_["offset_mapping"]; m = MODAL.search(t)
    idx = next(k for k, (a, b) in enumerate(offs) if b > m.start())
    return len(offs), idx, m.group(1).lower()

rows = []
for s in ["dev", "test"]:
    d = S[s]; P = proba(d["X"])
    for i, t in enumerate(d["texts"]):
        if d["y"][i] != PI or not MODAL.search(t): continue
        n, idx, mw = modal_info(t)
        rows.append(dict(split=s, i=i, text=t, len=n, pos=idx, rel=idx / n, modal=mw,
                         pp=float(P[i, PI]), pred=L[int(P[i].argmax())]))
out["B_n"] = dict(total=len(rows), dev=sum(r["split"] == "dev" for r in rows),
                  test=sum(r["split"] == "test" for r in rows))
pp = np.array([r["pp"] for r in rows]); ln = np.array([r["len"] for r in rows])
ps = np.array([r["pos"] for r in rows]); rl = np.array([r["rel"] for r in rows])
hit = np.array([r["pred"] == "policy" for r in rows])
sp = lambda a: (lambda r: dict(rho=float(r.statistic), p=float(r.pvalue)))(spearmanr(pp, a))
out["spearman"] = dict(length=sp(ln), abs_pos=sp(ps), rel_pos=sp(rl),
                       length_vs_abs_pos=dict(rho=float(spearmanr(ln, ps).statistic)),
                       length_vs_rel_pos=dict(rho=float(spearmanr(ln, rl).statistic)))
# partial Spearman: rank-residualize on length
from scipy.stats import rankdata
def partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda x: x - np.polyval(np.polyfit(rc, x, 1), rc)
    return float(np.corrcoef(res(ra), res(rb))[0, 1])
out["partial_spearman"] = dict(abs_pos_given_len=partial(pp, ps, ln), rel_pos_given_len=partial(pp, rl, ln),
                               len_given_rel_pos=partial(pp, ln, rl))

def grp(mask):
    return dict(n=int(mask.sum()), mean_pp=float(pp[mask].mean()) if mask.any() else None,
                recall=float(hit[mask].mean()) if mask.any() else None)
out["bins"] = {"<=15": grp(ln <= 15), "16-30": grp((ln > 15) & (ln <= 30)),
               "31-50": grp((ln > 30) & (ln <= 50)), ">50": grp(ln > 50)}
first = rl < 0.5
out["2x2"] = {"short_first": grp((ln <= 25) & first), "short_second": grp((ln <= 25) & ~first),
              "long_first": grp((ln > 25) & first), "long_second": grp((ln > 25) & ~first)}
out["all"] = grp(np.ones_like(hit))
out["rows"] = sorted(rows, key=lambda r: (r["split"], r["len"]))

# ---------- APPEND / PREPEND filler on short, correctly predicted dev policy sentences
F = {"F1": "and I am writing this comment in response to the proposed rule that was published by the bureau",
     "F2": "which is something that many people in my community have talked about over the past several months"}
out["filler_tokens"] = {k: len(tok(v, add_special_tokens=False)["input_ids"]) for k, v in F.items()}
d = S["dev"]; Pd = proba(d["X"])
base = [i for i, t in enumerate(d["texts"]) if d["y"][i] == PI and MODAL.search(t)
        and modal_info(t)[0] <= 15 and Pd[i].argmax() == PI]
orig = [d["texts"][i] for i in base]
def summ(P):
    return dict(mean_pp=float(P[:, PI].mean()), mean_pt=float(P[:, TI].mean()), mean_pv=float(P[:, VI].mean()),
                n_policy=int((P.argmax(1) == PI).sum()),
                dist={L[k]: int((P.argmax(1) == k).sum()) for k in range(4)})
Po = Pd[base]
C = {"n": len(base), "original": summ(Po), "orig_texts": orig}
for k, f in F.items():
    app = [t + " " + f for t in orig]; pre = [f + " " + t for t in orig]
    Pa, Pp = proba(emb(app)), proba(emb(pre))
    Pf = proba(emb([f]))[0]
    C[k] = dict(APPEND=summ(Pa), PREPEND=summ(Pp),
                filler_alone={L[j]: float(Pf[j]) for j in range(4)},
                paired_pp_diff_pre_minus_app=float((Pp[:, PI] - Pa[:, PI]).mean()),
                n_pre_lt_app=int((Pp[:, PI] < Pa[:, PI]).sum()),
                example=dict(append=app[0], prepend=pre[0], po=float(Po[0, PI]),
                             pa=float(Pa[0, PI]), pp=float(Pp[0, PI])),
                mean_len_orig=float(np.mean([modal_info(t)[0] for t in orig])),
                mean_rel_pos=dict(orig=float(np.mean([modal_info(t)[1] / modal_info(t)[0] for t in orig])),
                                  append=float(np.mean([modal_info(t)[1] / modal_info(t)[0] for t in app])),
                                  prepend=float(np.mean([modal_info(t)[1] / modal_info(t)[0] for t in pre]))))
out["C"] = C

json.dump(out, open(REPO / "training/results/hw3_diag" / "out2.json", "w"), indent=1, default=str)
o = dict(out); o["rows"] = f"{len(rows)} rows (see out2.json)"
print(json.dumps(o, indent=1, default=str))
