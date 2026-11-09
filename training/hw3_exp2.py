"""HW3 experiment 2 runner: cue-preserving pooling (see cue_pool.py).

    python hw3_exp2.py occlusion   FT-all seed 0, dev only: delete a cue phrase, compare probabilities
    python hw3_exp2.py check       (a) the copied loop with pool="mean" reproduces FT-all seed 0, epochs 0-2
                                   (b) start state of each cue variant: epoch-0 dev loss, max |z - m|
    python hw3_exp2.py run         CUE-GATE, CUE-ONLY, CUE-LIST x seeds 0, 1, 2 (test scored once per run),
                                   then `analyze`. Runs with saved metrics are skipped.
    python hw3_exp2.py analyze     tables, near-tie analysis, interpretability, figures, report.md

Writes training/results/hw3_exp2/ and weights to training/runs/hw3_exp2/ (gitignored).
Reads training/runs/hw3/ft_all_seed*/ and training/results/hw3/ without changing them.
Nothing is selected on test: every variant x seed is a final run with FT-all's hyperparameters.
"""

import collections
import json
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import torch

import cdcp
import cue_pool as cp
import finetune as ft

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results" / "hw3_exp2"
RUNS = HERE / "runs" / "hw3_exp2"
HW3 = HERE / "results" / "hw3"
HW3_RUNS = HERE / "runs" / "hw3"
SEEDS = [0, 1, 2]
VARIANTS = {"cue_gate": "CUE-GATE", "cue_only": "CUE-ONLY", "cue_list": "CUE-LIST"}
FT_CFG = json.loads((HW3_RUNS / "ft_all_seed0" / "config.json").read_text())  # the control's recipe
L = cdcp.LABELS
COLORS = {"fact": "#2a78b5", "policy": "#d1492e", "testimony": "#3a9a5b", "value": "#8a8a8a"}
LN4 = float(np.log(4))


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a") as f:
        f.write(line + "\n")


def jdump(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def jload(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def dev_ids(ex) -> list[str]:
    ids, k, prev = [], 0, None
    for e in ex:
        k = k + 1 if e.comment_id == prev else 0
        prev = e.comment_id
        ids.append(f"{e.comment_id}.{k}")
    return ids


def load_ft_all(seed: int) -> ft.FineTuneModel:
    cfg = json.loads((HW3_RUNS / f"ft_all_seed{seed}" / "config.json").read_text())
    m = ft.FineTuneModel(cfg)
    m.load_state_dict(torch.load(HW3_RUNS / f"ft_all_seed{seed}" / "model.pt", map_location="cpu",
                                 weights_only=True))
    return m.eval()


def merged_word(tokens: list[str], i: int) -> str:
    """Word piece i -> the whole word it belongs to ("exp", "##erience" -> "experience")."""
    s, e = i, i + 1
    while s > 0 and tokens[s].startswith("##"):
        s -= 1
    while e < len(tokens) and tokens[e].startswith("##"):
        e += 1
    return tokens[s] + "".join(t[2:] for t in tokens[s + 1:e])


# ---------------------------------------------------------------- occlusion (report §1)

OCCLUSION = [  # (dev id, phrase to delete, note on the result)
    ("dev-32.2", "It ' s important to understand that", "grammatical; now starts with lower-case 'a'"),
    ("dev-10.0", "In my experience ,", "grammatical"),
    ("dev-71.3", "My experience is", "grammatical; reads as a general claim instead of a report"),
    ("dev-39.6", "be required to", "grammatical fragment (it was already one); requirement becomes an action"),
    ("dev-47.2", "should not", "broken grammar ('They be allowed'); the negation is gone too"),
    ("dev-8.9", "should", "broken grammar ('Judges be able')"),
    ("dev-10.4", "should", "broken grammar ('the letter say so')"),
    ("dev-41.2", "should", "broken grammar ('Starting the process be simple')"),
]


def delete_phrase(text: str, phrase: str) -> str:
    toks, ph = text.split(" "), phrase.split(" ")
    for i in range(len(toks) - len(ph) + 1):
        if toks[i:i + len(ph)] == ph:
            return " ".join(toks[:i] + toks[i + len(ph):])
    raise ValueError(f"{phrase!r} not found in {text!r}")


def occlusion() -> None:
    ex = cdcp.load("dev")
    by_id = dict(zip(dev_ids(ex), ex))
    model = load_ft_all(0)
    rows = []
    for sid, phrase, note in OCCLUSION:
        e = by_id[sid]
        cut = delete_phrase(e.text, phrase)
        P = model.predict_proba([e.text, cut])
        rows.append({"id": sid, "gold": e.label, "phrase": phrase, "note": note,
                     "text": e.text, "text_after": cut,
                     "before": dict(zip(L, map(float, P[0]))), "after": dict(zip(L, map(float, P[1]))),
                     "pred_before": L[int(P[0].argmax())], "pred_after": L[int(P[1].argmax())]})
    jdump({"model": "ft_all_seed0", "split": "dev", "rows": rows}, OUT / "occlusion.json")
    print(occlusion_md(rows))


def occlusion_md(rows: list[dict]) -> str:
    out = ["| id | gold | deleted | | " + " | ".join(f"P({l})" for l in L) + " | pred |",
           "|---|---|---|---|" + "---|" * 5]
    for r in rows:
        for when in ("before", "after"):
            p = r[when]
            lab = f"`{r['phrase']}`" if when == "before" else ""
            out.append(f"| {r['id'] if when == 'before' else ''} | {r['gold'] if when == 'before' else ''} | "
                       f"{lab} | {when} | " + " | ".join(f"{p[l]:.3f}" for l in L)
                       + f" | {r['pred_' + when]} |")
    out.append("")
    # Ids only: CDCP has no redistribution license, so the report never quotes a sentence.
    # The edited text stays in occlusion.json (gitignored).
    out.append("Notes on the edited sentences (text in occlusion.json, not committed):")
    for r in rows:
        out.append(f"- **{r['id']}**: {r['note']}.")
    return "\n".join(out)


# ---------------------------------------------------------------- checks (report §0)


def check() -> bool:
    data = ft.load_splits()
    res = {}

    # (a) the copied loop, pool = "mean", must equal FT-all seed 0 for epochs 0-2.
    t0 = time.time()
    r = cp.train_cue({**FT_CFG, "pool": "mean", "seed": 0, "max_epochs": 2}, data, log=log)
    ref = jload(HW3 / "ft_all_seed0" / "history.json")
    keys = ["train_loss", "train_error", "dev_loss", "dev_error", "train_loss_running"]
    diffs = {k: max(abs((a[k] or 0) - (b[k] or 0)) for a, b in zip(r["history"], ref[:3])) for k in keys}
    res["equivalence"] = {"epochs_compared": [0, 1, 2], "max_abs_diff": diffs,
                          "pass": max(diffs.values()) < 1e-4, "seconds": round(time.time() - t0, 1),
                          "ours": [{k: h[k] for k in ["epoch"] + keys} for h in r["history"]],
                          "ft_all": [{k: h[k] for k in ["epoch"] + keys} for h in ref[:3]]}
    log(f"equivalence: max |diff| {max(diffs.values()):.2e} -> pass {res['equivalence']['pass']}")

    # (b) start state of each cue variant, seed 0, dev.
    loss_fn = torch.nn.CrossEntropyLoss()
    res["start_state"] = {}
    for pool in VARIANTS:
        torch.manual_seed(0)
        m = cp.CueModel({**FT_CFG, "pool": pool}).eval()
        d = ft.evaluate(m, data["dev"], loss_fn, False)
        det = m.details(data["dev"]["texts"])
        Z = np.stack([x["z"] for x in det])
        M = np.stack([x["m_norm"] for x in det])
        diff = np.abs(Z - M).max(1)
        cos = (Z * M).sum(1)
        g = np.array([x["g"] for x in det])
        matched = np.array([x["cue_mask"].any() for x in det])
        res["start_state"][pool] = {
            "epoch0_dev_loss": d["loss"], "ft_all_epoch0_dev_loss": ref[0]["dev_loss"],
            "max_abs_z_minus_m": float(diff.max()), "mean_of_per_sentence_max_abs": float(diff.mean()),
            "min_cosine_z_m": float(cos.min()), "mean_cosine_z_m": float(cos.mean()),
            "g_at_init": [float(g.min()), float(g.max())],
            "share_dev_with_cue_token": float(matched.mean()),
            "note": "z and m both L2-normalized (head inputs); seed 0, pre-trained encoder, untrained pooling",
        }
        log(f"start {pool}: epoch-0 dev loss {d['loss']:.4f} (FT-all {ref[0]['dev_loss']:.4f}, ln4 {LN4:.4f}); "
            f"max|z-m| {diff.max():.4f}; min cos {cos.min():.4f}")
    jdump(res, OUT / "check.json")
    return res["equivalence"]["pass"]


# ---------------------------------------------------------------- run


def save_run(name: str, r: dict, data: dict) -> None:
    model = r.pop("model")
    w = RUNS / name
    w.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), w / "model.pt")
    jdump(r["config"], w / "config.json")
    o = OUT / name
    jdump(r["config"], o / "config.json")
    jdump(r["history"], o / "history.json")
    jdump({k: v for k, v in r.items() if k not in ("config", "history")}, o / "metrics.json")
    np.save(o / "dev_proba.npy", model.predict_proba(data["dev"]["texts"]))
    np.save(o / "test_proba.npy", model.predict_proba(data["test"]["texts"]))
    if r["config"]["seed"] == 0:  # interpretability + map (analysis only, chooses nothing)
        det = model.details(data["test"]["texts"])
        np.save(o / "test_z.npy", np.stack([x["z"] for x in det]))
        rows = []
        for x in det:
            idx = [i for i in np.argsort(-x["a"]) if x["cue_mask"][i]][:3]
            rows.append({"g": x["g"], "top_word": merged_word(x["tokens"], idx[0]) if idx else None,
                         "top3": [[merged_word(x["tokens"], i), float(x["a"][i])] for i in idx],
                         "n_cue_tokens": int(x["cue_mask"].sum())})
        jdump(rows, o / "test_detail.json")


def ft_all_reference(data: dict) -> None:
    o = OUT / "ft_all_ref"
    if (o / "test_z_seed0.npy").exists():
        return
    o.mkdir(parents=True, exist_ok=True)
    ok = {}
    for s in SEEDS:
        m = load_ft_all(s)
        P = m.predict_proba(data["dev"]["texts"])
        np.save(o / f"dev_proba_seed{s}.npy", P)
        err = float((P.argmax(1) != data["dev"]["y"].numpy()).mean())
        ok[s] = {"dev_error_recomputed": err, "dev_error_saved": jload(HW3 / f"ft_all_seed{s}" / "metrics.json")["dev"]["error"]}
        if s == 0:
            np.save(o / "test_z_seed0.npy", m.embed(data["test"]["texts"]).numpy())
    jdump(ok, o / "check.json")
    log(f"FT-all reference recomputed: {ok}")


def run() -> None:
    data = ft.load_splits()
    for pool in VARIANTS:
        for s in SEEDS:
            name = f"{pool}_seed{s}"
            if (OUT / name / "metrics.json").exists():
                log(f"skip {name} (done)")
                continue
            log(f"=== {name}")
            t0 = time.time()
            r = cp.train_cue({**FT_CFG, "pool": pool, "seed": s}, data, evaluate_test=True, log=log)
            r["minutes"] = round((time.time() - t0) / 60, 1)
            save_run(name, r, data)
            log(f"=== {name} done in {r['minutes']} min: best epoch {r['best_epoch']}, "
                f"dev loss {r['dev']['loss']:.4f}, test error {r['test']['error']:.3f}")
    ft_all_reference(data)
    analyze(data)


# ---------------------------------------------------------------- analyze


def fmt(vals, d=3) -> str:
    v = np.array(vals, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def test_sets() -> dict:
    hw2 = jload(HW3 / "hw2_baseline.json")
    out = {"HW2": [s["test"] for s in hw2["seeds"]],
           "FT-all": [jload(HW3 / f"ft_all_seed{s}" / "metrics.json")["test"] for s in SEEDS]}
    for pool, name in VARIANTS.items():
        ms = [jload(OUT / f"{pool}_seed{s}" / "metrics.json") for s in SEEDS]
        if all(ms):
            out[name] = [m["test"] for m in ms]
    return out


def test_table(T: dict) -> str:
    cols = list(T)
    lines = ["| metric | " + " | ".join(f"{c} (n={len(T[c])})" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    lines.append("| **test error** | " + " | ".join(fmt([m["error"] for m in T[c]]) for c in cols) + " |")
    lines.append("| **macro-F1** | " + " | ".join(fmt([m["macro_f1"] for m in T[c]]) for c in cols) + " |")
    for l in L:
        for k in ("precision", "recall", "f1"):
            b = "**" if k == "f1" else ""
            lines.append(f"| {b}{l} {k[0].upper() if k != 'f1' else 'F1'}{b} | "
                         + " | ".join(fmt([m["per_class"][l][k] for m in T[c]]) for c in cols) + " |")
    return "\n".join(lines)


def cm_md(cm: list[list[int]]) -> str:
    cm = np.array(cm)
    out = ["| gold \\ pred | " + " | ".join(L) + " | recall |", "|---|" + "---|" * 5]
    for i, l in enumerate(L):
        out.append(f"| {l} | " + " | ".join(str(v) for v in cm[i]) + f" | {cm[i, i] / cm[i].sum():.2f} |")
    return "\n".join(out)


def near_ties(data: dict) -> dict:
    y = data["dev"]["y"].numpy()
    res = {}
    for pool, name in VARIANTS.items():
        if not (OUT / f"{pool}_seed0" / "dev_proba.npy").exists():
            continue
        per = {"near_tie": {}, "all_dev": {}}
        for scope in per:
            tot = {"n_scope": 0, "ft_all_errors": 0, "fixed": collections.Counter(),
                   "broken": collections.Counter(), "n_by_gold": collections.Counter(), "per_seed": []}
            for s in SEEDS:
                P0 = np.load(OUT / "ft_all_ref" / f"dev_proba_seed{s}.npy")
                P1 = np.load(OUT / f"{pool}_seed{s}" / "dev_proba.npy")
                srt = np.sort(P0, 1)
                sel = (srt[:, -1] - srt[:, -2] < 0.1) if scope == "near_tie" else np.ones(len(y), bool)
                p0, p1 = P0.argmax(1), P1.argmax(1)
                fixed = sel & (p0 != y) & (p1 == y)
                broken = sel & (p0 == y) & (p1 != y)
                tot["n_scope"] += int(sel.sum())
                tot["ft_all_errors"] += int((sel & (p0 != y)).sum())
                for i in np.where(fixed)[0]:
                    tot["fixed"][L[y[i]]] += 1
                for i in np.where(broken)[0]:
                    tot["broken"][L[y[i]]] += 1
                for i in np.where(sel)[0]:
                    tot["n_by_gold"][L[y[i]]] += 1
                tot["per_seed"].append({"seed": s, "n": int(sel.sum()), "ft_all_errors": int((sel & (p0 != y)).sum()),
                                        "fixed": int(fixed.sum()), "broken": int(broken.sum())})
            per[scope] = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in tot.items()}
        res[name] = per
    return res


def near_tie_md(NT: dict) -> str:
    out = []
    for scope, title in (("near_tie", "Near ties (FT-all top-2 margin < 0.1)"), ("all_dev", "All of dev (baseline)")):
        out.append(f"**{title}**, summed over seeds 0–2 (each variant seed vs FT-all same seed)\n")
        out.append("| variant | sentences | FT-all errors | fixed | broken | net | "
                   + " | ".join(f"{l} fixed/broken/n" for l in L) + " |")
        out.append("|---|---|---|---|---|---|" + "---|" * 4)
        for name, per in NT.items():
            t = per[scope]
            f, b = sum(t["fixed"].values()), sum(t["broken"].values())
            out.append(f"| {name} | {t['n_scope']} | {t['ft_all_errors']} | {f} | {b} | {f - b:+d} | "
                       + " | ".join(f"{t['fixed'].get(l, 0)}/{t['broken'].get(l, 0)}/{t['n_by_gold'].get(l, 0)}"
                                    for l in L) + " |")
        out.append("")
        out.append("Per seed (n / FT-all errors / fixed / broken): " + "; ".join(
            f"{name} " + ", ".join(f"s{p['seed']} {p['n']}/{p['ft_all_errors']}/{p['fixed']}/{p['broken']}"
                                   for p in per[scope]["per_seed"]) for name, per in NT.items()))
        out.append("")
    return "\n".join(out)


LIST_WORDS = {w.lower() for e in cp.read_wordlist() for w in e.split() if w.lower() != "to"}


def interpretability(data: dict) -> dict:
    y = data["test"]["y"].numpy()
    res = {}
    for pool, name in VARIANTS.items():
        rows = jload(OUT / f"{pool}_seed0" / "test_detail.json")
        if rows is None:
            continue
        r = {"top_words": {}, "gate": {}, "overlap": {}, "share_with_cue_token": {}}
        for ci, l in enumerate(L):
            idx = np.where(y == ci)[0]
            words = [rows[i]["top_word"] for i in idx if rows[i]["top_word"]]
            top = collections.Counter(words).most_common(15)
            r["top_words"][l] = [[w, c, round(c / len(idx), 3)] for w, c in top]
            r["overlap"][l] = [w for w, _ in top if w in LIST_WORDS]
            g = np.array([rows[i]["g"] for i in idx])
            r["gate"][l] = {"n": int(len(idx)), "mean": float(g.mean()), "median": float(np.median(g)),
                            "q25": float(np.quantile(g, .25)), "q75": float(np.quantile(g, .75)),
                            "share_above_0.5": float((g > 0.5).mean())}
            r["share_with_cue_token"][l] = float(np.mean([rows[i]["n_cue_tokens"] > 0 for i in idx]))
        res[name] = r
    return res


def interp_md(I: dict) -> str:
    out = [f"List words used for overlap: {', '.join(sorted(LIST_WORDS))} (\"to\" from \"need to\" left out).\n"]
    for name, r in I.items():
        out.append(f"**{name}** (seed 0, test) — word most often given the highest a_t, per gold class "
                   "(count, share of that class)\n")
        out.append("| rank | " + " | ".join(L) + " |")
        out.append("|---|" + "---|" * 4)
        for k in range(15):
            cells = []
            for l in L:
                t = r["top_words"][l]
                if k < len(t):
                    w, c, sh = t[k]
                    cells.append(f"{'**' + w + '**' if w in LIST_WORDS else w} ({c}, {sh:.0%})")
                else:
                    cells.append("")
            out.append(f"| {k + 1} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Overlap with the word list (bold above): " + "; ".join(
            f"{l} {len(r['overlap'][l])}/15 ({', '.join(r['overlap'][l]) or '—'})" for l in L))
        if name == "CUE-LIST":
            out.append("\nShare of test sentences with ≥1 list word: " + ", ".join(
                f"{l} {r['share_with_cue_token'][l]:.0%}" for l in L) + ". Sentences without one use z = m.")
        if name != "CUE-ONLY":
            out.append("\nGate g by gold class:\n")
            out.append("| gold | n | mean | median | q25–q75 | share g > 0.5 |")
            out.append("|---|---|---|---|---|---|")
            for l in L:
                g = r["gate"][l]
                out.append(f"| {l} | {g['n']} | {g['mean']:.3f} | {g['median']:.3f} | "
                           f"{g['q25']:.3f}–{g['q75']:.3f} | {g['share_above_0.5']:.0%} |")
        else:
            out.append("\n(g is fixed to 1 in CUE-ONLY.)")
        out.append("")
    return "\n".join(out)


def plots(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # training curves
    ref = jload(HW3 / "ft_all_seed0" / "history.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (pool, name) in zip(axes, VARIANTS.items()):
        for s in SEEDS:
            h = jload(OUT / f"{pool}_seed{s}" / "history.json")
            if not h:
                continue
            e = [r["epoch"] for r in h]
            lw = 2.2 if s == 0 else 0.9
            ax.plot(e, [r["train_loss"] for r in h], color="#2a78b5", lw=lw, label="train" if s == 0 else None)
            ax.plot(e, [r["dev_loss"] for r in h], color="#d1492e", lw=lw, label="dev" if s == 0 else None)
        ax.plot([r["epoch"] for r in ref], [r["dev_loss"] for r in ref], color="#555", ls=":", lw=1.5,
                label="FT-all dev (s0)")
        ax.axhline(LN4, color="#bbb", lw=0.8, ls="--")
        ax.set_title(f"{name} (seed 0 bold)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=130)
    plt.close(fig)

    # gate histograms
    y = data["test"]["y"].numpy()
    gated = [(p, n) for p, n in VARIANTS.items() if p != "cue_only" and (OUT / f"{p}_seed0" / "test_detail.json").exists()]
    if gated:
        fig, axes = plt.subplots(1, len(gated), figsize=(6 * len(gated), 3.8), squeeze=False)
        for ax, (pool, name) in zip(axes[0], gated):
            rows = jload(OUT / f"{pool}_seed0" / "test_detail.json")
            g = np.array([r["g"] for r in rows])
            for ci, l in enumerate(L):
                ax.hist(g[y == ci], bins=np.linspace(0, 1, 41), histtype="step", lw=1.6, color=COLORS[l],
                        density=True, label=l)
            ax.set_title(f"{name}: gate g on test (seed 0)")
            ax.set_xlabel("g  (0 = mean m, 1 = cue c)")
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "gate_by_class.png", dpi=130)
        plt.close(fig)

    # map
    if (OUT / "cue_gate_seed0" / "test_z.npy").exists():
        from sklearn.manifold import TSNE

        vecs = {"HW2 (frozen MiniLM)": data["test"]["X"].numpy(),
                "FT-all (seed 0)": np.load(OUT / "ft_all_ref" / "test_z_seed0.npy"),
                "CUE-GATE (seed 0)": np.load(OUT / "cue_gate_seed0" / "test_z.npy")}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        for ax, (title, X) in zip(axes, vecs.items()):
            Y = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                     random_state=0).fit_transform(X)
            for ci, l in enumerate(L):
                ax.scatter(Y[y == ci, 0], Y[y == ci, 1], s=7, alpha=0.75, color=COLORS[l], label=l, lw=0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].legend(frameon=False, markerscale=2.5)
        fig.suptitle("CDCP test (836 sentences), t-SNE of the head input, perplexity 30, seed 0 — "
                     "each panel fitted separately; compare class separation, not positions")
        fig.tight_layout()
        fig.savefig(OUT / "map_tsne.png", dpi=130)
        plt.close(fig)


def analyze(data: dict | None = None) -> None:
    data = data or ft.load_splits()
    T = test_sets()
    NT = near_ties(data)
    I = interpretability(data)
    jdump({"test": {k: v for k, v in T.items()}, "near_ties": NT, "interpretability": I}, OUT / "summary.json")
    plots(data)

    chk = jload(OUT / "check.json")
    occ = jload(OUT / "occlusion.json")
    refc = jload(OUT / "ft_all_ref" / "check.json")
    md = ["# HW3 experiment 2 — cue-preserving pooling with a learned gate",
          f"_Built {datetime.now():%Y-%m-%d %H:%M}. Control = FT-all (results/hw3), same recipe; only pooling "
          "changes. Dev used only for early stopping; nothing chosen on test; test scored once per run. "
          "± = std over seeds 0–2 (ddof 1). Interpretations are untested._", ""]

    md += ["## 0. Checks and start state", ""]
    if chk:
        eq = chk["equivalence"]
        md.append(f"- **Loop equivalence**: `train_cue` with `pool=\"mean\"`, seed 0, reproduces FT-all seed 0 for "
                  f"epochs 0–2: max |diff| over train/dev loss, train/dev error and running train loss = "
                  f"{max(eq['max_abs_diff'].values()):.1e} → **{'pass' if eq['pass'] else 'FAIL'}**.")
    if refc:
        md.append("- FT-all dev error recomputed from saved weights: " + ", ".join(
            f"s{s} {v['dev_error_recomputed']:.4f} (saved {v['dev_error_saved']:.4f})" for s, v in refc.items()))
    md += ["", "**Start state.** w = 0 makes the attention a_t uniform over word tokens, so at the start "
           "c = mean of word tokens *without* [CLS]/[SEP], whereas m includes them. The gate starts at "
           "g = sigmoid(0) = 0.5 (CUE-GATE, CUE-LIST), so z = (m + c)/2; in CUE-ONLY g = 1, so z = c. "
           "In CUE-LIST c = mean of list-word tokens (c = m when a sentence has none). So z is close to, "
           "but not exactly, FT-all's m at epoch 0. The head's initial weights, dropout masks and batch "
           "order are identical to FT-all's for the same seed (random state saved/restored around the new "
           "layers). The design was not changed to force equality.", ""]
    if chk:
        md.append("| variant (seed 0) | epoch-0 dev loss | FT-all epoch-0 | ln 4 | max \\|z − m\\| | min cos(z, m) | "
                  "dev sentences with a cue token |")
        md.append("|---|---|---|---|---|---|---|")
        for pool, name in VARIANTS.items():
            st = chk["start_state"][pool]
            md.append(f"| {name} | {st['epoch0_dev_loss']:.4f} | {st['ft_all_epoch0_dev_loss']:.4f} | {LN4:.4f} | "
                      f"{st['max_abs_z_minus_m']:.4f} | {st['min_cosine_z_m']:.4f} | {st['share_dev_with_cue_token']:.0%} |")
        md.append("\n_z and m are both L2-normalized (as the head sees them); max over dev sentences and "
                  "dimensions._\n")
    md.append("Epoch-0 dev loss by seed (from each run's history):\n")
    md.append("| seed | FT-all | " + " | ".join(VARIANTS.values()) + " |")
    md.append("|---|---|" + "---|" * len(VARIANTS))
    for s in SEEDS:
        fa = jload(HW3 / f"ft_all_seed{s}" / "history.json")
        cells = []
        for pool in VARIANTS:
            h = jload(OUT / f"{pool}_seed{s}" / "history.json")
            cells.append(f"{h[0]['dev_loss']:.4f}" if h else "—")
        md.append(f"| {s} | {fa[0]['dev_loss']:.4f} | " + " | ".join(cells) + " |")
    md.append("")

    if occ:
        md += ["## 1. Occlusion check (FT-all seed 0, dev, before any training)", "",
               "Delete the cue phrase; all four probabilities before and after.", "", occlusion_md(occ["rows"]), ""]

    md += ["## 2. Test (CDCP, 836 sentences)", "", test_table(T), "",
           "HW2 and FT-all are read from `results/hw3/` (not re-run).", ""]
    md += ["### Confusion matrices, seed 0 (rows = gold)", ""]
    for c, ms in T.items():
        md += [f"**{c}**", "", cm_md(ms[0]["confusion_matrix"]["rows_true_cols_pred"]), ""]

    md += ["## 3. Near-tie analysis (dev)", "", near_tie_md(NT)]
    md += ["## 4. Interpretability (seed 0, test — analysis only)", "", interp_md(I),
           "![gate](gate_by_class.png)", ""]

    md += ["## 5. Training", "", "![loss curves](loss_curves.png)", "",
           "| run | trainable params | best epoch | last epoch | stop | train s/epoch | dev loss | dev error | min |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        m = jload(HW3 / f"ft_all_seed{s}" / "metrics.json")
        md.append(f"| ft_all_seed{s} (HW3) | {m['n_parameters_trainable']:,} | {m['best_epoch']} | {m['last_epoch']} | "
                  f"{m['stop_reason']} | — | {m['dev']['loss']:.4f} | {m['dev']['error']:.3f} | — |")
    for pool in VARIANTS:
        for s in SEEDS:
            m = jload(OUT / f"{pool}_seed{s}" / "metrics.json")
            if not m:
                continue
            h = jload(OUT / f"{pool}_seed{s}" / "history.json")
            sec = np.mean([r["train_seconds"] for r in h[1:]])
            md.append(f"| {pool}_seed{s} | {m['n_parameters_trainable']:,} | {m['best_epoch']} | {m['last_epoch']} | "
                      f"{m['stop_reason']} | {sec:.0f} | {m['dev']['loss']:.4f} | {m['dev']['error']:.3f} | {m['minutes']} |")
    md += ["", "## 6. Map", "", "![map](map_tsne.png)", "",
           "t-SNE (perplexity 30, PCA init, random_state 0) of the 384-d head input: HW2 = frozen MiniLM "
           "embedding (cache), FT-all and CUE-GATE = fine-tuned z. Each panel is fitted separately, so axes "
           "and positions are not comparable across panels.", "",
           # The interpretation is hand-written in interpretation.md. This script never writes that
           # file, so re-running `analyze` (or `run`) cannot overwrite it.
           "## 7. Interpretation (untested)", "",
           "Hand-written, in [`interpretation.md`](interpretation.md) (not regenerated by this script).", ""]
    (OUT / "report.md").write_text("\n".join(md))
    log("report written: results/hw3_exp2/report.md")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "occlusion":
        occlusion()
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "run":
        run()
    elif cmd == "analyze":
        analyze()
    else:
        print(__doc__)
        sys.exit(2)
