"""HW3 overnight run: fine-tuning MiniLM vs a frozen-encoder control.

    caffeinate -i training/.venv/bin/python -u training/hw3_overnight.py     (caffeinate: macOS)

Steps, in priority order. Each step saves its results before the next starts;
a step that crashes is logged and skipped, and the run continues.

    0    HW2 baseline: re-train the 3 HW2 seeds in memory (nothing written to
         HW2 files), check them against sweep.json; Part C + probes on the saved
         HW2 model; check that finetune.SentenceEncoder reproduces the cache
    1    lr grid: FT-all, seed 0, 4 epochs per encoder lr; pick the lr with the
         lowest dev loss (rule fixed before the run; test never looked at)
    1.5  Frozen-new control: encoder frozen (HW2 cache), head trained with the
         FT recipe, seeds 0-2; Part C + probes on seed 0
    2    FT-all seed 0 + test + Part C + probes
    3    FT-all seeds 1, 2
    4    FT-top2 seed 0 (+ Part C, probes), then seeds 1, 2 (lr from step 1)
    5    RFI rates for the FT-all seed with the lowest dev loss, then the same for FT-top2

Writes training/results/hw3/ (JSON, plots, summary.md, run.log) and weights to
training/runs/hw3/ (gitignored). Never writes to training/artifacts/ or the cache.

HW3_SMOKE=1 runs every step on tiny data, with HW3_OUT / HW3_RUNS pointing
elsewhere, to catch crashes before the real run.
"""

import json
import os
import pathlib
import re
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cdcp  # noqa: E402
import finetune as ft  # noqa: E402
from model import SentenceTypeMLP  # noqa: E402

SMOKE = os.environ.get("HW3_SMOKE") == "1"
OUT = pathlib.Path(os.environ.get("HW3_OUT", HERE / "results" / "hw3"))
RUNS = pathlib.Path(os.environ.get("HW3_RUNS", HERE / "runs" / "hw3"))
OUT.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

LR_GRID = [2e-5, 5e-5] if SMOKE else [1e-5, 2e-5, 3e-5, 5e-5, 1e-4]
GRID_EPOCHS = 1 if SMOKE else 4
FULL = {"max_epochs": 1} if SMOKE else {}
SEEDS = [0, 1] if SMOKE else [0, 1, 2]
LIMIT = 48 if SMOKE else None
RFI_LIMIT = 300 if SMOKE else None
FALLBACK_LR = 2e-5

L = cdcp.LABELS
PI, TI, VI = L.index("policy"), L.index("testimony"), L.index("value")
MODAL = re.compile(r"\b(should|must|need to|has to)\b", re.I)
FILLERS = {
    "F1": "and I am writing this comment in response to the proposed rule that was published by the bureau",
    "F2": "which is something that many people in my community have talked about over the past several months",
}
PROBES = ["The agency should require collectors to verify the debt.",
          "The agency can require collectors to verify the debt.",
          "The agency requires collectors to verify the debt."]
MODELS = [("hw2", "HW2 baseline"), ("frozen", "Frozen-new"), ("ft_top2", "FT-top2"), ("ft_all", "FT-all")]


# ---------------------------------------------------------------- plumbing


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def jdump(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def jload(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


STATUS_PATH = OUT / "status.json"
STATUS = jload(STATUS_PATH) or {}

_data = None
def data() -> dict:
    global _data
    if _data is None:
        _data = ft.load_splits(LIMIT)
    return _data


def run_step(name: str, fn) -> None:
    log(f"=== step {name} start")
    t0 = time.time()
    STATUS[name] = {"state": "running", "started": datetime.now().isoformat(timespec="seconds")}
    jdump(STATUS, STATUS_PATH)
    try:
        note = fn()
        STATUS[name] = {"state": "done", "minutes": round((time.time() - t0) / 60, 1), "note": note}
        log(f"=== step {name} done in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:  # log and move on
        log(f"!!! step {name} FAILED: {e!r}\n{traceback.format_exc()}")
        STATUS[name] = {"state": "failed", "error": repr(e), "minutes": round((time.time() - t0) / 60, 1)}
    jdump(STATUS, STATUS_PATH)
    try:
        build_summary()
    except Exception as e:
        log(f"!!! summary failed: {e!r}\n{traceback.format_exc()}")


# ---------------------------------------------------------------- analyses


def content_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def base_set(tok, probs_dev: np.ndarray) -> list[int]:
    """Gold policy, <= 15 tokens, contains a modal, predicted policy (the Part C base set)."""
    dv = data()["dev"]
    return [i for i, t in enumerate(dv["texts"])
            if int(dv["y"][i]) == PI and MODAL.search(t) and content_tokens(tok, t) <= 15
            and probs_dev[i].argmax() == PI]


def summarize_probs(P: np.ndarray) -> dict:
    a = P.argmax(1)
    return {"n": len(P), "mean_p_policy": float(P[:, PI].mean()), "mean_p_testimony": float(P[:, TI].mean()),
            "mean_p_value": float(P[:, VI].mean()), "n_policy": int((a == PI).sum()),
            "dist": {L[k]: int((a == k).sum()) for k in range(4)}}


def part_c(proba, tok, base_sets: dict) -> dict:
    """APPEND / PREPEND with each filler, on each base set. `proba(texts)` -> (n, 4)."""
    dv = data()["dev"]
    out = {}
    for set_name, idx in base_sets.items():
        orig = [dv["texts"][i] for i in idx]
        if not orig:
            out[set_name] = {"n": 0}
            continue
        Po = proba(orig)
        res = {"n": len(orig), "indices": idx, "texts": orig, "original": summarize_probs(Po)}
        for fname, f in FILLERS.items():
            Pa, Pp = proba([t + " " + f for t in orig]), proba([f + " " + t for t in orig])
            res[fname] = {"APPEND": summarize_probs(Pa), "PREPEND": summarize_probs(Pp),
                          "n_prepend_below_append": int((Pp[:, PI] < Pa[:, PI]).sum()),
                          "filler_alone": dict(zip(L, map(float, proba([f])[0])))}
        res["per_sentence_p_policy"] = {"original": Po[:, PI].tolist()}
        out[set_name] = res
    return out


def probes(proba) -> list[dict]:
    P = proba(PROBES)
    return [{"text": t, **{L[k]: float(P[j, k]) for k in range(4)}} for j, t in enumerate(PROBES)]


_hw2_head = None
def hw2_head() -> SentenceTypeMLP:
    global _hw2_head
    if _hw2_head is None:
        cfg = json.loads((HERE / "artifacts" / "config.json").read_text())
        m = SentenceTypeMLP(384, cfg["hidden"], 4, cfg["activation"], cfg["dropout"])
        m.load_state_dict(torch.load(HERE / "artifacts" / "model.pt", map_location="cpu", weights_only=True))
        m.eval()
        _hw2_head = m
    return _hw2_head


_hw2_set = None
def hw2_base_set(tok) -> list[int]:
    """Part C base set A: defined by the saved HW2 model (= the filler base set in hw3/diagnostics/diag2.py)."""
    global _hw2_set
    if _hw2_set is None:
        P = hw2_head().predict_proba(data()["dev"]["X"]).numpy()
        _hw2_set = base_set(tok, P)
    return _hw2_set


def analyses_for(name: str, model: ft.FineTuneModel) -> None:
    """Part C (HW2 base set + the model's own) and the 3 probe sentences."""
    tok = model.encoder.tokenizer
    proba = lambda texts: model.predict_proba(texts)
    own = base_set(tok, proba(data()["dev"]["texts"]))
    jdump(part_c(proba, tok, {"hw2_set": hw2_base_set(tok), "own_set": own}), OUT / f"partC_{name}.json")
    jdump(probes(proba), OUT / f"probes_{name}.json")
    log(f"  Part C + probes saved for {name}")


def save_run(name: str, res: dict) -> None:
    from train import plot_curves

    d = OUT / name
    cfg = res["config"]
    meta = {k: res[k] for k in ["optimizer", "n_parameters_total", "n_parameters_trainable",
                                "best_epoch", "last_epoch", "stop_reason", "under_trained"]}
    jdump({**cfg, **meta}, d / "config.json")
    jdump(res["history"], d / "history.json")
    jdump({**meta, "train": res["train"], "dev": res["dev"], "test": res.get("test")}, d / "metrics.json")
    plot_curves(res["history"], res["best_epoch"], d / "loss_curve.png", weighted=False)
    w = RUNS / name
    w.mkdir(parents=True, exist_ok=True)
    model = res["model"]
    if cfg["mode"] == "frozen":
        torch.save(model.head.state_dict(), w / "head.pt")   # the encoder is the unchanged pre-trained one
    else:
        torch.save(model.state_dict(), w / "model.pt")
    jdump(cfg, w / "config.json")
    t = res.get("test") or {}
    log(f"  saved {name}: best epoch {res['best_epoch']} ({res['stop_reason']}), dev loss {res['dev']['loss']:.4f} "
        f"err {res['dev']['error']:.3f}; test err {t.get('error', float('nan')):.3f} macro-F1 {t.get('macro_f1', float('nan')):.3f}")


def full_run(mode: str, seed: int, lr: float, with_analyses: bool) -> None:
    name = f"{mode}_seed{seed}"
    res = ft.finetune({"mode": mode, "seed": seed, "encoder_lr": lr, **FULL}, data(),
                      evaluate_test=True, log=log)
    save_run(name, res)
    if with_analyses:
        analyses_for(name, res["model"])
    del res


def chosen_lr() -> tuple[float, str]:
    g = jload(OUT / "lr_grid.json")
    if g and g.get("chosen_lr") is not None:
        return g["chosen_lr"], "by rule (lowest dev loss, step 1)"
    log(f"  !! no lr from step 1; using fallback {FALLBACK_LR:g} (NOT chosen by the rule)")
    return FALLBACK_LR, "FALLBACK — step 1 did not finish"


# ---------------------------------------------------------------- steps


def step0_hw2() -> str:
    import train as hw2

    cfg_saved = json.loads((HERE / "artifacts" / "config.json").read_text())
    keys = ["encoder", "hidden", "activation", "dropout", "class_weight_power", "lr",
            "weight_decay", "batch_size", "max_epochs", "patience"]
    sweep = jload(HERE / "results" / "sweep.json")["final_setting_3_seeds"]["seeds"]
    seeds = []
    for s in [0, 1, 2]:
        r = hw2.train({**{k: cfg_saved[k] for k in keys}, "seed": s}, evaluate_test=True, verbose=False)
        te = r["test"]
        cm = np.array(te["confusion_matrix"]["rows_true_cols_pred"])
        missed = cm[PI].sum() - cm[PI, PI]
        ref = next(x for x in sweep if x["seed"] == s)
        seeds.append({"seed": s, "best_epoch": r["best_epoch"], "dev_loss": r["dev"]["loss"],
                      "test": {"error": te["error"], "macro_f1": te["macro_f1"], "per_class": te["per_class"],
                               "confusion_matrix": te["confusion_matrix"],
                               "missed_policy_to_value_share": float(cm[PI, VI] / missed)},
                      "matches_sweep_json": abs(te["error"] - ref["test_error"]) < 1e-6
                                            and abs(te["macro_f1"] - ref["test_macro_f1"]) < 1e-6})
        if s == 0:  # also compare with the saved weights
            saved = torch.load(HERE / "artifacts" / "model.pt", map_location="cpu", weights_only=True)
            diff = max((saved[k] - v).abs().max().item() for k, v in r["model"].state_dict().items())
            seeds[-1]["max_abs_weight_diff_vs_artifacts"] = diff
        log(f"  HW2 seed {s}: test err {te['error']:.4f} macro-F1 {te['macro_f1']:.4f} "
            f"(matches sweep.json: {seeds[-1]['matches_sweep_json']})")

    # Part C + probes on the saved HW2 model, through finetune.SentenceEncoder
    enc = ft.SentenceEncoder().eval()
    def proba(texts):
        with torch.no_grad():
            X = torch.cat([enc(texts[i:i + 128]) for i in range(0, len(texts), 128)])
        return hw2_head().predict_proba(X).numpy()
    dv = data()["dev"]
    n = min(64, len(dv["texts"]))
    with torch.no_grad():
        enc_diff = float((enc(dv["texts"][:n]) - dv["X"][:n]).abs().max())
    log(f"  SentenceEncoder vs HW2 cache, first {n} dev sentences: max |diff| = {enc_diff:.2e}")
    tok = enc.tokenizer
    out = {"note": "HW2 setting re-trained in memory for seeds 0-2; nothing written to HW2 files",
           "seeds": seeds, "encoder_vs_cache_max_abs_diff": enc_diff}
    jdump(out, OUT / "hw2_baseline.json")
    jdump(part_c(proba, tok, {"hw2_set": hw2_base_set(tok), "own_set": hw2_base_set(tok)}), OUT / "partC_hw2.json")
    jdump(probes(proba), OUT / "probes_hw2.json")
    ok = all(s["matches_sweep_json"] for s in seeds)
    return f"reproduces sweep.json: {ok}; encoder vs cache {enc_diff:.1e}"


def step1_grid() -> str:
    path = OUT / "lr_grid.json"
    grid = {"rule": "FT-all, seed 0, %d epochs per lr; choose the lr whose lowest dev loss is lowest. "
                    "Test never used." % GRID_EPOCHS,
            "lrs": LR_GRID, "runs": [], "chosen_lr": None}
    for lr in LR_GRID:
        res = ft.finetune({"mode": "ft_all", "seed": 0, "encoder_lr": lr, "max_epochs": GRID_EPOCHS,
                           "patience": 99}, data(), evaluate_test=False, log=log)
        h = res["history"]
        best = min(h[1:], key=lambda r: r["dev_loss"])
        grid["runs"].append({"encoder_lr": lr, "min_dev_loss": best["dev_loss"], "at_epoch": best["epoch"],
                             "dev_error_at_min": best["dev_error"],
                             "dev_policy_f1_at_min": best["dev_policy_f1"], "history": h})
        log(f"  grid lr {lr:g}: min dev loss {best['dev_loss']:.4f} at epoch {best['epoch']} "
            f"(dev err {best['dev_error']:.3f})")
        del res
        jdump(grid, path)  # partial results survive a crash
    win = min(grid["runs"], key=lambda r: r["min_dev_loss"])
    grid["chosen_lr"] = win["encoder_lr"]
    grid["chosen_at_grid_edge"] = win["encoder_lr"] in (min(LR_GRID), max(LR_GRID))
    jdump(grid, path)
    if grid["chosen_at_grid_edge"]:
        log(f"  !! chosen lr {win['encoder_lr']:g} is at the edge of the grid (not extended; run is unattended)")
    log(f"  chosen encoder lr: {win['encoder_lr']:g}")
    return f"chosen lr {win['encoder_lr']:g}" + (" (grid edge)" if grid["chosen_at_grid_edge"] else "")


def step15_frozen() -> str:
    for s in SEEDS:
        full_run("frozen", s, 0.0, with_analyses=(s == 0))
    return "seeds " + ", ".join(map(str, SEEDS))


def step2_ftall0() -> str:
    lr, how = chosen_lr()
    full_run("ft_all", 0, lr, with_analyses=True)
    return f"lr {lr:g} {how}"


def step3_ftall12() -> str:
    lr, how = chosen_lr()
    for s in SEEDS[1:]:
        try:
            full_run("ft_all", s, lr, with_analyses=False)
        except Exception as e:
            log(f"!!! ft_all seed {s} FAILED: {e!r}\n{traceback.format_exc()}")
    return f"lr {lr:g} {how}"


def step4_top2() -> str:
    lr, how = chosen_lr()
    for s in SEEDS:
        try:
            full_run("ft_top2", s, lr, with_analyses=(s == 0))
        except Exception as e:
            log(f"!!! ft_top2 seed {s} FAILED: {e!r}\n{traceback.format_exc()}")
    return f"lr {lr:g} {how} (FT-all's lr reused, no separate grid)"


def load_saved(name: str) -> ft.FineTuneModel:
    cfg = json.loads((RUNS / name / "config.json").read_text())
    m = ft.FineTuneModel(cfg)
    if cfg["mode"] == "frozen":
        m.head.load_state_dict(torch.load(RUNS / name / "head.pt", map_location="cpu", weights_only=True))
    else:
        m.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu", weights_only=True))
    return m.eval()


def best_seed(mode: str) -> str | None:
    runs = [(jload(OUT / f"{mode}_seed{s}" / "metrics.json"), f"{mode}_seed{s}") for s in SEEDS]
    runs = [(m["dev"]["loss"], n) for m, n in runs if m and (RUNS / n / "config.json").exists()]
    return min(runs)[1] if runs else None


def step5_rfi() -> str:
    sys.path.insert(0, str(ROOT / "backend"))
    import predict  # same splitter and text clean-up as HW2's predict_corpus.py
    from predict_corpus import DEONTIC, SHORT

    segments = json.loads(predict.SEGMENTS.read_text())
    docs, texts, seg_of = {}, [], []
    for seg in segments:
        if seg["doc_id"] not in docs:
            docs[seg["doc_id"]] = (predict.FULLTEXT / f"{seg['doc_id']}.txt").read_text(encoding="utf-8")
        para = docs[seg["doc_id"]][seg["char_offset_in_doc"]:seg["char_end_in_doc"]]
        for start, end in predict.split_sentences(para):
            texts.append(predict.model_text(para[start:end]))
            seg_of.append(seg["segment_id"])
        if RFI_LIMIT and len(texts) >= RFI_LIMIT:
            break
    log(f"  RFI: {len(texts):,} sentences")
    words = np.array([len(t.split()) for t in texts])
    deontic = np.array([bool(DEONTIC.search(t)) for t in texts])
    hw2_ref = jload(HERE / "results" / "rfi_policy_share.json")
    out = jload(OUT / "rfi_rates.json") or {}
    out.update({"n_sentences": len(texts), "deontic_pattern": DEONTIC.pattern,
                "hw2_reference": {"policy_share": hw2_ref["predicted_labels"]["policy"]["share"],
                                  "predicted_policy_among_deontic":
                                      hw2_ref["deontic_heuristic"]["rfi_predicted_policy_among_them"],
                                  "mean_p_policy_among_deontic":
                                      hw2_ref["deontic_heuristic"]["rfi_mean_p_policy_among_them"]}})
    done = []
    for mode in ["ft_all", "ft_top2"]:
        name = best_seed(mode)
        if name is None:
            log(f"  RFI: no saved {mode} run, skipped")
            continue
        model = load_saved(name)
        tok = model.encoder.tokenizer
        if "n_over_256_tokens" not in out:
            lens = [len(tok(t)["input_ids"]) for t in texts]
            out["n_over_256_tokens"] = int(sum(x > 256 for x in lens))
            out["token_length_p99"] = float(np.percentile(lens, 99))
        t0 = time.time()
        P = model.predict_proba(texts)
        pred = P.argmax(1)
        long_ = words >= SHORT
        paras = {}
        for sid, k in zip(seg_of, pred):
            paras[sid] = paras.get(sid, False) or k == PI
        Pdv = model.predict_proba(data()["dev"]["texts"])
        dv_deon = np.array([bool(DEONTIC.search(t)) for t in data()["dev"]["texts"]])
        out[name] = {
            "shares": {L[k]: float((pred == k).mean()) for k in range(4)},
            "shares_without_short": {L[k]: float((pred[long_] == k).mean()) for k in range(4)},
            "deontic_share_of_sentences": float(deontic.mean()),
            "predicted_policy_among_deontic": float((pred[deontic] == PI).mean()) if deontic.any() else None,
            "mean_p_policy_among_deontic": float(P[deontic, PI].mean()) if deontic.any() else None,
            "by_threshold": {f"p_policy>={t}": float((P[:, PI] >= t).mean()) for t in [0.4, 0.3, 0.2]},
            "paragraphs_with_policy_share": float(np.mean(list(paras.values()))),
            "cdcp_dev_predicted_policy_among_deontic": float((Pdv[dv_deon].argmax(1) == PI).mean()),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        jdump(out, OUT / "rfi_rates.json")
        log(f"  RFI {name}: policy share {out[name]['shares']['policy']:.3f}, "
            f"policy among deontic {out[name]['predicted_policy_among_deontic']:.3f}")
        done.append(name)
        del model
    return "models: " + ", ".join(done)


# ---------------------------------------------------------------- summary


def fmt(vals: list[float], digits: int = 3) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f} (n=1)"
    s = f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"
    return s if len(vals) == 3 else s + f" (n={len(vals)})"


def test_metrics(key: str) -> list[dict]:
    if key == "hw2":
        b = jload(OUT / "hw2_baseline.json")
        return [s["test"] for s in b["seeds"]] if b else []
    out = []
    for s in SEEDS:
        m = jload(OUT / f"{key}_seed{s}" / "metrics.json")
        if m and m.get("test"):
            out.append(m["test"])
    return out


def plot_dev_loss() -> str | None:
    hf, ha = jload(OUT / "frozen_seed0" / "history.json"), jload(OUT / "ft_all_seed0" / "history.json")
    if not (hf and ha):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for h, label, color in [(hf, "Frozen-new (seed 0)", "#2a78d6"), (ha, "FT-all (seed 0)", "#eb6834")]:
        ep, y = [r["epoch"] for r in h], [r["dev_loss"] for r in h]
        ax.plot(ep, y, color=color, linewidth=2, marker="o", markersize=3, label=label)
        b = int(np.argmin(y))
        ax.plot(ep[b], y[b], marker="o", markersize=8, markerfacecolor="none", color=color)
    ax.set_xlabel("epoch (0 = before training)", color="#52514e")
    ax.set_title("Dev cross-entropy loss — same recipe, frozen vs fine-tuned encoder (circle = best epoch)",
                 loc="left", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dev_loss_frozen_vs_ftall.png", dpi=150)
    plt.close(fig)
    return "dev_loss_frozen_vs_ftall.png"


def build_summary() -> None:
    md = [f"# HW3 overnight — fine-tuning vs frozen encoder",
          f"_Rebuilt {datetime.now():%Y-%m-%d %H:%M} after each step. All selection on CDCP dev; "
          f"test scored once per final run. ± = std over seeds (ddof 1)._", ""]
    if SMOKE:
        md.insert(1, "**SMOKE TEST — tiny data, numbers meaningless.**")

    md += ["## Step status", "", "| step | state | minutes | note |", "|---|---|---|---|"]
    for k, v in STATUS.items():
        md.append(f"| {k} | {v.get('state')} | {v.get('minutes', '')} | {v.get('note') or v.get('error', '')} |")
    md.append("")

    # main comparison
    cols = [(k, lab, test_metrics(k)) for k, lab in MODELS]
    md += ["## Main comparison (CDCP test, 836 sentences)", "",
           "| metric | " + " | ".join(f"{lab} (n={len(ms)})" for _, lab, ms in cols) + " |",
           "|---|" + "---|" * len(cols)]
    rows = [("test error", lambda m: m["error"]), ("macro-F1", lambda m: m["macro_f1"])]
    rows += [(f"F1 {l}", (lambda l: lambda m: m["per_class"][l]["f1"])(l)) for l in L]
    rows += [("missed policy → value share", lambda m: m.get("missed_policy_to_value_share"))]
    for label, get in rows:
        md.append(f"| {label} | " + " | ".join(fmt([get(m) for m in ms]) for _, _, ms in cols) + " |")
    md += ["", "HW2 baseline = the HW2 setting (frozen MiniLM, SGD lr 0.1) re-trained in memory for seeds "
               "0–2 and checked against `results/sweep.json`. Frozen-new = same frozen encoder, FT recipe "
               "(AdamW, head lr 1e-3, wd 0.01, warmup 50, patience 3, max 15).", ""]

    # training facts
    md += ["### Runs", "", "| run | trainable params | best epoch | stop | dev loss | dev error |",
           "|---|---|---|---|---|---|"]
    for k, _ in MODELS[1:]:
        for s in SEEDS:
            m = jload(OUT / f"{k}_seed{s}" / "metrics.json")
            if m:
                md.append(f"| {k}_seed{s} | {m['n_parameters_trainable']:,} | {m['best_epoch']} | "
                          f"{m['stop_reason']} | {m['dev']['loss']:.4f} | {m['dev']['error']:.3f} |")
    b = jload(OUT / "hw2_baseline.json")
    if b:
        md += ["", f"HW2 re-training reproduces sweep.json: "
                   f"{all(s['matches_sweep_json'] for s in b['seeds'])}; seed 0 max |weight diff| vs "
                   f"artifacts/model.pt = {b['seeds'][0].get('max_abs_weight_diff_vs_artifacts', float('nan')):.2e}; "
                   f"SentenceEncoder vs cache max |diff| = {b['encoder_vs_cache_max_abs_diff']:.2e}"]
    md.append("")

    p = plot_dev_loss()
    if p:
        md += ["## Dev loss: Frozen-new vs FT-all (seed 0)", "", f"![dev loss]({p})", ""]

    g = jload(OUT / "lr_grid.json")
    if g:
        md += ["## lr grid (FT-all, seed 0)", "", g["rule"], "",
               "| encoder lr | min dev loss | at epoch | dev error there | dev policy F1 there |",
               "|---|---|---|---|---|"]
        for r in g["runs"]:
            mark = " **← chosen**" if r["encoder_lr"] == g.get("chosen_lr") else ""
            md.append(f"| {r['encoder_lr']:g}{mark} | {r['min_dev_loss']:.4f} | {r['at_epoch']} | "
                      f"{r['dev_error_at_min']:.3f} | {r['dev_policy_f1_at_min']:.3f} |")
        if g.get("chosen_at_grid_edge"):
            md.append("\n⚠️ chosen lr is at the edge of the grid.")
        md.append("")

    # Part C
    pcs = {k: jload(OUT / (f"partC_{k}.json" if k == "hw2" else f"partC_{k}_seed0.json")) for k, _ in MODELS}
    if any(pcs.values()):
        md += ["## Part C — APPEND / PREPEND filler (seed 0 models)", "",
               "Cell: mean P(policy) · P(testimony) · P(value) · still policy / n."]
        for set_name, title in [("hw2_set", "Base set A — the HW2 model's 10 sentences"),
                                ("own_set", "Base set B — each model's own base set")]:
            md += ["", f"**{title}**", "", "| condition | " + " | ".join(lab for _, lab in MODELS) + " |",
                   "|---|" + "---|" * len(MODELS)]
            conds = [("original", lambda r: r["original"])]
            for f in FILLERS:
                for kind in ["APPEND", "PREPEND"]:
                    conds.append((f"{f} {kind}", (lambda f, kind: lambda r: r[f][kind])(f, kind)))
            for cname, get in conds:
                cells = []
                for k, _ in MODELS:
                    r = (pcs.get(k) or {}).get(set_name)
                    if not r or not r.get("n"):
                        cells.append("—")
                        continue
                    s = get(r)
                    cells.append(f"{s['mean_p_policy']:.2f} · {s['mean_p_testimony']:.2f} · "
                                 f"{s['mean_p_value']:.2f} · {s['n_policy']}/{s['n']}")
                md.append(f"| {cname} | " + " | ".join(cells) + " |")
            md.append("| PREPEND < APPEND (F1, F2) | " + " | ".join(
                (f"{r['F1']['n_prepend_below_append']}, {r['F2']['n_prepend_below_append']} of {r['n']}"
                 if (r := (pcs.get(k) or {}).get(set_name)) and r.get("n") else "—") for k, _ in MODELS) + " |")
        md.append("")

    prs = {k: jload(OUT / (f"probes_{k}.json" if k == "hw2" else f"probes_{k}_seed0.json")) for k, _ in MODELS}
    if any(prs.values()):
        md += ["## Probe sentences (seed 0 models)", "", "Cell: P(fact) / P(policy) / P(testimony) / P(value).", "",
               "| sentence | " + " | ".join(lab for _, lab in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for j, t in enumerate(PROBES):
            cells = [" / ".join(f"{pr[j][l]:.2f}" for l in L) if pr else "—" for pr in prs.values()]
            md.append(f"| {t} | " + " | ".join(cells) + " |")
        md.append("")

    r = jload(OUT / "rfi_rates.json")
    if r:
        md += ["## RFI rates (unvalidated; no RFI labels)", "",
               f"{r['n_sentences']:,} sentences; {r.get('n_over_256_tokens', '?')} longer than 256 tokens "
               f"(truncated by the encoder).", "",
               "| model | policy share | policy among should/must/need-to… | mean P(policy) there | "
               "paragraphs with ≥1 policy | CDCP dev: policy among deontic |", "|---|---|---|---|---|---|",
               f"| HW2 (frozen) | {r['hw2_reference']['policy_share']:.3f} | "
               f"{r['hw2_reference']['predicted_policy_among_deontic']:.3f} | "
               f"{r['hw2_reference']['mean_p_policy_among_deontic']:.3f} | 0.127 | 0.657 |"]
        for name, v in r.items():
            if isinstance(v, dict) and "shares" in v:
                md.append(f"| {name} | {v['shares']['policy']:.3f} | {v['predicted_policy_among_deontic']:.3f} | "
                          f"{v['mean_p_policy_among_deontic']:.3f} | {v['paragraphs_with_policy_share']:.3f} | "
                          f"{v['cdcp_dev_predicted_policy_among_deontic']:.3f} |")
        md.append("")

    (OUT / "summary.md").write_text("\n".join(md))
    jdump({"status": STATUS,
           "main": {k: {"n": len(ms), "test_error": [m["error"] for m in ms],
                        "macro_f1": [m["macro_f1"] for m in ms],
                        "per_class_f1": {l: [m["per_class"][l]["f1"] for m in ms] for l in L},
                        "missed_policy_to_value_share": [m.get("missed_policy_to_value_share") for m in ms]}
                    for k, _, ms in cols},
           "chosen_lr": (g or {}).get("chosen_lr")}, OUT / "summary.json")


# ---------------------------------------------------------------- main


def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f"HW3 overnight start (smoke={SMOKE}) out={OUT} runs={RUNS} torch threads={torch.get_num_threads()}")
    steps = [("0_hw2_baseline", step0_hw2), ("1_lr_grid", step1_grid), ("1.5_frozen_new", step15_frozen),
             ("2_ft_all_seed0", step2_ftall0), ("3_ft_all_seeds12", step3_ftall12),
             ("4_ft_top2", step4_top2), ("5_rfi_rates", step5_rfi)]
    only = os.environ.get("HW3_ONLY")  # e.g. "5_rfi_rates" to re-run one step
    for name, fn in steps:
        if only and name not in only.split(","):
            continue
        run_step(name, fn)
    log("HW3 overnight finished")


if __name__ == "__main__":
    main()
