"""Train the HW2 network on CDCP with an explicit PyTorch loop.

The loop is written out by hand (no Trainer) so every step can be pointed at:

    [1] Initialize   weights, class-weighted loss, plain SGD
    [2] Forward      batch of embeddings -> logits
    [3] Loss         class-weighted cross-entropy
    [4-5] Backprop   zero old gradients, compute new ones with loss.backward()
    [6] Update       w <- w - lr * gradient   (SGD, momentum 0)
    [7] Repeat       next batch / next epoch; after each epoch measure dev loss
                     and stop when it has not improved for `patience` epochs

Data: CDCP train (2,742) is learned from, dev (525) decides when to stop, test
(836) is scored once at the end and only when asked (`evaluate_test=True`).
Decisions behind the defaults: CLAUDE.md, Decision log 2026-10-11.

    python training/train.py                                   # defaults, no test, nothing saved
    python training/train.py --lr 0.3 --class-weight-power 1.0
    python training/train.py --config my.json --test --out training/artifacts

`sweep.py` calls `train()` over the grid and trains the chosen setting.
"""

import argparse
import copy
import json
import math
import pathlib
from functools import lru_cache

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

import cdcp
from embed import ENCODERS, embed
from model import SentenceTypeMLP, count_parameters

HERE = pathlib.Path(__file__).resolve().parent
POLICY = cdcp.LABELS.index("policy")

DEFAULTS = {
    "encoder": "minilm",
    "hidden": [128],
    "activation": "relu",
    "dropout": 0.2,
    "class_weight_power": 0.5,  # p in w_c = (N / (K * n_c)) ** p; 0 = no weighting
    "lr": 0.1,
    "weight_decay": 1e-4,       # L2 penalty, added to the gradient by SGD
    "batch_size": 32,
    "max_epochs": 300,
    "patience": 20,             # epochs without a better dev loss before stopping
    "seed": 0,
}


# ---------------------------------------------------------------- data


@lru_cache(maxsize=4)
def load_data(encoder: str) -> dict:
    """CDCP splits as tensors. Embeddings come from the on-disk cache after the first run."""
    train_texts = [e.text for e in cdcp.load("train")]
    data = {}
    for split in ["train", "dev", "test"]:
        examples = cdcp.load(split)
        X = embed([e.text for e in examples], encoder, fit_texts=train_texts)
        data[split] = {
            "X": torch.from_numpy(X),
            "y": torch.tensor(cdcp.label_index([e.label for e in examples])),
            "comment_ids": [e.comment_id for e in examples],
        }
    return data


def class_weights(y: torch.Tensor, n_classes: int, power: float) -> torch.Tensor:
    """w_c = (N / (K * n_c)) ** power, counted on the training labels only.

    power 1 ("balanced"): a class with half the average count weighs 2x.
    power 0: every weight is 1, i.e. ordinary cross-entropy.
    """
    counts = torch.bincount(y, minlength=n_classes).float()
    return (len(y) / (n_classes * counts)) ** power


# ---------------------------------------------------------------- metrics


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> dict:
    """Loss and error on a whole split, with dropout switched off (eval mode)."""
    model.eval()
    logits = model(X)
    pred = logits.argmax(dim=1)
    p, r, f, support = precision_recall_fscore_support(
        y.numpy(), pred.numpy(), labels=list(range(len(cdcp.LABELS))), zero_division=0)
    return {
        "loss": loss_fn(logits, y).item(),
        "error": 1 - (pred == y).float().mean().item(),
        "macro_f1": float(f.mean()),
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(support[i])}
                      for i, label in enumerate(cdcp.LABELS)},
        "pred": pred,
    }


def policy_f1_ci(y: torch.Tensor, pred: torch.Tensor, comment_ids: list[str],
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap 95% interval for policy F1, resampling whole comments.

    Sentences of one comment resemble each other, so resampling single
    sentences would pretend we have more independent evidence than we do.
    """
    y, pred = y.numpy(), pred.numpy()
    groups = {}
    for i, cid in enumerate(comment_ids):
        groups.setdefault(cid, []).append(i)
    groups = [np.array(ix) for ix in groups.values()]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        _, _, f, _ = precision_recall_fscore_support(y[idx], pred[idx], labels=[POLICY], zero_division=0)
        scores.append(f[0])
    _, _, f, _ = precision_recall_fscore_support(y, pred, labels=[POLICY], zero_division=0)
    return {"f1": float(f[0]), "ci95": [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))],
            "n_boot": n_boot, "unit": "comment", "n_units": len(groups)}


def _public(metrics: dict) -> dict:
    """Drop the prediction tensor so the dict can be written as JSON."""
    return {k: v for k, v in metrics.items() if k != "pred"}


# ---------------------------------------------------------------- training


def train(config: dict | None = None, evaluate_test: bool = False, verbose: bool = True) -> dict:
    """Train one network with one seed. Returns history, metrics and the model (best dev-loss epoch)."""
    cfg = {**DEFAULTS, **(config or {})}
    data = load_data(cfg["encoder"])
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xdv, ydv = data["dev"]["X"], data["dev"]["y"]
    n_classes = len(cdcp.LABELS)

    # [1] Initialize ----------------------------------------------------------
    # Seed first: it fixes the initial weights, the dropout masks and the
    # batch order, so the same config always gives the same model.
    torch.manual_seed(cfg["seed"])
    model = SentenceTypeMLP(in_dim=Xtr.shape[1], hidden=cfg["hidden"], n_classes=n_classes,
                            activation=cfg["activation"], dropout=cfg["dropout"])  # weights: model.init_weights()
    weights = class_weights(ytr, n_classes, cfg["class_weight_power"])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # Plain SGD (momentum 0). weight_decay adds weight_decay * w to each gradient (L2).
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch_order = torch.Generator().manual_seed(cfg["seed"])

    def record(epoch: int) -> dict:
        tr, dv = evaluate(model, Xtr, ytr, loss_fn), evaluate(model, Xdv, ydv, loss_fn)
        pol = dv["per_class"]["policy"]
        return {"epoch": epoch, "train_loss": tr["loss"], "train_error": tr["error"],
                "dev_loss": dv["loss"], "dev_error": dv["error"], "dev_macro_f1": dv["macro_f1"],
                "dev_policy_precision": pol["precision"], "dev_policy_recall": pol["recall"],
                "dev_policy_f1": pol["f1"]}

    history = [record(0)]  # epoch 0 = the untrained network, as a reference point
    best = {"dev_loss": history[0]["dev_loss"], "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    epochs_without_gain = 0
    stop_reason = "max_epochs"

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()  # dropout on
        order = torch.randperm(len(ytr), generator=batch_order)  # new shuffle every epoch
        for start in range(0, len(ytr), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            xb, yb = Xtr[idx], ytr[idx]

            # [2] Forward: embeddings -> logits (batch, 4)
            logits = model(xb)

            # [3] Loss: class-weighted cross-entropy against the true labels
            loss = loss_fn(logits, yb)

            # [4-5] Backprop / gradients: clear last batch's gradients, then
            # backward() fills p.grad = d loss / d p for every parameter p
            optimizer.zero_grad()
            loss.backward()

            # [6] Update (SGD): p <- p - lr * p.grad
            optimizer.step()

        # [7] Repeat / convergence check ----------------------------------
        # After each pass over the training data: measure train and dev,
        # keep the weights with the lowest dev loss, stop when dev loss has
        # not improved for `patience` epochs (the network has started to
        # fit noise in the training set rather than the pattern).
        row = record(epoch)
        history.append(row)
        if verbose:
            print(f"epoch {epoch:3d}  train loss {row['train_loss']:.4f} err {row['train_error']:.3f}  "
                  f"dev loss {row['dev_loss']:.4f} err {row['dev_error']:.3f}  "
                  f"policy P {row['dev_policy_precision']:.2f} R {row['dev_policy_recall']:.2f} "
                  f"F1 {row['dev_policy_f1']:.2f}")

        if not math.isfinite(row["dev_loss"]):
            stop_reason = "diverged"  # lr too large: the loss blew up
            break
        if row["dev_loss"] < best["dev_loss"]:
            best = {"dev_loss": row["dev_loss"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["patience"]:
                stop_reason = "early_stop"
                break

    # Go back to the best epoch before reporting anything.
    model.load_state_dict(best["state"])
    results = {
        "config": cfg,
        "class_weights": dict(zip(cdcp.LABELS, [round(w, 4) for w in weights.tolist()])),
        "n_parameters": count_parameters(model),
        "best_epoch": best["epoch"],
        "last_epoch": history[-1]["epoch"],
        "stop_reason": stop_reason,
        "under_trained": stop_reason == "max_epochs",  # still improving when time ran out
        "history": history,
        "train": _public(evaluate(model, Xtr, ytr, loss_fn)),
        "dev": _public(evaluate(model, Xdv, ydv, loss_fn)),
        "model": model,
    }

    # Test: once, at the end, only for the final setting.
    if evaluate_test:
        te = data["test"]
        test = evaluate(model, te["X"], te["y"], loss_fn)
        test["confusion_matrix"] = {"labels": cdcp.LABELS,
                                    "rows_true_cols_pred": confusion_matrix(
                                        te["y"].numpy(), test["pred"].numpy(),
                                        labels=list(range(n_classes))).tolist()}
        test["policy_f1_ci"] = policy_f1_ci(te["y"], test["pred"], te["comment_ids"])
        results["test"] = _public(test)
    return results


# ---------------------------------------------------------------- saving


def plot_curves(history: list[dict], best_epoch: int, path: pathlib.Path, weighted: bool = True) -> None:
    """Three stacked panels, one y-scale each: loss, error, dev policy F1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"  # train / dev / policy
    ink, muted = "#0b0b0b", "#52514e"
    ep = [h["epoch"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, facecolor="#fcfcfb")
    panels = [
        ("Weighted cross-entropy loss" if weighted else "Cross-entropy loss", [("train", "train_loss", blue), ("dev", "dev_loss", orange)]),
        ("Error (1 − accuracy)", [("train", "train_error", blue), ("dev", "dev_error", orange)]),
        ("Dev policy F1", [("dev policy F1", "dev_policy_f1", aqua)]),
    ]
    for ax, (title, series) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for label, key, color in series:
            ys = [h[key] for h in history]
            ax.plot(ep, ys, color=color, linewidth=2, label=label)
            if len(series) > 1:  # direct label at the line end
                ax.annotate(label, (ep[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                            color=ink, fontsize=9, va="center")
        ax.axvline(best_epoch, color=muted, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#c9c8c2")
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].annotate(f"best epoch {best_epoch}", (best_epoch, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -12), textcoords="offset points", color=muted, fontsize=9)
    axes[-1].set_xlabel("epoch", color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_artifacts(results: dict, out_dir: pathlib.Path, extra_metrics: dict | None = None) -> None:
    """model.pt, label_map.json, config.json, history.json, metrics.json, loss_curve.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = results["config"]
    model = results["model"]

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "label_map.json").write_text(json.dumps({
        "labels": cdcp.LABELS,
        "index_to_label": {str(i): l for i, l in enumerate(cdcp.LABELS)},
        "encoder": cfg["encoder"],
        "encoder_model": ENCODERS.get(cfg["encoder"], cfg["encoder"]),
        "input_dim": model.net[0].in_features,
        "normalize_embeddings": True,
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        **cfg, "class_weights": results["class_weights"], "best_epoch": results["best_epoch"],
        "n_parameters": results["n_parameters"], "optimizer": "SGD (momentum 0)",
        "loss": "CrossEntropyLoss(weight=class_weights)",
        "early_stopping": "dev loss, restore best epoch",
    }, indent=2))
    (out_dir / "history.json").write_text(json.dumps(results["history"], indent=2))
    metrics = {
        "best_epoch": results["best_epoch"], "last_epoch": results["last_epoch"],
        "stop_reason": results["stop_reason"], "under_trained": results["under_trained"],
        "train": results["train"], "dev": results["dev"], "test": results.get("test"),
        **(extra_metrics or {}),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_curves(results["history"], results["best_epoch"], out_dir / "loss_curve.png",
                weighted=cfg["class_weight_power"] != 0)


# ---------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=pathlib.Path, help="JSON file with any of the DEFAULTS keys")
    ap.add_argument("--hidden", type=int, nargs="*", help="hidden layer widths, e.g. --hidden 128 64")
    ap.add_argument("--activation", choices=["relu", "tanh", "gelu"])
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--class-weight-power", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--encoder", choices=list(ENCODERS) + ["lsa"])
    ap.add_argument("--test", action="store_true", help="also score the test split (final setting only)")
    ap.add_argument("--out", type=pathlib.Path, help="write artifacts here, e.g. training/artifacts")
    args = ap.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    results = train(config, evaluate_test=args.test)
    print(f"\nbest epoch {results['best_epoch']} ({results['stop_reason']})  "
          f"dev error {results['dev']['error']:.3f}  dev policy F1 {results['dev']['per_class']['policy']['f1']:.2f}")
    if "test" in results:
        ci = results["test"]["policy_f1_ci"]
        print(f"TEST error {results['test']['error']:.3f}  macro-F1 {results['test']['macro_f1']:.3f}  "
              f"policy F1 {ci['f1']:.2f} [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]")
    if args.out:
        save_artifacts(results, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
