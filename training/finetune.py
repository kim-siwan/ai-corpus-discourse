"""HW3: fine-tune the MiniLM encoder together with the HW2 head.

HW2 kept the encoder frozen and trained only the head on cached vectors.
Here gradients also flow into the encoder, so the 384 numbers themselves can
change to make the four sentence types easier to tell apart.

    text -> MiniLM tokenizer -> 6 transformer layers -> mean pooling -> L2 normalize   (encoder)
         -> Linear(384 -> 128) -> ReLU -> Dropout(0.2) -> Linear(128 -> 4)             (HW2 head, model.py)

Three modes, one training loop:

    frozen    encoder fixed; the head reads the HW2 embedding cache (control, "Frozen-new")
    ft_top2   the top two transformer layers (4 and 5) and the head are trained
    ft_all    every encoder weight and the head are trained

The loop is written out by hand, with the same numbered steps as train.py:

    [1] Initialize   seed, encoder + fresh head, AdamW with two learning rates
    [2] Forward      batch of sentences (or cached vectors) -> logits
    [3] Loss         cross-entropy (unweighted, p = 0 as chosen in HW2)
    [4-5] Backprop   zero old gradients, loss.backward()
    [6] Update       AdamW step, learning rate warmed up over the first 50 steps
    [7] Repeat       after each epoch measure train and dev; keep the lowest dev
                     loss; stop after `patience` epochs without improvement

Why AdamW and not HW2's plain SGD: a pre-trained transformer needs very small,
per-weight steps. AdamW divides each weight's step by a running estimate of
its gradient size, so one learning rate works across layers whose gradients
differ by orders of magnitude. The encoder gets a tiny learning rate (it is
already good); the fresh head gets a larger one.

Test is scored once, at the end, only when `evaluate_test=True`.
Nothing here writes to training/artifacts/ or training/cache/.
"""

import copy
import math

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

import cdcp
from embed import CACHE, _key
from model import SentenceTypeMLP, count_parameters

ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
POLICY = cdcp.LABELS.index("policy")
VALUE = cdcp.LABELS.index("value")

DEFAULTS = {
    "mode": "ft_all",        # frozen | ft_top2 | ft_all
    "encoder_lr": 2e-5,      # ignored in frozen mode; HW3 runs use the grid's choice (1e-5)
    "head_lr": 1e-3,
    "weight_decay": 0.01,    # AdamW: w <- w - lr * wd * w, separate from the gradient step
    "warmup_steps": 50,      # lr rises linearly from ~0 to its value over these steps, then stays
    "batch_size": 32,
    "max_epochs": 15,
    "patience": 3,
    "seed": 0,
    "max_length": 256,       # MiniLM's limit; no CDCP sentence is longer (longest: 166 tokens)
    "hidden": [128],
    "activation": "relu",
    "dropout": 0.2,
    "device": "cpu",
}


# ---------------------------------------------------------------- data


def cached_embeddings(texts: list[str]) -> np.ndarray:
    """Read the HW2 embedding cache for exactly these texts. Never computes or writes."""
    path = CACHE / f"minilm-{_key('minilm', texts)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — the frozen control needs the HW2 cache")
    return np.load(path)


def load_splits(limit: int | None = None) -> dict:
    """CDCP train/dev/test: texts, labels, comment ids, cached frozen embeddings.

    `limit` keeps only the first n examples of each split (smoke test only).
    The cache is looked up with the full split, then sliced, so it is never rebuilt.
    """
    out = {}
    for split in ["train", "dev", "test"]:
        ex = cdcp.load(split)
        texts = [e.text for e in ex]
        X = cached_embeddings(texts)
        n = len(ex) if limit is None else min(limit, len(ex))
        out[split] = {
            "texts": texts[:n],
            "y": torch.tensor(cdcp.label_index([e.label for e in ex[:n]])),
            "comment_ids": [e.comment_id for e in ex[:n]],
            "X": torch.from_numpy(X[:n]),
        }
    return out


# ---------------------------------------------------------------- network


class SentenceEncoder(nn.Module):
    """MiniLM + mean pooling + L2 normalization — the same function as
    SentenceTransformer(all-MiniLM-L6-v2).encode(..., normalize_embeddings=True),
    but written out so that gradients can flow through it."""

    def __init__(self, max_length: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
        # add_pooling_layer=False: BERT's [CLS] pooler is not used by this model.
        self.bert = AutoModel.from_pretrained(ENCODER_MODEL, add_pooling_layer=False)
        self.max_length = max_length

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(device)
        tokens = self.bert(**batch).last_hidden_state            # (B, T, 384), one vector per token
        mask = batch["attention_mask"].unsqueeze(-1).float()      # 1 for real tokens, 0 for padding
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean over real tokens only
        return nn.functional.normalize(mean, p=2, dim=1)          # length 1, as in HW2


class FineTuneModel(nn.Module):
    """Encoder + HW2 head. `forward` takes either a list of sentences or a
    tensor of ready-made (cached) embeddings — the latter skips the encoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = SentenceEncoder(cfg["max_length"])
        self.head = SentenceTypeMLP(in_dim=384, hidden=cfg["hidden"], n_classes=len(cdcp.LABELS),
                                    activation=cfg["activation"], dropout=cfg["dropout"])
        self.mode = cfg["mode"]
        self._freeze()

    def _freeze(self) -> None:
        bert = self.encoder.bert
        if self.mode == "frozen":
            self.frozen_modules = [bert]
        elif self.mode == "ft_top2":
            # Word/position embeddings and layers 0-3 stay fixed; layers 4 and 5 learn.
            self.frozen_modules = [bert.embeddings, *bert.encoder.layer[:4]]
        elif self.mode == "ft_all":
            self.frozen_modules = []
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        for m in self.frozen_modules:
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Dropout on in the trained parts only. Frozen parts always run in
        eval mode, so they compute exactly the fixed function they are."""
        super().train(mode)
        for m in self.frozen_modules:
            m.eval()
        return self

    def forward(self, inputs) -> torch.Tensor:
        x = inputs if isinstance(inputs, torch.Tensor) else self.encoder(inputs)
        return self.head(x)

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        """Encoder output for many texts, in eval mode. Sorted by length so each
        batch pads little; returned in the original order."""
        self.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), 384)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            out[idx] = self.encoder([texts[i] for i in idx]).cpu()
        return out

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        X = self.embed(texts, batch_size)
        self.eval()
        return torch.softmax(self.head(X.to(next(self.parameters()).device)), dim=1).cpu().numpy()


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- metrics


def metrics_from(y: np.ndarray, logits: torch.Tensor, loss_fn: nn.Module) -> dict:
    pred = logits.argmax(1).numpy()
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(cdcp.LABELS))),
                                                   zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(cdcp.LABELS))))
    missed = cm[POLICY].sum() - cm[POLICY, POLICY]
    return {
        "loss": float(loss_fn(logits, torch.as_tensor(y)).item()),
        "error": float((pred != y).mean()),
        "macro_f1": float(f.mean()),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(sup[i])} for i, l in enumerate(cdcp.LABELS)},
        "confusion_matrix": {"labels": cdcp.LABELS, "rows_true_cols_pred": cm.tolist()},
        "missed_policy": int(missed),
        "missed_policy_to_value_share": float(cm[POLICY, VALUE] / missed) if missed else None,
        "pred": pred,
    }


@torch.no_grad()
def evaluate(model: FineTuneModel, split: dict, loss_fn: nn.Module, use_cache: bool) -> dict:
    """Loss, error and per-class scores on a whole split, dropout off."""
    model.eval()
    X = split["X"] if use_cache else model.embed(split["texts"])
    logits = model.head(X.to(next(model.parameters()).device)).cpu()
    return metrics_from(split["y"].numpy(), logits, loss_fn)


def public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "pred"}


# ---------------------------------------------------------------- training


def finetune(config: dict | None = None, data: dict | None = None, evaluate_test: bool = False,
             log=print) -> dict:
    """Train one model with one seed. Returns config, history, metrics and the
    model restored to its best dev-loss epoch."""
    cfg = {**DEFAULTS, **(config or {})}
    data = data or load_splits()
    tr, dv = data["train"], data["dev"]
    use_cache = cfg["mode"] == "frozen"   # frozen encoder: its output is the HW2 cache
    device = torch.device(cfg["device"])

    # [1] Initialize ----------------------------------------------------------
    # The seed fixes the head's initial weights, every dropout mask and the
    # batch order. The encoder starts from the same pre-trained weights in
    # every run.
    torch.manual_seed(cfg["seed"])
    model = FineTuneModel(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": list(model.head.parameters()), "lr": cfg["head_lr"]}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": cfg["encoder_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    # Warmup: during the first steps the fresh head's gradients are large and
    # random; a small lr keeps them from wrecking the pre-trained encoder.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / cfg["warmup_steps"]))
    batch_order = torch.Generator().manual_seed(cfg["seed"])

    def record(epoch: int, running_loss: float | None) -> dict:
        t, d = evaluate(model, tr, loss_fn, use_cache), evaluate(model, dv, loss_fn, use_cache)
        pol = d["per_class"]["policy"]
        return {"epoch": epoch, "train_loss": t["loss"], "train_error": t["error"],
                "train_loss_running": running_loss,
                "dev_loss": d["loss"], "dev_error": d["error"], "dev_macro_f1": d["macro_f1"],
                "dev_policy_precision": pol["precision"], "dev_policy_recall": pol["recall"],
                "dev_policy_f1": pol["f1"]}

    history = [record(0, None)]  # epoch 0 = pre-trained encoder + untrained head
    best = {"dev_loss": history[0]["dev_loss"], "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    without_gain, stop_reason, step = 0, "max_epochs", 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(tr["y"]), generator=batch_order)
        losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            inputs = tr["X"][idx].to(device) if use_cache else [tr["texts"][i] for i in idx]
            yb = tr["y"][idx].to(device)

            # [2] Forward: sentences -> encoder -> head -> logits (batch, 4)
            logits = model(inputs)

            # [3] Loss: cross-entropy against the true labels
            loss = loss_fn(logits, yb)

            # [4-5] Backprop: clear old gradients, compute d loss / d w for every
            # trainable weight — through the head and, unless frozen, the encoder
            optimizer.zero_grad()
            loss.backward()

            # [6] Update: AdamW step for each group with its own lr, then advance warmup
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(loss.item())

        # [7] Repeat / convergence check ----------------------------------
        row = record(epoch, float(np.mean(losses)))
        history.append(row)
        log(f"  [{cfg['mode']} s{cfg['seed']} enc_lr {cfg['encoder_lr']:g}] epoch {epoch:2d}  "
            f"train loss {row['train_loss']:.4f} err {row['train_error']:.3f}  "
            f"dev loss {row['dev_loss']:.4f} err {row['dev_error']:.3f}  "
            f"policy F1 {row['dev_policy_f1']:.2f}")
        if not math.isfinite(row["dev_loss"]):
            stop_reason = "diverged"
            break
        if row["dev_loss"] < best["dev_loss"]:
            best = {"dev_loss": row["dev_loss"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            without_gain = 0
        else:
            without_gain += 1
            if without_gain >= cfg["patience"]:
                stop_reason = "early_stop"
                break

    model.load_state_dict(best["state"])
    results = {
        "config": cfg,
        "optimizer": "AdamW (head lr / encoder lr), linear warmup then constant",
        "n_parameters_total": count_parameters(model),
        "n_parameters_trainable": trainable_parameters(model),
        "best_epoch": best["epoch"], "last_epoch": history[-1]["epoch"],
        "stop_reason": stop_reason, "under_trained": stop_reason == "max_epochs",
        "history": history,
        "train": public(evaluate(model, tr, loss_fn, use_cache)),
        "dev": public(evaluate(model, dv, loss_fn, use_cache)),
        "model": model,
    }
    if evaluate_test:  # once, at the end
        from train import policy_f1_ci

        te = data["test"]
        test = evaluate(model, te, loss_fn, use_cache)
        test["policy_f1_ci"] = policy_f1_ci(te["y"], torch.as_tensor(test["pred"]), te["comment_ids"])
        results["test"] = public(test)
    return results
