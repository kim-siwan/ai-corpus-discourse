"""HW3 experiment 2: cue-preserving pooling with a learned gate.

FT-all (finetune.py) turns a sentence into one vector by averaging the token
vectors. An average lets a single word such as "should" be outvoted by the
rest of the sentence. This module adds a second summary that can concentrate
on a few tokens, and a gate that decides, per sentence, how much of it to use.

    h_t   = MiniLM token vectors (B, T, 384), fine-tuned as in FT-all
    m     = mean over real tokens incl. [CLS]/[SEP]          (B, 384)  — FT-all's pooling, unchanged
    e_t   = w . h_t                                          (B, T)    — w: Linear(384 -> 1), no bias
    a_t   = softmax over word tokens (padding, [CLS], [SEP] excluded)
    c     = sum_t a_t h_t                                    (B, 384)  — the "cue" summary
    g     = sigmoid(W_g [m; c] + b)                          (B, 1)    — W_g: Linear(768 -> 1)
    z     = g * c + (1 - g) * m  ->  L2 normalize  ->  HW2 head 384 -> 128 -> 4

Pooling variants (config key "pool"):

    mean       z = m                               (= FT-all; used only to check this loop)
    cue_gate   learned w, learned gate             (main)
    cue_only   learned w, g fixed to 1
    cue_list   e_t fixed from a word list, learned gate. "Hard" version:
               c = mean of the tokens matching a list entry; c = m if none match

Start state (w = 0, W_g = 0, b = 0): attention is uniform over word tokens, so
c = mean of word tokens *without* [CLS]/[SEP], which is close to but not equal
to m; g = sigmoid(0) = 0.5, so z = (m + c) / 2 at the start, not exactly m.

The new layers are created after the head with the random-number state saved
and restored, so the head's initial weights, every dropout mask and the batch
order are the same as FT-all with the same seed.

The training loop is a copy of finetune.finetune() with the same numbered
steps; only the model differs, plus one extra optimizer group for the new
pooling weights (head learning rate). Nothing here writes files.
"""

import copy
import math
import pathlib
import time

import numpy as np
import torch
from torch import nn

import finetune as ft
from model import count_parameters

POOLS = ["mean", "cue_gate", "cue_only", "cue_list"]
WORDLIST = pathlib.Path(__file__).resolve().parent / "hw3_exp2_wordlist.txt"


def read_wordlist(path: pathlib.Path = WORDLIST) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


class CueModel(ft.FineTuneModel):
    def __init__(self, cfg: dict, wordlist: list[str] | None = None):
        # Encoder + head exactly as FineTuneModel builds them (same random draws as FT-all).
        super().__init__(cfg)
        self.pool = cfg["pool"]
        if self.pool not in POOLS:
            raise ValueError(f"unknown pool {self.pool!r}")
        rng = torch.get_rng_state()
        self.score = None   # w: one weight per embedding dimension -> one score per token
        self.gate = None    # W_g, b: reads [m; c] -> one number per sentence
        if self.pool in ("cue_gate", "cue_only"):
            self.score = nn.Linear(384, 1, bias=False)  # a bias would cancel inside softmax
            nn.init.zeros_(self.score.weight)            # uniform attention at the start
        if self.pool in ("cue_gate", "cue_list"):
            self.gate = nn.Linear(2 * 384, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)               # g = 0.5 at the start
        torch.set_rng_state(rng)  # creating the layers above must not shift later random draws

        tok = self.encoder.tokenizer
        self.phrases = []
        if self.pool == "cue_list":
            words = wordlist if wordlist is not None else read_wordlist()
            # Each entry -> its word-piece ids, e.g. "need to" -> (need, to). Uncased tokenizer = case-insensitive.
            self.phrases = [tuple(tok.convert_tokens_to_ids(tok.tokenize(w))) for w in words]
        # Word pieces that continue a word ("##s"): a list match must not be followed by one (whole words).
        vocab = tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
        self.register_buffer("is_continuation",
                             torch.tensor([t.startswith("##") for t in vocab]), persistent=False)

    def pool_parameters(self) -> list[nn.Parameter]:
        return [p for m in (self.score, self.gate) if m is not None for p in m.parameters()]

    def list_match(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) bool: token is part of a whole-word occurrence of a list entry."""
        B, T = ids.shape
        hit = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
        cont = self.is_continuation[ids]
        for ph in self.phrases:
            k = len(ph)
            if k > T:
                continue
            ok = torch.ones(B, T - k + 1, dtype=torch.bool, device=ids.device)
            for j, tid in enumerate(ph):
                ok &= ids[:, j:T - k + 1 + j] == tid
            after = torch.zeros(B, T - k + 1, dtype=torch.bool, device=ids.device)  # next piece is "##..."
            after[:, :T - k] = cont[:, k:]
            ok &= ~after
            for j in range(k):
                hit[:, j:T - k + 1 + j] |= ok
        return hit

    def encode(self, texts: list[str], detail: bool = False):
        """Sentences -> z (B, 384), L2-normalized. With detail=True also returns
        the pieces (m, c, a, g, ids, cue_mask) for analysis."""
        enc = self.encoder
        device = next(self.parameters()).device
        batch = enc.tokenizer(texts, padding=True, truncation=True, max_length=enc.max_length,
                              return_tensors="pt", return_special_tokens_mask=True).to(device)
        special = batch.pop("special_tokens_mask").bool()
        h = enc.bert(**batch).last_hidden_state                       # (B, T, 384)
        attn = batch["attention_mask"]
        mask = attn.unsqueeze(-1).float()
        m = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)          # same formula as SentenceEncoder

        if self.pool == "mean":
            z = m
            out = {"m": m}
        else:
            cue_mask = attn.bool() & ~special                         # word tokens only
            if self.pool == "cue_list":
                cue_mask = cue_mask & self.list_match(batch["input_ids"])
                e = torch.zeros(h.shape[:2], device=device)           # equal score on every matched token
            else:
                e = self.score(h).squeeze(-1)                         # (B, T)
            has = cue_mask.any(1, keepdim=True)                       # (B, 1); False only for cue_list misses
            e = e.masked_fill(~cue_mask, float("-inf"))
            e = e.masked_fill(~has, 0.0)                              # avoid an all -inf row (NaN)
            a = torch.softmax(e, dim=1) * has                         # (B, T)
            c = (a.unsqueeze(-1) * h).sum(1)                          # (B, 384)
            c = torch.where(has, c, m)                                # no cue token -> c = m
            if self.gate is None:
                g = torch.ones(len(texts), 1, device=device)          # cue_only
            else:
                g = torch.sigmoid(self.gate(torch.cat([m, c], dim=1)))  # (B, 1)
            z = g * c + (1 - g) * m
            out = {"m": m, "c": c, "a": a, "g": g, "ids": batch["input_ids"], "cue_mask": cue_mask}
        z = nn.functional.normalize(z, p=2, dim=1)
        return (z, out) if detail else z

    def forward(self, inputs) -> torch.Tensor:
        x = inputs if isinstance(inputs, torch.Tensor) else self.encode(inputs)
        return self.head(x)

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        """z for many texts, eval mode, length-sorted batches, original order."""
        self.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), 384)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            out[idx] = self.encode([texts[i] for i in idx]).cpu()
        return out

    @torch.no_grad()
    def details(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        """Per sentence: z, normalized m, g, word pieces, a_t over them (eval mode)."""
        self.eval()
        tok = self.encoder.tokenizer
        rows = [None] * len(texts)
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            z, o = self.encode(chunk, detail=True)
            mn = nn.functional.normalize(o["m"], p=2, dim=1)
            for i in range(len(chunk)):
                r = {"z": z[i].cpu().numpy(), "m_norm": mn[i].cpu().numpy()}
                if "a" in o:
                    n = int(o["ids"][i].ne(tok.pad_token_id).sum())
                    r["tokens"] = tok.convert_ids_to_tokens(o["ids"][i][:n].tolist())
                    r["a"] = o["a"][i][:n].cpu().numpy()
                    r["cue_mask"] = o["cue_mask"][i][:n].cpu().numpy()
                    r["g"] = float(o["g"][i])
                rows[s + i] = r
        return rows


def train_cue(config: dict, data: dict | None = None, evaluate_test: bool = False, log=print,
              wordlist: list[str] | None = None) -> dict:
    """Same as finetune.finetune(), with CueModel. One model, one seed."""
    cfg = {**ft.DEFAULTS, "mode": "ft_all", "pool": "cue_gate", **config}
    data = data or ft.load_splits()
    tr, dv = data["train"], data["dev"]
    device = torch.device(cfg["device"])

    # [1] Initialize ----------------------------------------------------------
    torch.manual_seed(cfg["seed"])
    model = CueModel(cfg, wordlist).to(device)
    loss_fn = nn.CrossEntropyLoss()
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": list(model.head.parameters()), "lr": cfg["head_lr"]}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": cfg["encoder_lr"]})
    if model.pool_parameters():  # new pooling weights: fresh like the head, so the head's lr
        groups.append({"params": model.pool_parameters(), "lr": cfg["head_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / cfg["warmup_steps"]))
    batch_order = torch.Generator().manual_seed(cfg["seed"])

    def record(epoch: int, running_loss: float | None, train_seconds: float | None) -> dict:
        t, d = ft.evaluate(model, tr, loss_fn, False), ft.evaluate(model, dv, loss_fn, False)
        pol = d["per_class"]["policy"]
        return {"epoch": epoch, "train_loss": t["loss"], "train_error": t["error"],
                "train_loss_running": running_loss,
                "dev_loss": d["loss"], "dev_error": d["error"], "dev_macro_f1": d["macro_f1"],
                "dev_policy_precision": pol["precision"], "dev_policy_recall": pol["recall"],
                "dev_policy_f1": pol["f1"], "train_seconds": train_seconds}

    history = [record(0, None, None)]  # epoch 0 = pre-trained encoder, untrained head and pooling
    best = {"dev_loss": history[0]["dev_loss"], "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    without_gain, stop_reason = 0, "max_epochs"

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(tr["y"]), generator=batch_order)
        losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            texts = [tr["texts"][i] for i in idx]
            yb = tr["y"][idx].to(device)

            # [2] Forward: sentences -> encoder -> pooling (m, c, gate) -> head -> logits
            logits = model(texts)
            # [3] Loss: unweighted cross-entropy
            loss = loss_fn(logits, yb)
            # [4-5] Backprop: gradients for encoder, pooling weights and head
            optimizer.zero_grad()
            loss.backward()
            # [6] Update: AdamW per group, then advance warmup
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        train_seconds = time.time() - t0

        # [7] Repeat / convergence check ------------------------------------
        row = record(epoch, float(np.mean(losses)), train_seconds)
        history.append(row)
        log(f"  [{cfg['pool']} s{cfg['seed']}] epoch {epoch:2d}  train loss {row['train_loss']:.4f} "
            f"err {row['train_error']:.3f}  dev loss {row['dev_loss']:.4f} err {row['dev_error']:.3f}  "
            f"policy F1 {row['dev_policy_f1']:.2f}  ({train_seconds:.0f}s)")
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
        "n_parameters_total": count_parameters(model),
        "n_parameters_trainable": ft.trainable_parameters(model),
        "best_epoch": best["epoch"], "last_epoch": history[-1]["epoch"],
        "stop_reason": stop_reason, "under_trained": stop_reason == "max_epochs",
        "history": history,
        "train": ft.public(ft.evaluate(model, tr, loss_fn, False)),
        "dev": ft.public(ft.evaluate(model, dv, loss_fn, False)),
        "model": model,
    }
    if evaluate_test:  # once, at the end
        from train import policy_f1_ci

        te = data["test"]
        test = ft.evaluate(model, te, loss_fn, False)
        test["policy_f1_ci"] = policy_f1_ci(te["y"], torch.as_tensor(test["pred"]), te["comment_ids"])
        results["test"] = ft.public(test)
    return results
