"""Training monitor and interactive training (HW2 optional parts).

    GET  /train/served          the served model's training record: loss per
                                epoch, train / dev / test results (monitor)
    POST /train                 {lr, max_epochs, class_weight_power} -> {run_id}
                                starts training.train.train() in a background
                                thread on the cached CDCP embeddings
    GET  /train/current         the latest run of this server session, or null
    GET  /train/{run_id}        status and the history so far (the viewer polls)
    POST /train/{run_id}/stop   stop after the current epoch; the best epoch is kept

The training loop is training/train.py's, unchanged: these endpoints only
pass it two hooks — one that records each epoch as it finishes, one that
asks whether Stop was pressed.

Two rules from the HW2 plan are enforced here, not left to the caller:
  - runs are saved under training/runs/ and never into training/artifacts/,
    which holds the model the app serves;
  - interactive runs never score the CDCP test split (it is used once, for
    the served model only).

/train/served only reads JSON, so it works from backend/.venv too; starting
a run needs torch, i.e. the server started from training/.venv.
"""

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
ARTIFACTS = TRAINING / "artifacts"
RUNS = TRAINING / "runs"

router = APIRouter()


# ---------------------------------------------------------------- monitor


@router.get("/train/served")
def served() -> dict:
    """What the served model is and how it trained — read from its artifact files."""
    try:
        config = json.loads((ARTIFACTS / "config.json").read_text())
        history = json.loads((ARTIFACTS / "history.json").read_text())
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    except FileNotFoundError as e:
        raise HTTPException(503, f"no trained model: {Path(e.filename).name} is missing")
    return {"config": config, "history": history, "metrics": metrics}


# ---------------------------------------------------------------- interactive training


class TrainRequest(BaseModel):
    lr: float = Field(0.1, ge=0.001, le=3.0)
    max_epochs: int = Field(100, ge=1, le=300)
    class_weight_power: float = 0.0

    @field_validator("class_weight_power")
    @classmethod
    def on_grid(cls, p: float) -> float:
        if p not in (0.0, 0.5, 1.0):
            raise ValueError("class_weight_power must be 0, 0.5 or 1 (the sweep's grid)")
        return p


class Run:
    def __init__(self, run_id: str, config: dict):
        self.id = run_id
        self.config = config
        self.status = "running"  # running | done | stopped | error
        self.history: list[dict] = []
        self.stop = threading.Event()
        self.error: str | None = None
        self.result: dict | None = None
        self.out_dir: Path | None = None
        self.started = datetime.now().isoformat(timespec="seconds")

    def view(self) -> dict:
        best = min(self.history[1:], key=lambda r: r["dev_loss"], default=None)
        return {
            "run_id": self.id, "status": self.status, "config": self.config, "started": self.started,
            "epoch": self.history[-1]["epoch"] if self.history else 0,
            "best_epoch_so_far": best["epoch"] if best else None,
            "history": self.history, "result": self.result, "error": self.error,
            "out_dir": str(self.out_dir.relative_to(ROOT)) if self.out_dir else None,
        }


_runs: dict[str, Run] = {}
_lock = threading.Lock()


def _safe_out_dir(run_id: str) -> Path:
    """A run's folder under training/runs/. Refuses anything else — above all the served model's folder."""
    out = (RUNS / run_id).resolve()
    if RUNS.resolve() not in out.parents or ARTIFACTS.resolve() in (out, *out.parents):
        raise RuntimeError(f"refusing to save a run to {out}")
    return out


def _train_in_background(run: Run) -> None:
    try:
        if str(TRAINING) not in sys.path:
            sys.path.insert(0, str(TRAINING))
        from train import save_artifacts, train  # training/train.py: the 7-step loop

        results = train(run.config, evaluate_test=False, verbose=False,
                        on_epoch=run.history.append, should_stop=run.stop.is_set)
        run.out_dir = _safe_out_dir(run.id)
        save_artifacts(results, run.out_dir)
        pol = results["dev"]["per_class"]["policy"]
        run.result = {
            "best_epoch": results["best_epoch"], "last_epoch": results["last_epoch"],
            "stop_reason": results["stop_reason"],
            "train_error": results["train"]["error"], "dev_error": results["dev"]["error"],
            "dev_macro_f1": results["dev"]["macro_f1"], "dev_policy_precision": pol["precision"],
            "dev_policy_recall": pol["recall"], "dev_policy_f1": pol["f1"],
        }
        run.status = "stopped" if results["stop_reason"] == "stopped" else "done"
    except Exception as e:  # shown in the panel; the server keeps running
        run.status = "error"
        run.error = f"{type(e).__name__}: {e}"


@router.post("/train")
def start(body: TrainRequest) -> dict:
    try:
        import torch  # noqa: F401  — fail here, not in the thread, when the environment lacks it
    except ImportError:
        raise HTTPException(503, "training needs training/.venv (torch is not installed here)")
    with _lock:
        if any(r.status == "running" for r in _runs.values()):
            raise HTTPException(409, "a run is already in progress — wait for it or press Stop")
        p, lr = body.class_weight_power, body.lr
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-p{p:g}-lr{lr:g}"
        # Fixed parts match the served model (DEFAULTS in train.py); seed 0.
        run = Run(run_id, {"lr": lr, "max_epochs": body.max_epochs, "class_weight_power": p, "seed": 0})
        _runs[run_id] = run
    threading.Thread(target=_train_in_background, args=(run,), daemon=True).start()
    return {"run_id": run_id}


@router.get("/train/current")
def current() -> dict | None:
    return _runs[max(_runs)].view() if _runs else None


@router.get("/train/{run_id}")
def status(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(404, f"no run {run_id!r} in this server session")
    return _runs[run_id].view()


@router.post("/train/{run_id}/stop")
def stop(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(404, f"no run {run_id!r} in this server session")
    _runs[run_id].stop.set()
    return {"run_id": run_id, "stopping": _runs[run_id].status == "running"}
