"""Choose the class-weight power p and the learning rate on CDCP dev, then train the final model.

Rule (CLAUDE.md, Decision log 2026-10-11):

  1. For each p in {0, 0.5, 1.0}: train every lr in {0.03, 0.1, 0.3, 1.0}
     with seeds 0, 1, 2, and keep the lr with the lowest mean dev error.
  2. Among those three p, take every p whose dev error is within 1 pt of the
     lowest; of those, choose the one with the highest mean dev policy recall.
  3. Train the chosen (p, lr) with seeds 0, 1, 2, now also scoring test.
     The seed with the lowest dev loss is saved to training/artifacts/
     (chosen by dev, never by test). Test error is reported as the mean of
     the three seeds.

Everything up to step 3 uses dev only. Dev scores are optimistic because dev
both stops training and picks the setting.

    python training/sweep.py

Writes training/results/sweep.json and training/artifacts/.
"""

import json
import pathlib
import time

import numpy as np

from train import DEFAULTS, HERE, save_artifacts, train

P_GRID = [0.0, 0.5, 1.0]
LR_GRID = [0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]
TIE_MARGIN = 0.01  # "within 1 pt" of dev error


def summarize_run(r: dict) -> dict:
    pol = r["dev"]["per_class"]["policy"]
    return {"seed": r["config"]["seed"], "best_epoch": r["best_epoch"], "stop_reason": r["stop_reason"],
            "dev_loss": r["dev"]["loss"], "dev_error": r["dev"]["error"], "dev_macro_f1": r["dev"]["macro_f1"],
            "dev_policy_precision": pol["precision"], "dev_policy_recall": pol["recall"],
            "dev_policy_f1": pol["f1"], "train_error": r["train"]["error"]}


def mean_of(runs: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in runs]))


def main() -> None:
    t0 = time.time()

    # ---- grid: every (p, lr) with 3 seeds, dev only
    settings = []
    for p in P_GRID:
        for lr in LR_GRID:
            runs = [summarize_run(train({**DEFAULTS, "class_weight_power": p, "lr": lr, "seed": s}, verbose=False))
                    for s in SEEDS]
            row = {"p": p, "lr": lr, "runs": runs,
                   **{f"mean_{k}": mean_of(runs, k) for k in
                      ["dev_error", "dev_macro_f1", "dev_policy_precision", "dev_policy_recall",
                       "dev_policy_f1", "best_epoch", "train_error"]},
                   "any_diverged": any(r["stop_reason"] == "diverged" for r in runs),
                   "any_under_trained": any(r["stop_reason"] == "max_epochs" for r in runs)}
            settings.append(row)
            print(f"p {p:3.1f}  lr {lr:<5}  dev err {row['mean_dev_error']:.3f}  "
                  f"policy P {row['mean_dev_policy_precision']:.2f} R {row['mean_dev_policy_recall']:.2f} "
                  f"F1 {row['mean_dev_policy_f1']:.2f}  best epoch {row['mean_best_epoch']:.0f}"
                  + ("  DIVERGED" if row["any_diverged"] else "")
                  + ("  UNDER-TRAINED" if row["any_under_trained"] else ""))

    # ---- step 1: best lr per p, by mean dev error
    best_per_p = []
    for p in P_GRID:
        rows = [s for s in settings if s["p"] == p]
        best_per_p.append(min(rows, key=lambda s: s["mean_dev_error"]))

    # ---- step 2: within 1 pt of the lowest dev error -> highest dev policy recall
    lowest = min(s["mean_dev_error"] for s in best_per_p)
    eligible = [s for s in best_per_p if s["mean_dev_error"] <= lowest + TIE_MARGIN]
    chosen = max(eligible, key=lambda s: s["mean_dev_policy_recall"])
    print(f"\nlowest dev error {lowest:.3f}; within {TIE_MARGIN:.2f}: "
          + ", ".join(f"p={s['p']}" for s in eligible)
          + f"  ->  chosen p={chosen['p']}, lr={chosen['lr']}")

    # ---- step 3: final setting, 3 seeds, test scored once per seed
    final_runs = [train({**DEFAULTS, "class_weight_power": chosen["p"], "lr": chosen["lr"], "seed": s},
                        evaluate_test=True, verbose=False) for s in SEEDS]
    keep = min(final_runs, key=lambda r: r["dev"]["loss"])  # selected by dev loss, not test
    seeds_summary = [{**summarize_run(r), "test_error": r["test"]["error"], "test_macro_f1": r["test"]["macro_f1"],
                      "test_policy_f1": r["test"]["per_class"]["policy"]["f1"]} for r in final_runs]
    test_errors = [r["test"]["error"] for r in final_runs]
    selection = {
        "rule": "per p: lr with lowest mean dev error; then among p within 1 pt of the lowest, "
                "highest mean dev policy recall; saved seed = lowest dev loss",
        "chosen": {"class_weight_power": chosen["p"], "lr": chosen["lr"]},
        "eligible_p": [s["p"] for s in eligible],
        "saved_seed": keep["config"]["seed"],
    }
    seed_block = {"seeds": seeds_summary,
                  "test_error_mean": float(np.mean(test_errors)),
                  "test_error_sd": float(np.std(test_errors, ddof=1)),
                  "test_error_range": [float(min(test_errors)), float(max(test_errors))]}

    out_dir = HERE / "artifacts"
    save_artifacts(keep, out_dir, extra_metrics={"selection": selection, "final_setting_3_seeds": seed_block})

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "sweep.json").write_text(json.dumps({
        "grid": {"p": P_GRID, "lr": LR_GRID, "seeds": SEEDS, "tie_margin": TIE_MARGIN},
        "fixed": {k: v for k, v in DEFAULTS.items() if k not in ("class_weight_power", "lr", "seed")},
        "settings": settings,
        "best_lr_per_p": [{"p": s["p"], "lr": s["lr"], "mean_dev_error": s["mean_dev_error"],
                           "mean_dev_policy_recall": s["mean_dev_policy_recall"]} for s in best_per_p],
        "selection": selection,
        "final_setting_3_seeds": seed_block,
    }, indent=2))

    t = keep["test"]
    ci = t["policy_f1_ci"]
    print(f"\nFINAL p={chosen['p']} lr={chosen['lr']}  saved seed {keep['config']['seed']} "
          f"(best epoch {keep['best_epoch']}, {keep['stop_reason']})")
    print(f"test error, 3 seeds: {seed_block['test_error_mean']:.3f} ± {seed_block['test_error_sd']:.3f} "
          f"(range {min(test_errors):.3f}–{max(test_errors):.3f})")
    print(f"saved seed test: error {t['error']:.3f}  macro-F1 {t['macro_f1']:.3f}  "
          f"policy F1 {ci['f1']:.2f} [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]")
    print(f"wrote {results_dir / 'sweep.json'} and {out_dir}/  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
