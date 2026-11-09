"""HW3: HW2 vs Frozen/FT-top2/FT-all missed-policy counts and per-epoch cost. Prints only."""
import json, re, datetime as dt, pathlib, os, statistics as st
os.chdir(pathlib.Path(__file__).resolve().parents[2] / "training")  # paths below are relative to training/
R = pathlib.Path("results/hw3"); PI, VI = 1, 3  # label order: fact, policy, testimony, value
def cm_stats(cm):
    row = cm[PI]; return sum(row) - row[PI], row[VI], sum(row)
out = {}
# HW2 max epochs
out["hw2_max_epochs"] = json.load(open("artifacts/config.json"))["max_epochs"]
# dev loss, epochs 0-6: Frozen-new vs FT-all (seed 0)
H = {k: json.load(open(R/f"{k}_seed0/history.json")) for k in ["frozen","ft_all"]}
out["dev_loss_0_6"] = {e: {k: next((h["dev_loss"] for h in H[k] if h["epoch"]==e), None) for k in H} for e in range(7)}
# missed policy on test, from each run's confusion matrix: (missed, -> value, total policy)
cms = {}
b = json.load(open(R/"hw2_baseline.json"))
cms["HW2"] = [s["test"]["confusion_matrix"]["rows_true_cols_pred"] for s in b["seeds"]]
for k, n in [("frozen","Frozen-new"),("ft_top2","FT-top2"),("ft_all","FT-all")]:
    cms[n] = [json.load(open(R/f"{k}_seed{s}/metrics.json"))["test"]["confusion_matrix"]["rows_true_cols_pred"] for s in range(3)]
art = json.load(open("artifacts/metrics.json"))["test"]["confusion_matrix"]["rows_true_cols_pred"]
out["hw2_seed0_equals_artifacts"] = cms["HW2"][0] == art
out["missed"] = {n: [cm_stats(c) for c in v] for n, v in cms.items()}
# timings from run.log (2nd half: full runs, not the lr grid)
log = open(R/"run.log").read().splitlines()
ts = lambda l: dt.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
runs = {}
for i, l in enumerate(log):
    m = re.search(r"\[(\w+) s(\d) enc_lr [^\]]+\] epoch\s+(\d+)", l)
    if m:
        key = f"{m[1]}_seed{m[2]}"
        if m[3] == "1": runs[key] = {"start": ts(log[i-1]), "epochs": []}   # later full run overwrites grid
        runs[key]["epochs"].append(ts(l))
    m = re.search(r"saved (\w+_seed\d)", l)
    if m: runs[m[1]]["end"] = ts(l)
cost = {}
for k, r in runs.items():
    e = [r["start"]] + r["epochs"]
    d = [(b-a).total_seconds() for a, b in zip(e, e[1:])]
    met = json.load(open(R/k/"metrics.json"))
    cost[k] = dict(sec_per_epoch=st.mean(d), epochs_run=len(r["epochs"]), best=met["best_epoch"],
                   total=(r["end"]-r["start"]).total_seconds(), trainable=met["n_parameters_trainable"])
out["cost"] = cost
hw2 = json.load(open("artifacts/metrics.json"))["final_setting_3_seeds"]["seeds"]
out["hw2_best"] = [(s["seed"], s["best_epoch"]) for s in hw2]
out["hw2_baseline_best"] = [(s["seed"], s["best_epoch"]) for s in b["seeds"]]
print(json.dumps(out, indent=1, default=str))
