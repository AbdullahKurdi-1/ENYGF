"""Sparse-data experiment: how much CFD data does the network need, and how
much do the physics terms help? (docs/research_plan.md)

Each run trains a copy of the config with a few settings changed, then scores
it on the same held-out cells and against the CFD's own Nusselt numbers.

Notebook:
    import experiments as ex
    runs = {"10% PINN": {"data_fraction": 0.1},
            "10% plain NN": {"data_fraction": 0.1, "use_physics": False}}
    table = ex.run_all(cfg, runs)

Terminal:
    cd ml && python3 src/experiments.py            # all runs in RUNS below
"""
import argparse
import copy
import json
import re
import time
from pathlib import Path

import pandas as pd
import yaml

import physics as ph
import train as trainer
from dataset import BC_CODE

# The research-plan table: amount of data x with/without physics.
# A run may give "checkpoint": path to score an already-trained model instead
# of training (e.g. the 100% PINN from notebook Step 3).
RUNS = {
    "100% PINN": {},
    "100% plain NN": {"use_physics": False},
    "10% PINN": {"data_fraction": 0.1},
    "10% plain NN": {"data_fraction": 0.1, "use_physics": False},
    "1% PINN": {"data_fraction": 0.01},
    "1% plain NN": {"data_fraction": 0.01, "use_physics": False},
    "lines PINN": {"data_source": "lines"},
    "lines plain NN": {"data_source": "lines", "use_physics": False},
}

# Extra random seeds (1 and 2; seed 0 is the main run). The seed changes which
# cells are drawn for training, the held-out cells, the network's starting
# weights and the collocation points - the spread shows how robust each
# result is. Most important for the sparse rows (1%, lines).
SEEDS = (1, 2)
SEED_RUNS = {
    f"{data} {model} (seed {s})": {**base, "seed": s, **extra}
    for data, base in (("10%", {"data_fraction": 0.1}), ("1%", {"data_fraction": 0.01}),
                       ("lines", {"data_source": "lines"}))
    for s in SEEDS for model, extra in (("PINN", {}), ("plain NN", {"use_physics": False}))
}

# Sensor importance: the lines-only PINN with one measurement line removed at
# a time. The bigger the error grows, the more that line matters - i.e. where
# measurements are most valuable.
ALL_LINES = ["line15", "seg1", "seg2", "seg3"]
SENSOR_RUNS = {
    f"lines PINN without {drop}": {"data_source": "lines", "data_lines": [l for l in ALL_LINES if l != drop]}
    for drop in ALL_LINES
}
SENSOR_SEED_RUNS = {
    f"lines PINN without {drop} (seed {s})": {"data_source": "lines", "seed": s,
                                              "data_lines": [l for l in ALL_LINES if l != drop]}
    for drop in ALL_LINES for s in SEEDS
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def score(model, cfg, device="cpu"):
    """Held-out error, pressure gradient and Nusselt number vs. the CFD."""
    re_value = float(cfg["flow"]["re_values"][0])
    out = {}
    _, test, _ = trainer.load_training_data(cfg, device)
    for q, e in trainer.test_errors(model, test).items():
        out[f"test_err_{q}"] = e
    out["dpdz_wall_shear"] = ph.pressure_gradient_from_wall_shear(model, re_value, device)
    cfd = pd.read_csv(cfg["paths"]["cfd_nusselt"])
    errs = {}
    for r in cfd.itertuples():
        nu = ph.nusselt(model, re_value, float(r.Pr), BC_CODE[r.bc], device)
        errs[f"Pr={r.Pr:g} {r.bc}"] = nu / r.Nu_CFD - 1
    out["Nu_err_max_Pr<=2"] = max(abs(v) for k, v in errs.items() if not k.startswith("Pr=7"))
    out["Nu_err_max_all"] = max(abs(v) for v in errs.values())
    out["Nu_errors"] = errs
    return out


def run_one(cfg, name, overrides, out_dir):
    c = copy.deepcopy(cfg)
    overrides = dict(overrides)
    existing = overrides.pop("checkpoint", None)
    c["training"].update(overrides)
    d = Path(out_dir) / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name}: {overrides or 'default settings'} =====")
    t0 = time.time()
    if existing and Path(existing).exists():
        from evaluate import load_model
        c["paths"]["checkpoint"] = str(existing)
        model = load_model(c, "cpu")
        print(f"scoring the already-trained model {existing} (no training)")
    else:
        c["paths"].update(checkpoint=str(d / "checkpoint.pt"), loss_history=str(d / "loss_history.csv"))
        model = trainer.train(c)
    res = {"run": name, "minutes": (time.time() - t0) / 60, **score(model, c)}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def group_name(run):
    return re.sub(r" \(seed \d+\)$", "", run)


def plot_summary(results, path):
    """Bar chart of held-out error and worst Nusselt error, PINN vs plain NN,
    per amount of data (seed repeats averaged, their range shown)."""
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.DataFrame(results)
    df = df[df["run"].str.contains("PINN|plain NN") & ~df["run"].str.contains("without")]
    df["group"] = df["run"].map(group_name)
    df["data"] = df["group"].str.replace(" PINN", "").str.replace(" plain NN", "")
    df["model"] = np.where(df["group"].str.endswith("PINN"), "PINN", "plain NN")
    order = [d for d in ["100%", "10%", "1%", "lines"] if d in set(df["data"])]
    metrics = [("test_err_w", "held-out error, velocity"), ("test_err_T", "held-out error, temperature"),
               ("Nu_err_max_Pr<=2", "worst Nusselt error (Pr <= 2)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(order))
    for ax, (m, title) in zip(axes, metrics):
        for k, (model, off) in enumerate((("PINN", -0.2), ("plain NN", 0.2))):
            g = df[df["model"] == model].groupby("data")[m]
            mean = [g.mean().get(d, np.nan) * 100 for d in order]
            lo = [mean[i] - g.min().get(d, np.nan) * 100 for i, d in enumerate(order)]
            hi = [g.max().get(d, np.nan) * 100 - mean[i] for i, d in enumerate(order)]
            ax.bar(x + off, mean, 0.4, yerr=[lo, hi], capsize=3, label=model)
        ax.set_xticks(x, [f"{d} data" for d in order])
        ax.set_ylabel("%")
        ax.set_yscale("log")
        ax.set_title(title)
    axes[0].legend()
    fig.suptitle("Does the physics help? Same network, trained with (PINN) and without (plain NN) the equations")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.show()
    plt.close(fig)


def summarise(results):
    rows = []
    for r in results:
        rows.append({
            "run": r["run"],
            "held-out w": f"{r['test_err_w']:.1%}",
            "held-out T": f"{r['test_err_T']:.1%}",
            "dp/dz": f"{r['dpdz_wall_shear']:.4f}",
            "worst Nu error (Pr<=2)": f"{r['Nu_err_max_Pr<=2']:.1%}",
            "worst Nu error (all)": f"{r['Nu_err_max_all']:.1%}",
            "minutes": f"{r['minutes']:.0f}",
        })
    return pd.DataFrame(rows)


def run_all(cfg, runs=None, out_dir="outputs/experiments", plot=False):
    """Runs every entry of `runs` (default: RUNS) and returns the summary table.
    Results already on disk are reused, so an interrupted batch can be resumed."""
    results = []
    for name, overrides in (runs or RUNS).items():
        done = Path(out_dir) / slug(name) / "result.json"
        results.append(json.loads(done.read_text()) if done.exists() else run_one(cfg, name, overrides, out_dir))
    table = summarise(results)
    pd.DataFrame(results).drop(columns=["Nu_errors"]).to_csv(Path(out_dir) / "summary.csv", index=False)
    if plot:
        plot_summary(results, Path(out_dir) / "summary.png")
    print("\n" + table.to_string(index=False))
    print("CFD values: dp/dz from the CFD wall shear; Nu errors are PINN/CFD - 1. "
          "Held-out cells are identical for every run.")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run_all(yaml.safe_load(Path(parser.parse_args().config).read_text()))
