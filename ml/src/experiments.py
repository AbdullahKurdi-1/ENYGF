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
        nu = ph.nusselt(model, re_value, float(r.Pr), BC_CODE[r.bc], device, n_bulk=20000)
        errs[f"Pr={r.Pr:g} {r.bc}"] = nu / r.Nu_CFD - 1
    out["Nu_err_max_Pr<=2"] = max(abs(v) for k, v in errs.items() if not k.startswith("Pr=7"))
    out["Nu_err_max_all"] = max(abs(v) for v in errs.values())
    out["Nu_errors"] = errs
    return out


def run_one(cfg, name, overrides, out_dir):
    c = copy.deepcopy(cfg)
    c["training"].update(overrides)
    d = Path(out_dir) / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    c["paths"].update(checkpoint=str(d / "checkpoint.pt"), loss_history=str(d / "loss_history.csv"))
    print(f"\n===== {name}: {overrides or 'default settings'} =====")
    t0 = time.time()
    model = trainer.train(c)
    res = {"run": name, "minutes": (time.time() - t0) / 60, **score(model, c)}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


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


def run_all(cfg, runs=None, out_dir="outputs/experiments"):
    """Runs every entry of `runs` (default: RUNS) and returns the summary table.
    Results already on disk are reused, so an interrupted batch can be resumed."""
    results = []
    for name, overrides in (runs or RUNS).items():
        done = Path(out_dir) / slug(name) / "result.json"
        results.append(json.loads(done.read_text()) if done.exists() else run_one(cfg, name, overrides, out_dir))
    table = summarise(results)
    print("\n" + table.to_string(index=False))
    print("CFD values: dp/dz from the CFD wall shear; Nu errors are PINN/CFD - 1. "
          "Held-out cells are identical for every run.")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run_all(yaml.safe_load(Path(parser.parse_args().config).read_text()))
