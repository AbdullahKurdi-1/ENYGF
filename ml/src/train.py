"""Train the rod-bundle PINN.

Phase 1 (epochs_data): fit the CFD data only - w, nu_t, k and all
temperature fields - plus the simple constraints (bulk velocity, symmetry
planes, iso-flux gauge). This puts the network near the solution first.
Phase 2 (epochs_physics): add the PDE residuals (axial momentum, energy) and
the iso-flux wall heat flux, ramped in over ramp_epochs so they refine the
fitted fields instead of fighting them. (Starting with the PDEs on a random
network lets the energy residual pull every temperature to the trivial
T = 0 state - what happened in the first version of this code.)

Data come from cfd_field.csv (every mesh cell; a fixed fraction held out for
testing) or, if that file is missing, from the line profiles.

Usage:
    cd ml && python3 src/train.py --config configs/default.yaml
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import load_cfd_field, load_cfd_profiles, temperature_scales
from model import RodBundlePINN
import physics as ph

FLOW = (("w", 0), ("nut", 1), ("k", 2))


def _batch(d, n, gen):
    if n is None or d["x"].shape[0] <= n:
        return d
    idx = torch.randint(0, d["x"].shape[0], (n,), generator=gen, device="cpu").to(d["x"].device)
    return {k: v[idx] for k, v in d.items()}


def data_losses(model, data, batch=None, gen=None):
    """Mean squared error of each quantity, each on its own O(1) scale."""
    scales = {"w": model.u_bulk, "nut": model.nut_scale, "k": model.k_scale}
    losses = {}
    for q, idx in FLOW:
        if q in data:
            d = _batch(data[q], batch, gen)
            pred = model.momentum(d["x"], d["y"], d["re"])[idx]
            losses[f"data_{q}"] = (((pred - d["value"]) / scales[q]) ** 2).mean()
    if "T" in data:
        d = _batch(data["T"], batch, gen)
        pred = model.temperature(d["x"], d["y"], d["re"], d["pr"], d["bc"])
        losses["data_T"] = (((pred - d["value"]) / model.t_ref(d["re"], d["pr"], d["bc"])) ** 2).mean()
    # Wall gradients (Nusselt number and wall shear depend on them directly).
    if "tau_w" in data:
        d = data["tau_w"]
        tau = ph.wall_shear_at(model, d["x"], d["y"], d["re"])
        losses["data_tau_w"] = (((tau - d["value"]) / model.tau_scale) ** 2).mean()
    if "q_w" in data:
        d = data["q_w"]
        _, q = ph.wall_temperature_at(model, d["x"], d["y"], d["re"], d["pr"], d["bc"])
        losses["data_q_w"] = (((q - d["value"]) / model.q_scale) ** 2).mean()
    return losses


@torch.no_grad()
def test_errors(model, test):
    """Relative RMSE (RMSE / data range) on the held-out cells."""
    out = {}
    for q, idx in FLOW:
        if q in test:
            d = test[q]
            pred = model.momentum(d["x"], d["y"], d["re"])[idx]
            out[q] = float(((pred - d["value"]) ** 2).mean().sqrt() / (d["value"].max() - d["value"].min()))
    if "T" in test:
        d = test["T"]
        pred = model.temperature(d["x"], d["y"], d["re"], d["pr"], d["bc"])
        err = ((pred - d["value"]) / model.t_ref(d["re"], d["pr"], d["bc"])) ** 2
        out["T"] = float(err.mean().sqrt())
    return out


def load_training_data(cfg, device):
    """Returns (train, test, training_frame). training.data_source chooses the
    training data: 'field' (mesh cells, optionally thinned by data_fraction)
    or 'lines' (the sampled line profiles only). Testing is always on the
    same held-out cells of the field."""
    tr = cfg["training"]
    field = cfg["paths"].get("cfd_field")
    source = tr.get("data_source", "field")
    fraction = tr.get("data_fraction", 1.0)
    test, frame = {}, None
    if field and Path(field).exists():
        train, test, df = load_cfd_field(field, device, tr.get("holdout_fraction", 0.2), tr["seed"],
                                         fraction if source == "field" else 0.0)
        frame = df[df["split"] == "train"]
        print(f"CFD field data: {field}" + (f"  (data_fraction = {fraction:g})" if source == "field" else ""))
    else:
        source = "lines"
    if source == "lines":
        train, frame = load_cfd_profiles(cfg["paths"]["cfd_profiles"], device)
        if train:
            print(f"training on the line profiles only: {cfg['paths']['cfd_profiles']}")
    return train, test, frame


def train(cfg):
    """Train from a config dict; returns the trained model. Also used by the notebook."""
    tr = cfg["training"]
    w = tr["loss_weights"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    gen = torch.Generator().manual_seed(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    re_values = cfg["flow"]["re_values"]
    pr_values = cfg["thermal"]["pr_values"]

    data, test, df = load_training_data(cfg, device)
    if not data:
        print(
            "WARNING: no CFD data found - training on physics only. Without CFD nu_t the eddy "
            "viscosity is not identifiable, so results will not be meaningful. Run the OpenFOAM pipeline first."
        )
    else:
        print("training points:", {q: int(d["x"].shape[0]) for q, d in data.items()},
              "| held out:", {q: int(d["x"].shape[0]) for q, d in test.items()})

    model = RodBundlePINN(cfg).to(device)
    if df is not None and (df["quantity"] == "T").any():
        model.set_temperature_scales(temperature_scales(df, pr_values))
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    total_epochs = tr["epochs_data"] + tr["epochs_physics"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=tr["lr"] * tr["lr_final_fraction"])

    n_c, n_b, n_d = tr["n_collocation"], tr["n_boundary"], tr.get("batch_data")
    # use_physics: false gives the plain-NN baseline - data loss only, no
    # equations and no constraints (bulk velocity, symmetry, gauge).
    constraints = tr.get("use_physics", True)
    if not constraints:
        total_epochs = tr["epochs_data"] + tr["epochs_physics"]
        tr = {**tr, "epochs_data": total_epochs, "epochs_physics": 0}
        print("use_physics = false: plain neural network (data only) for", total_epochs, "epochs")
    history = []
    t0 = time.time()

    for epoch in range(total_epochs):
        physics = epoch >= tr["epochs_data"]
        ramp = min(1.0, (epoch - tr["epochs_data"] + 1) / max(tr["ramp_epochs"], 1)) if physics else 0.0
        opt.zero_grad()
        terms = data_losses(model, data, n_d, gen)

        if constraints:
            xu, yu = ph.sample_interior(model, n_c, device)
            terms["bulk_velocity"] = sum(ph.bulk_velocity_loss(model, xu, yu, r) for r in re_values) / len(re_values)
            planes = ph.sample_symmetry(model, n_b, device)
            terms["symmetry"] = ph.symmetry_loss(
                model, planes, ph.conditions(re_values, n_b, device), ph.conditions(pr_values, n_b, device),
                ph.conditions([0.0, 1.0], n_b, device), include_thermal=True)
            terms["gauge"] = ph.gauge_loss(model, device, re_values, pr_values)

        if physics:
            x, y = ph.sample_interior(model, n_c, device, wall_biased=True)
            re = ph.conditions(re_values, n_c, device)
            terms["pde_momentum"] = (ph.momentum_residual(model, x, y, re) ** 2).mean()
            terms["wall_shear_balance"] = ph.wall_shear_balance_loss(model, n_b, device, re_values)
            x, y = ph.sample_interior(model, n_c, device, wall_biased=True)
            re = ph.conditions(re_values, n_c, device)
            pr = ph.conditions(pr_values, n_c, device)
            bc = ph.conditions([0.0, 1.0], n_c, device)
            terms["pde_energy"] = (ph.energy_residual(model, x, y, re, pr, bc) ** 2).mean()
            terms["wall_heat_flux"] = ph.wall_heat_flux_loss(model, n_b, device, re_values, pr_values)

        ramped = {"pde_momentum", "pde_energy", "wall_heat_flux", "wall_shear_balance"}
        loss = sum(w[k] * (ramp if k in ramped else 1.0) * v for k, v in terms.items())
        loss.backward()
        opt.step()
        sched.step()

        row = {"epoch": epoch, "phase": 2 if physics else 1, "total": loss.item()}
        row.update({k: v.item() for k, v in terms.items()})
        with torch.no_grad():
            row["dpdz"] = model.pressure_gradient(torch.tensor([[float(re_values[0])]], device=device)).item()
        history.append(row)
        if epoch % tr["print_every"] == 0 or epoch == total_epochs - 1:
            parts = "  ".join(f"{k}={v:.2e}" for k, v in row.items() if k not in ("epoch", "phase"))
            print(f"[phase {row['phase']}] epoch {epoch:5d}  {parts}  ({time.time() - t0:.0f} s)")

    with torch.no_grad():
        G = model.pressure_gradient(torch.tensor([[float(r)] for r in re_values], device=device))
    for r, g in zip(re_values, G.squeeze(1).tolist()):
        u_tau = (g * model.dh / 4) ** 0.5
        print(f"Re={r:g}: learned dp/dz = {g:.4f} m/s2  ->  mean u_tau = {u_tau:.4f} m/s "
              f"(2023 DNS at Re=9800: 0.0637)")
    if test:
        errs = test_errors(model, test)
        print("held-out cells, RMSE / range:", {k: f"{v:.2%}" for k, v in errs.items()})

    ckpt = Path(cfg["paths"]["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt)
    hist_path = Path(cfg["paths"]["loss_history"])
    keys = sorted({k for row in history for k in row}, key=lambda k: (k not in ("epoch", "phase", "total"), k))
    with open(hist_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(history)
    print(f"saved {ckpt} and {hist_path}")
    return model


def main(cfg_path):
    return train(yaml.safe_load(Path(cfg_path).read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
