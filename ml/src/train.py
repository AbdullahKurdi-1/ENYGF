"""Train the rod-bundle PINN.

Phase 1 (epochs_momentum): flow only - CFD w / nu_t / k data, axial momentum
residual, bulk-velocity constraint, symmetry planes.
Phase 2 (epochs_joint): adds temperature - CFD T data, energy residual,
iso-flux wall heat flux, gauge.

Usage:
    cd ml && python3 src/train.py --config configs/default.yaml
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import load_cfd_profiles
from model import RodBundlePINN
import physics as ph


def data_losses(model, data):
    losses = {}
    if "w" in data or "nut" in data or "k" in data:
        for q, idx, scale in (("w", 0, model.u_bulk), ("nut", 1, model.nut_scale), ("k", 2, model.k_scale)):
            if q in data:
                d = data[q]
                pred = model.momentum(d["x"], d["y"], d["re"])[idx]
                losses[f"data_{q}"] = (((pred - d["value"]) / scale) ** 2).mean()
    if "T" in data:
        d = data["T"]
        pred = model.temperature(d["x"], d["y"], d["re"], d["pr"], d["bc"])
        t_ref = model.t_ref(d["re"], d["pr"], d["bc"])
        losses["data_T"] = (((pred - d["value"]) / t_ref) ** 2).mean()
    return losses


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    tr = cfg["training"]
    w = tr["loss_weights"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    re_values = cfg["flow"]["re_values"]
    pr_values = cfg["thermal"]["pr_values"]

    data, df = load_cfd_profiles(cfg["paths"]["cfd_profiles"], device)
    if not data:
        print(
            "WARNING: no CFD profiles found at", cfg["paths"]["cfd_profiles"],
            "- training on physics only. Without CFD nu_t the eddy viscosity is not "
            "identifiable, so results will not be meaningful. Run the OpenFOAM pipeline first."
        )
    else:
        print("CFD data points:", {q: int(d["x"].shape[0]) for q, d in data.items()})
    thermal_data = {k: v for k, v in data.items() if k == "T"}
    flow_data = {k: v for k, v in data.items() if k != "T"}

    model = RodBundlePINN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tr["lr_decay_step"], gamma=tr["lr_decay_gamma"])

    n_c, n_b = tr["n_collocation"], tr["n_boundary"]
    total_epochs = tr["epochs_momentum"] + tr["epochs_joint"]
    history = []
    ema = {}

    for epoch in range(total_epochs):
        joint = epoch >= tr["epochs_momentum"]
        opt.zero_grad()
        terms = {}

        terms.update(data_losses(model, flow_data))

        x, y = ph.sample_interior(model, n_c, device, wall_biased=True)
        re = ph.conditions(re_values, n_c, device)
        terms["pde_momentum"] = (ph.momentum_residual(model, x, y, re) ** 2).mean()

        xu, yu = ph.sample_interior(model, n_c, device)
        terms["bulk_velocity"] = sum(ph.bulk_velocity_loss(model, xu, yu, r) for r in re_values) / len(re_values)

        planes = ph.sample_symmetry(model, n_b, device)
        re_b = ph.conditions(re_values, n_b, device)
        pr_b = ph.conditions(pr_values, n_b, device)
        bc_b = ph.conditions([0.0, 1.0], n_b, device)
        terms["symmetry"] = ph.symmetry_loss(model, planes, re_b, pr_b, bc_b, include_thermal=joint)

        if joint:
            terms.update(data_losses(model, thermal_data))
            x, y = ph.sample_interior(model, n_c, device, wall_biased=True)
            re = ph.conditions(re_values, n_c, device)
            pr = ph.conditions(pr_values, n_c, device)
            bc = ph.conditions([0.0, 1.0], n_c, device)
            terms["pde_energy"] = (ph.energy_residual(model, x, y, re, pr, bc) ** 2).mean()
            terms["wall_heat_flux"] = ph.wall_heat_flux_loss(model, n_b, device, re_values, pr_values)
            terms["gauge"] = ph.gauge_loss(model, device, re_values, pr_values)

        if tr["adaptive_loss_normalisation"]:
            for k, v in terms.items():
                val = float(v.detach())
                ema[k] = val if k not in ema else 0.99 * ema[k] + 0.01 * val
            loss = sum(w[k] * v / (ema[k] + 1e-12) for k, v in terms.items())
        else:
            loss = sum(w[k] * v for k, v in terms.items())
        loss.backward()
        opt.step()
        sched.step()

        row = {"epoch": epoch, "phase": 2 if joint else 1, "total": loss.item()}
        row.update({k: float(v) for k, v in terms.items()})
        history.append(row)
        if epoch % tr["print_every"] == 0 or epoch == total_epochs - 1:
            parts = "  ".join(f"{k}={v:.3e}" for k, v in row.items() if k not in ("epoch", "phase"))
            print(f"[phase {row['phase']}] epoch {epoch:5d}  {parts}")

    with torch.no_grad():
        G = model.pressure_gradient(torch.tensor([[float(r)] for r in re_values], device=device))
    for r, g in zip(re_values, G.squeeze(1).tolist()):
        u_tau = (g * model.dh / 4) ** 0.5
        print(f"Re={r:g}: learned dp/dz = {g:.4f} m/s2  ->  mean u_tau = {u_tau:.4f} m/s "
              f"(2023 DNS at Re=9800: 0.0637)")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
