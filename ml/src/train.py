"""Train the RANS-PINN surrogate on paper_reference tables + digitized_data
(and, once you've run the OpenFOAM pipeline, digitized_data/cfd_generated).

Usage:
    cd ml && python3 src/train.py --config configs/default.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import load_profiles, samples_to_arrays
from model import RodBundlePINN
from physics import energy_residual, momentum_residuals, symmetry_bc_loss, wall_bc_loss


def sample_interior(n, a, r, device):
    pts = []
    while len(pts) < n:
        batch = (torch.rand(n, 2, device=device) * 2 - 1) * a
        mask = (batch[:, 0] ** 2 + batch[:, 1] ** 2) > (r * 1.02) ** 2
        pts.append(batch[mask])
    xy = torch.cat(pts, dim=0)[:n]
    return xy[:, 0:1].requires_grad_(True), xy[:, 1:2].requires_grad_(True)


def sample_wall(n, r, device):
    theta = torch.rand(n, 1, device=device) * 2 * np.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y


def sample_symmetry_edge(n, a, device):
    coord = (torch.rand(n, 1, device=device) * 2 - 1) * a
    edge_val = torch.full((n, 1), a, device=device)
    sign = torch.where(torch.rand(n, 1, device=device) > 0.5, edge_val, -edge_val)
    return coord, sign


def random_re(n, re_min, re_max, device):
    log_re = torch.rand(n, 1, device=device) * (np.log(re_max) - np.log(re_min)) + np.log(re_min)
    return torch.exp(log_re)


def random_pr_bc(n, pr_values, device):
    pr = torch.tensor(np.random.choice(pr_values, size=n), dtype=torch.float32, device=device).unsqueeze(-1)
    bc = torch.randint(0, 2, (n, 1), device=device)
    return pr, bc


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = cfg["geometry"]
    pranges = cfg["parameter_ranges"]

    samples = load_profiles(Path(cfg["paths"]["digitized_data"]), Path(cfg["paths"]["table2"]), geom)
    data = samples_to_arrays(samples)
    print("Loaded data points per quantity:", {k: len(v[0]) for k, v in data.items()})

    data_tensors = {
        q: (torch.tensor(inp, device=device), torch.tensor(tgt, device=device))
        for q, (inp, tgt) in data.items()
    }

    model = RodBundlePINN(cfg, pranges["re_min"], pranges["re_max"], pranges["pr_values"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg["training"]["lr_decay_step"], gamma=cfg["training"]["lr_decay_gamma"]
    )

    w = cfg["training"]["loss_weights"]
    a, r = geom["half_pitch_m"], geom["rod_radius_m"]
    nu = 1.5e-05
    n_colloc = cfg["training"]["n_collocation"]

    for epoch in range(cfg["training"]["epochs"]):
        opt.zero_grad()

        # --- data loss (velocity/tke/theta against digitized + CFD profiles) ---
        data_loss = torch.tensor(0.0, device=device)
        if "u_mag" in data_tensors:
            inp, tgt = data_tensors["u_mag"]
            x, y, re = inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]
            u, v, p, nut, k_pred = model.momentum(torch.cat([x, y], dim=-1), re.squeeze(-1))
            pred = torch.sqrt(u**2 + v**2 + 1e-12)
            data_loss = data_loss + torch.mean((pred - tgt) ** 2)
        if "k" in data_tensors:
            inp, tgt = data_tensors["k"]
            x, y, re = inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]
            u, v, p, nut, k_pred = model.momentum(torch.cat([x, y], dim=-1), re.squeeze(-1))
            data_loss = data_loss + torch.mean((k_pred - tgt) ** 2)
        if "theta" in data_tensors:
            inp, tgt = data_tensors["theta"]
            x, y, re, pr, bc = inp[:, 0:1], inp[:, 1:2], inp[:, 2:3], inp[:, 3:4], inp[:, 4:5]
            out = model(x, y, re.squeeze(-1), pr.squeeze(-1), bc.squeeze(-1))
            data_loss = data_loss + torch.mean((out["theta"] - tgt) ** 2)

        # --- physics residuals at random collocation points ---
        x, y = sample_interior(n_colloc, a, r, device)
        re = random_re(n_colloc, pranges["re_min"], pranges["re_max"], device).squeeze(-1)
        cont, mom_x, mom_y, _ = momentum_residuals(model.momentum, x, y, re, nu)
        physics_loss = (
            w["physics_continuity"] * torch.mean(cont**2)
            + w["physics_momentum"] * torch.mean(mom_x**2 + mom_y**2)
        )

        x_e, y_e = sample_interior(n_colloc, a, r, device)
        re_e = random_re(n_colloc, pranges["re_min"], pranges["re_max"], device).squeeze(-1)
        pr_e, bc_e = random_pr_bc(n_colloc, pranges["pr_values"], device)
        x_e.requires_grad_(True)
        y_e.requires_grad_(True)
        energy_res, _ = energy_residual(model, x_e, y_e, re_e, pr_e.squeeze(-1), bc_e.squeeze(-1))
        physics_loss = physics_loss + w["physics_energy"] * torch.mean(energy_res**2)

        # --- boundary condition losses ---
        xw, yw = sample_wall(n_colloc // 4, r, device)
        re_w = random_re(n_colloc // 4, pranges["re_min"], pranges["re_max"], device).squeeze(-1)
        bc_loss = w["bc_wall"] * wall_bc_loss(model, xw, yw, re_w)

        n_sym = n_colloc // 4
        coord, edge = sample_symmetry_edge(n_sym, a, device)
        re_s = random_re(n_sym, pranges["re_min"], pranges["re_max"], device).squeeze(-1)
        bc_loss = bc_loss + w["bc_symmetry"] * symmetry_bc_loss(model, edge, coord, re_s, normal_is_x=True)
        coord2, edge2 = sample_symmetry_edge(n_sym, a, device)
        re_s2 = random_re(n_sym, pranges["re_min"], pranges["re_max"], device).squeeze(-1)
        bc_loss = bc_loss + w["bc_symmetry"] * symmetry_bc_loss(model, coord2, edge2, re_s2, normal_is_x=False)

        loss = w["data"] * data_loss + physics_loss + bc_loss
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 200 == 0 or epoch == cfg["training"]["epochs"] - 1:
            print(
                f"epoch {epoch:5d}  loss {loss.item():.4e}  "
                f"data {data_loss.item():.4e}  physics {physics_loss.item():.4e}  bc {bc_loss.item():.4e}"
            )

    ckpt_path = Path(cfg["paths"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
