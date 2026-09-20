"""Plot predicted vs. reference (digitized-paper or CFD-generated) profiles,
and report RMSE per case. This is the artifact you actually put in a
conference poster: "here's our surrogate against the published data."

Usage:
    cd ml && python3 src/evaluate.py --config configs/default.yaml
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset import load_profiles
from model import RodBundlePINN


def predict(model, samples, device):
    xs = torch.tensor([[s.x, s.y] for s in samples], dtype=torch.float32, device=device)
    re = torch.tensor([s.re for s in samples], dtype=torch.float32, device=device)
    pr = torch.tensor([s.pr for s in samples], dtype=torch.float32, device=device)
    bc = torch.tensor([s.bc for s in samples], dtype=torch.float32, device=device)
    x, y = xs[:, 0:1], xs[:, 1:2]
    with torch.no_grad():
        u, v, p, nut, k = model.momentum(torch.cat([x, y], dim=-1), re)
        theta = model.thermal(torch.cat([x, y], dim=-1), re, pr, bc, u, v)
    u_mag = torch.sqrt(u**2 + v**2 + 1e-12)
    return {
        "u_mag": u_mag.squeeze(-1),
        "k": k.squeeze(-1),
        "nut": nut.squeeze(-1),
        "theta": theta.squeeze(-1),
    }


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg["paths"]["checkpoint"], map_location=device)
    model = RodBundlePINN(
        cfg, cfg["parameter_ranges"]["re_min"], cfg["parameter_ranges"]["re_max"],
        cfg["parameter_ranges"]["pr_values"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = load_profiles(
        Path(cfg["paths"]["digitized_data"]), Path(cfg["paths"]["table2"]), cfg["geometry"]
    )
    if not samples:
        print("No reference samples found - digitize the paper's figures into "
              "digitized_data/ first (see digitized_data/README.md).")
        return

    preds = predict(model, samples, device)

    by_case = defaultdict(list)
    for i, s in enumerate(samples):
        by_case[(s.quantity, _case_label(s))].append(
            (s.x, s.y, s.target, preds[s.quantity][i].item())
        )

    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    rmse_report = []
    for (quantity, case_label), pts in sorted(by_case.items()):
        pts.sort(key=lambda p: p[0] + p[1])
        targets = np.array([p[2] for p in pts])
        predictions = np.array([p[3] for p in pts])
        rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
        rmse_report.append((quantity, case_label, rmse, len(pts)))

        arc = np.linspace(0, 1, len(pts))
        plt.figure(figsize=(5, 4))
        plt.plot(arc, targets, "o-", label="reference")
        plt.plot(arc, predictions, "s--", label="PINN prediction")
        plt.xlabel("normalized position along line")
        plt.ylabel(quantity)
        plt.title(f"{quantity} - {case_label} (RMSE={rmse:.4g})")
        plt.legend()
        plt.tight_layout()
        fname = plots_dir / f"{quantity}_{case_label}.png".replace(" ", "_")
        plt.savefig(fname, dpi=150)
        plt.close()

    print(f"{'quantity':<10}{'case':<20}{'n':<6}{'rmse'}")
    for quantity, case_label, rmse, n in rmse_report:
        print(f"{quantity:<10}{case_label:<20}{n:<6}{rmse:.4g}")


def _case_label(sample):
    return f"Re{sample.re:.0f}_Pr{sample.pr:g}_bc{sample.bc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
