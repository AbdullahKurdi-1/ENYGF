"""Evaluate a trained PINN.

1. Fit to the CFD: error on held-out cells (never trained on), and profiles
   along the sampled lines.
2. Nusselt numbers: PINN vs. the CFD it learned from vs. the 2023 DNS.
3. Independent validation against the 2023 DNS (never used in training),
   CFD and PINN side by side (validation.py): wall shear (Fig. 6), U+(r+)
   (Fig. 7), turbulent kinetic energy (Fig. 9a), iso-temperature wall heat
   flux (Fig. 11a) and temperature along the unit-cell path (Fig. 12a).

Usage:
    cd ml && python3 src/evaluate.py --config configs/default.yaml
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import BC_CODE, load_cfd_field, load_cfd_profiles
from model import RodBundlePINN
import physics as ph

BCS = ["isoT", "isoFlux"]


def load_model(cfg, device):
    ckpt = torch.load(cfg["paths"]["checkpoint"], map_location=device)
    model = RodBundlePINN(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def col(v, device):
    return torch.tensor(np.asarray(v, dtype=np.float32), device=device).unsqueeze(1)


def predict(model, quantity, x, y, re, pr=None, bc=None):
    with torch.no_grad():
        if quantity == "T":
            return model.temperature(x, y, re, pr, bc)
        w, nut, k = model.momentum(x, y, re)
        return {"w": w, "nut": nut, "k": k}[quantity]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---- 1. CFD fit ----------------------------------------------------------------
def evaluate_holdout(model, cfg, device):
    """RMSE on the held-out cells, per quantity and temperature case."""
    tr = cfg["training"]
    _, test, df = load_cfd_field(cfg["paths"]["cfd_field"], device, tr.get("holdout_fraction", 0.2), tr["seed"])
    if not test:
        return None
    rows = []
    for q, d in test.items():
        pred = predict(model, q, d["x"], d["y"], d["re"], d["pr"], d["bc"]).cpu().numpy().ravel()
        val = d["value"].cpu().numpy().ravel()
        keys = [("flow", np.ones(len(val), bool))] if q != "T" else [
            (f"Pr={p:g} {bc}", (np.isclose(d["pr"].cpu().numpy().ravel(), p)) & (d["bc"].cpu().numpy().ravel() == BC_CODE[bc]))
            for p in cfg["thermal"]["pr_values"] for bc in BCS]
        for case, m in keys:
            if m.any():
                rng = val[m].max() - val[m].min()
                e = rmse(pred[m], val[m])
                rows.append({"quantity": q, "case": case, "n": int(m.sum()), "rmse": e, "rmse_rel": e / rng})
    return pd.DataFrame(rows)


def evaluate_cfd_fit(model, df, device, plots):
    rows = []
    lines = sorted(df["line"].unique())

    flow = df[df["quantity"].isin(["w", "nut", "k"])]
    if not flow.empty:
        qs = [q for q in ("w", "nut", "k") if q in flow["quantity"].unique()]
        fig, axes = plt.subplots(len(qs), len(lines), figsize=(4 * len(lines), 3 * len(qs)), squeeze=False)
        for i, q in enumerate(qs):
            for j, line in enumerate(lines):
                g = flow[(flow["quantity"] == q) & (flow["line"] == line)].sort_values("s")
                if g.empty:
                    continue
                pred = predict(model, q, col(g["x"], device), col(g["y"], device), col(g["Re"], device)).cpu().numpy().ravel()
                rows.append({"case": "flow", "line": line, "quantity": q, "n": len(g),
                             "rmse": rmse(pred, g["value"]), "cfd_max": float(g["value"].abs().max())})
                ax = axes[i][j]
                ax.plot(g["s"], g["value"], "k-", label="CFD")
                ax.plot(g["s"], pred, "r--", label="PINN")
                ax.set_title(f"{q} on {line}")
                ax.set_xlabel("distance along line [m]")
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(plots / "cfd_fit_flow.png", dpi=130)
        plt.close(fig)

    thermal = df[df["quantity"] == "T"]
    for case, gcase in thermal.groupby("case_id"):
        fig, axes = plt.subplots(1, len(lines), figsize=(4 * len(lines), 3), squeeze=False)
        for j, line in enumerate(lines):
            g = gcase[gcase["line"] == line].sort_values("s")
            if g.empty:
                continue
            bc = col(g["bc"].map(BC_CODE), device)
            pred = predict(model, "T", col(g["x"], device), col(g["y"], device), col(g["Re"], device),
                           col(g["Pr"], device), bc).cpu().numpy().ravel()
            rows.append({"case": case, "line": line, "quantity": "T", "n": len(g),
                         "rmse": rmse(pred, g["value"]), "cfd_max": float(g["value"].abs().max())})
            ax = axes[0][j]
            ax.plot(g["s"], g["value"], "k-", label="CFD")
            ax.plot(g["s"], pred, "r--", label="PINN")
            ax.set_title(f"T {case} on {line}")
            ax.set_xlabel("distance along line [m]")
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(plots / f"cfd_fit_T_{case}.png", dpi=130)
        plt.close(fig)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["rmse_rel"] = table["rmse"] / table["cfd_max"].where(table["cfd_max"] > 0, 1.0)
    return table


# ---- 2. DNS validation -----------------------------------------------------------
def evaluate_nusselt(model, cfg, device, plots):
    re = float(cfg["flow"]["re_values"][0])
    dns = pd.read_csv(cfg["paths"]["dns_nusselt"]) if Path(cfg["paths"]["dns_nusselt"]).exists() else None
    cfd_path = cfg["paths"].get("cfd_nusselt")
    cfd = pd.read_csv(cfd_path) if cfd_path and Path(cfd_path).exists() else None
    rows = []
    for pr in cfg["thermal"]["pr_values"]:
        ref = dns[np.isclose(dns["Pr"], pr)] if dns is not None else None
        for bc in BCS:
            nu = ph.nusselt(model, re, pr, BC_CODE[bc], device)
            dns_val = float(ref[f"Nu_{bc}"].iloc[0]) if ref is not None and len(ref) else float("nan")
            c = cfd[np.isclose(cfd["Pr"], pr) & (cfd["bc"] == bc)] if cfd is not None else []
            cfd_val = float(c["Nu_CFD"].iloc[0]) if len(c) else float("nan")
            rows.append({"Pr": pr, "bc": bc, "Nu_PINN": nu, "Nu_CFD": cfd_val, "Nu_DNS_2023": dns_val,
                         "PINN_vs_CFD": nu / cfd_val - 1, "CFD_vs_DNS": cfd_val / dns_val - 1,
                         "PINN_vs_DNS": nu / dns_val - 1})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [f"Pr={r.Pr:g}\n{r.bc}" for r in table.itertuples()]
    xs = np.arange(len(table))
    ax.bar(xs - 0.27, table["Nu_PINN"], 0.27, label="PINN")
    ax.bar(xs, table["Nu_CFD"], 0.27, label="CFD it learned from (RANS)")
    ax.bar(xs + 0.27, table["Nu_DNS_2023"], 0.27, label="DNS (Mathur et al. 2023)")
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("Nu")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "nusselt_vs_dns.png", dpi=130)
    plt.close(fig)
    return table


def wall_shear_distribution(model, re, device, n=181):
    ang = np.linspace(0, math.pi / 2, n)
    tau = ph.wall_shear(model, re, ang, device).cpu().numpy().ravel()
    tau_m = tau[ang <= math.pi / 4 + 1e-9].mean()
    return np.degrees(ang), tau / tau_m, tau


def ray_points(model, angle_deg, n=400):
    phi = math.radians(angle_deg)
    rho_max = model.a / max(math.cos(phi), math.sin(phi))
    d = np.geomspace(1e-7, rho_max - model.r, n)
    rho = model.r + d
    return d, rho * math.cos(phi), rho * math.sin(phi)


def unit_cell_boundary_path(model, n_per_seg=200):
    """The DNS xi path: rod@0deg -> gap centre -> subchannel centre -> rod@45deg."""
    r, a = model.r, model.a
    c = r / math.sqrt(2)
    segs = [((r, 0.0), (a, 0.0)), ((a, 0.0), (a, a)), ((a, a), (c, c))]
    xs, ys, xis, start = [], [], [], 0.0
    for (x0, y0), (x1, y1) in segs:
        L = math.hypot(x1 - x0, y1 - y0)
        t = np.linspace(0, 1, n_per_seg)
        xs.append(x0 + t * (x1 - x0))
        ys.append(y0 + t * (y1 - y0))
        xis.append(start + t * L)
        start += L
    x, y = np.concatenate(xs), np.concatenate(ys)
    # nudge points off the rod surface so they are inside the fluid
    rho = np.hypot(x, y)
    scale = np.where(rho < r * (1 + 1e-6), r * (1 + 1e-6) / np.maximum(rho, 1e-12), 1.0)
    return x * scale, y * scale, np.concatenate(xis)


def evaluate(cfg):
    """Evaluate from a config dict; returns (cfd_fit, nusselt, dns_figures) tables. Also used by the notebook."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    plots = Path(cfg["paths"]["plots_dir"])
    plots.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    holdout = evaluate_holdout(model, cfg, device) if cfg["paths"].get("cfd_field") else None
    if holdout is not None:
        print("\n=== Error on held-out CFD cells, never used in training (rmse_rel = RMSE / range) ===")
        print(holdout.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    _, df = load_cfd_profiles(cfg["paths"]["cfd_profiles"], device)
    if df is not None and not df.empty:
        fit = evaluate_cfd_fit(model, df, device, plots)
        print("\n=== Along the sampled lines (rmse_rel = RMSE / max|CFD|) ===")
        print(fit.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        fit = holdout
        print("No CFD line profiles found - skipping line plots.")

    nu_table = evaluate_nusselt(model, cfg, device, plots)
    print("\n=== Nusselt number: PINN vs. CFD it learned from vs. 2023 DNS Table 1 ===")
    print("All use Dh = 0.0712 m, as the DNS does. The DNS was never used in training.")
    shown = nu_table.copy()
    for c in ("PINN_vs_CFD", "CFD_vs_DNS", "PINN_vs_DNS"):
        shown[c] = shown[c].map(lambda v: f"{v:+.1%}" if v == v else "-")
    shown["Pr"] = shown["Pr"].map("{:g}".format)
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    nu_table.to_csv(plots.parent / "nusselt_comparison.csv", index=False)

    import validation
    results = validation.validate(cfg, model, plots)
    if results.empty:
        print("\nNo digitized DNS figures found - see digitized_data/README.md.")
    print(f"\nPlots written to {plots}/")
    return fit, nu_table, results


def main(cfg_path):
    return evaluate(yaml.safe_load(Path(cfg_path).read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
