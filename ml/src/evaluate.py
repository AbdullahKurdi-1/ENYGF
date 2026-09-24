"""Evaluate a trained PINN.

1. Fit to the CFD it was trained on (per line / quantity / case), with RMSE.
2. Independent validation against the 2023 DNS (never used in training):
   - Nusselt numbers vs. the DNS Table 1 (no digitizing needed)
   - wall shear distribution (Fig. 6), velocity in wall units (Fig. 7),
     iso-temperature wall heat flux (Fig. 11a), iso-temperature profiles
     along the unit-cell boundary (Fig. 12a) - each only if you have
     digitized it into digitized_data/ (see that folder's README).

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

from dataset import BC_CODE, load_cfd_profiles, load_dns_digitized
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
    rows = []
    for pr in cfg["thermal"]["pr_values"]:
        ref = dns[np.isclose(dns["Pr"], pr)] if dns is not None else None
        for bc in BCS:
            nu = ph.nusselt(model, re, pr, BC_CODE[bc], device)
            dns_val = float(ref[f"Nu_{bc}"].iloc[0]) if ref is not None and len(ref) else float("nan")
            rows.append({"Pr": pr, "bc": bc, "Nu_PINN": nu, "Nu_DNS_2023": dns_val,
                         "rel_error": (nu - dns_val) / dns_val if dns_val == dns_val else float("nan")})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = [f"Pr={r.Pr:g}\n{r.bc}" for r in table.itertuples()]
    xs = np.arange(len(table))
    ax.bar(xs - 0.2, table["Nu_PINN"], 0.4, label="PINN")
    ax.bar(xs + 0.2, table["Nu_DNS_2023"], 0.4, label="DNS (Mathur et al. 2023)")
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


def fold_angle(theta_deg):
    return np.abs(np.asarray(theta_deg, dtype=float)) % 90.0


def compare_fig6(model, digit, re, device, plots):
    deg, ratio, _ = wall_shear_distribution(model, re, device)
    pred_at = np.interp(fold_angle(digit["x"]), deg, ratio)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deg, ratio, "r-", label="PINN")
    ax.plot(fold_angle(digit["x"]), digit["y"], "ko", ms=3, label="DNS Fig. 6")
    ax.set_xlabel("angle from narrow gap [deg]")
    ax.set_ylabel("tau_w / tau_w,m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "dns_fig6_wall_shear.png", dpi=130)
    plt.close(fig)
    return [{"figure": "6", "case": "all", "n": len(digit), "rmse": rmse(pred_at, digit["y"])}]


def ray_points(model, angle_deg, n=400):
    phi = math.radians(angle_deg)
    rho_max = model.a / max(math.cos(phi), math.sin(phi))
    d = np.geomspace(1e-7, rho_max - model.r, n)
    rho = model.r + d
    return d, rho * math.cos(phi), rho * math.sin(phi)


def compare_fig7(model, digit, re, device, plots):
    out = []
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for angle, g in digit.groupby(digit["case_id"].astype(float)):
        d, x, y = ray_points(model, angle)
        with torch.no_grad():
            w, _, _ = model.momentum(col(x, device), col(y, device), torch.full((len(x), 1), re, device=device))
        tau = ph.wall_shear(model, re, [math.radians(angle)], device).item()
        u_tau = math.sqrt(max(tau, 1e-12))
        nu = model.nu(torch.tensor(re)).item()
        r_plus, u_plus = d * u_tau / nu, w.cpu().numpy().ravel() / u_tau
        ax.semilogx(r_plus, u_plus, "-", label=f"PINN {angle:g} deg")
        ax.semilogx(g["x"], g["y"], "o", ms=3, label=f"DNS {angle:g} deg")
        pred_at = np.interp(np.log(g["x"]), np.log(r_plus), u_plus)
        out.append({"figure": "7", "case": f"{angle:g} deg", "n": len(g), "rmse": rmse(pred_at, g["y"])})
    ax.set_xlabel("r+")
    ax.set_ylabel("U+")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "dns_fig7_velocity_wall_units.png", dpi=130)
    plt.close(fig)
    return out


def compare_fig11a(model, digit, re, device, plots):
    out = []
    ang = np.linspace(0, math.pi / 2, 181)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for pr, g in digit.groupby(digit["case_id"].astype(float)):
        _, flux = ph.wall_temperature_data(model, re, pr, 0.0, ang, device)
        flux = flux.cpu().numpy().ravel()
        ratio = flux / flux[ang <= math.pi / 4 + 1e-9].mean()
        ax.plot(np.degrees(ang), ratio, "-", label=f"PINN Pr={pr:g}")
        ax.plot(fold_angle(g["x"]), g["y"], "o", ms=3, label=f"DNS Pr={pr:g}")
        pred_at = np.interp(fold_angle(g["x"]), np.degrees(ang), ratio)
        out.append({"figure": "11a", "case": f"Pr={pr:g}", "n": len(g), "rmse": rmse(pred_at, g["y"])})
    ax.set_xlabel("angle from narrow gap [deg]")
    ax.set_ylabel("phi / phi_m (iso-temperature)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "dns_fig11a_wall_heat_flux.png", dpi=130)
    plt.close(fig)
    return out


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


def compare_fig12a(model, digit, re, dh_dns, device, plots):
    out = []
    x, y, xi = unit_cell_boundary_path(model)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for pr, g in digit.groupby(digit["case_id"].astype(float)):
        n = len(x)
        with torch.no_grad():
            T = model.temperature(col(x, device), col(y, device), torch.full((n, 1), re, device=device),
                                  torch.full((n, 1), pr, device=device), torch.zeros(n, 1, device=device))
        Tb = ph.bulk_temperature(model, re, pr, 0.0, 50000, device)
        prof = T.cpu().numpy().ravel() / Tb
        ax.plot(xi / dh_dns, prof, "-", label=f"PINN Pr={pr:g}")
        ax.plot(g["x"], g["y"], "o", ms=3, label=f"DNS Pr={pr:g}")
        pred_at = np.interp(g["x"], xi / dh_dns, prof)
        out.append({"figure": "12a", "case": f"Pr={pr:g}", "n": len(g), "rmse": rmse(pred_at, g["y"])})
    ax.set_xlabel("xi / Dh (DNS Dh = %.4f m)" % dh_dns)
    ax.set_ylabel("Theta / Theta_b (iso-temperature)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "dns_fig12a_temperature.png", dpi=130)
    plt.close(fig)
    return out


def evaluate(cfg):
    """Evaluate from a config dict; returns (cfd_fit, nusselt, dns_figures) tables. Also used by the notebook."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    plots = Path(cfg["paths"]["plots_dir"])
    plots.mkdir(parents=True, exist_ok=True)
    re = float(cfg["flow"]["re_values"][0])
    pd.set_option("display.width", 140)

    _, df = load_cfd_profiles(cfg["paths"]["cfd_profiles"], device)
    if df is not None and not df.empty:
        fit = evaluate_cfd_fit(model, df, device, plots)
        print("\n=== Fit to training CFD (rmse_rel = RMSE / max|CFD|) ===")
        print(fit.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        print("No CFD profiles found - skipping CFD fit.")

    nu_table = evaluate_nusselt(model, cfg, device, plots)
    print("\n=== Nusselt number vs. 2023 DNS Table 1 (independent - not used in training) ===")
    print("Note: PINN Nu uses the unit-cell Dh (0.0785 m); the DNS domain's Dh is 0.0712 m.")
    print(nu_table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    nu_table.to_csv(plots.parent / "nusselt_comparison.csv", index=False)

    digit = load_dns_digitized(cfg["paths"]["digitized_dir"])
    results = []
    if "fig6_wall_shear" in digit:
        results += compare_fig6(model, digit["fig6_wall_shear"], re, device, plots)
    if "fig7_velocity_wall_units" in digit:
        results += compare_fig7(model, digit["fig7_velocity_wall_units"], re, device, plots)
    if "fig11a_wall_heat_flux_isoT" in digit:
        results += compare_fig11a(model, digit["fig11a_wall_heat_flux_isoT"], re, device, plots)
    if "fig12a_temperature_isoT" in digit:
        results += compare_fig12a(model, digit["fig12a_temperature_isoT"], re,
                                  cfg["geometry"]["dh_dns"], device, plots)
    if results:
        print("\n=== Against digitized DNS figures (independent - not used in training) ===")
        print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    else:
        print("\nNo digitized DNS figures found yet - see digitized_data/README.md.")
    print(f"\nPlots written to {plots}/")
    return (fit if df is not None and not df.empty else None), nu_table, pd.DataFrame(results)


def main(cfg_path):
    return evaluate(yaml.safe_load(Path(cfg_path).read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
