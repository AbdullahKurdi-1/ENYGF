"""Physics-based explanations of the trained PINN (docs/research_plan.md).

Standard XAI tools (SHAP, LIME) rank input features; here the inputs are just
the coordinates (x, y), so "which input mattered" has no physical meaning.
These explanations use the physics instead:

1. residual_maps     where the network violates the momentum and energy
                     equations - i.e. where not to trust it.
2. energy_budget     which mechanism carries the heat: molecular conduction
                     (nu/Pr) or turbulent mixing (nu_t/Pr_t), mapped over the
                     cell - the model's own explanation of its Nusselt numbers.
3. sensor_importance which measurement line matters most (from the sparse-data
                     runs in experiments.py) - where sensors are most valuable.

Notebook:
    import xai
    model = xai.load_trained(cfg)
    xai.residual_maps(model, cfg)
    xai.energy_budget(model, cfg)
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch

import physics as ph
from evaluate import load_model, unit_cell_boundary_path

BCS = {"isoT": 0.0, "isoFlux": 1.0}


def load_trained(cfg, device="cpu"):
    return load_model(cfg, device)


def fluid_grid(model, n=90):
    """Points of an n x n grid that lie in the fluid, plus the rod-surface
    refinement the thin wall layers need (geometric spacing in wall distance)."""
    a, r = model.a, model.r
    g = np.linspace(0, a, n)
    X, Y = np.meshgrid(g, g)
    x, y = X.ravel(), Y.ravel()
    keep = np.hypot(x, y) > r * (1 + 1e-4)
    phi = np.linspace(0, math.pi / 2, 60)
    d = np.geomspace(2e-5, 3e-3, 12)
    P, D = np.meshgrid(phi, d)
    xw, yw = ((r + D) * np.cos(P)).ravel(), ((r + D) * np.sin(P)).ravel()
    return np.r_[x[keep], xw], np.r_[y[keep], yw]


def _col(v):
    return torch.tensor(np.asarray(v, dtype=np.float32)).unsqueeze(1)


def regions(model, x, y):
    """Labels used to summarise maps: near-wall, narrow gap, subchannel centre, bulk."""
    d = np.hypot(x, y) - model.r
    half_gap = model.a - model.r
    lab = np.full(len(x), "bulk", dtype=object)
    lab[np.hypot(x - model.a, y - model.a) < 0.02] = "subchannel centre"
    lab[(np.minimum(x, y) < half_gap) & (d >= 1e-3)] = "narrow gap"
    lab[d < 1e-3] = "near wall (< 1 mm)"
    return lab


def _map(ax, model, x, y, v, title, cmap="viridis", log=False, vmin=None, vmax=None):
    tri = mtri.Triangulation(x, y)
    xm, ym = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    tri.set_mask(np.hypot(xm, ym) < model.r)
    vals = np.log10(np.maximum(v, 1e-6)) if log else v
    levels = np.linspace(vmin, vmax, 21) if vmin is not None else 30
    if vmin is not None:
        vals = np.clip(vals, vmin, vmax)
    cs = ax.tricontourf(tri, vals, levels=levels, cmap=cmap)
    ax.add_patch(plt.Circle((0, 0), model.r, color="0.8"))
    ax.set_aspect("equal")
    ax.set_xlim(0, model.a)
    ax.set_ylim(0, model.a)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(cs, ax=ax, fraction=0.046)
    if log:
        cb.set_label("log10", fontsize=8)


def residual_maps(model, cfg, cases=((0.025, "isoT"), (1.0, "isoT"), (7.0, "isoT"), (1.0, "isoFlux")),
                  out=None):
    """|momentum residual| / G and |energy residual| / sink over the cell.
    Returns an RMS table by region."""
    re = float(cfg["flow"]["re_values"][0])
    x, y = fluid_grid(model)
    lab = regions(model, x, y)
    X, Y, R = _col(x), _col(y), torch.full((len(x), 1), re)
    maps = {"momentum": (ph.momentum_residual(model, X.clone(), Y.clone(), R) * model.g_scale
                         / model.pressure_gradient(R)).detach().abs().numpy().ravel()}
    for pr, bc in cases:
        res = ph.energy_residual(model, X.clone(), Y.clone(), R, torch.full_like(X, pr), torch.full_like(X, BCS[bc]))
        maps[f"energy Pr={pr:g} {bc}"] = res.detach().abs().numpy().ravel()

    fig, axes = plt.subplots(1, len(maps), figsize=(3.6 * len(maps), 3.4))
    for ax, (name, v) in zip(axes, maps.items()):
        _map(ax, model, x, y, v, f"|residual|, {name}", cmap="magma", log=True, vmin=-3, vmax=1)
    fig.suptitle("Where the PINN breaks the equations (log10 of residual relative to the driving term; "
                 "-2 = 1%, 0 = 100%)", fontsize=10)
    fig.tight_layout()
    out = Path(out or Path(cfg["paths"]["plots_dir"]) / "xai_residual_maps.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.show()
    plt.close(fig)

    rows = []
    for name, v in maps.items():
        row = {"equation": name}
        for reg in ["near wall (< 1 mm)", "narrow gap", "subchannel centre", "bulk"]:
            m = lab == reg
            row[reg] = float(np.sqrt(np.mean(v[m] ** 2))) if m.any() else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    print("RMS residual by region (1.0 = as large as the driving term G or the heat sink S):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


def energy_budget(model, cfg, out=None):
    """Share of the heat transport carried by turbulence, alpha_t / (alpha_m + alpha_t)
    with alpha_m = nu/Pr and alpha_t = nu_t/Pr_t. Both act on the same temperature
    gradient, so this ratio is exactly the turbulent fraction of the local heat flux."""
    re = float(cfg["flow"]["re_values"][0])
    prs = sorted(cfg["thermal"]["pr_values"])
    x, y = fluid_grid(model)
    lab = regions(model, x, y)
    X, Y, R = _col(x), _col(y), torch.full((len(x), 1), re)
    nu = float(model.nu(torch.tensor(re)))
    with torch.no_grad():
        _, nut, _ = model.momentum(X, Y, R)
    nut = nut.numpy().ravel()

    # 1. maps
    fig, axes = plt.subplots(1, len(prs), figsize=(3.6 * len(prs), 3.4))
    shares = {}
    for ax, pr in zip(axes, prs):
        at, am = nut / model.pr_t, nu / pr
        shares[pr] = at / (am + at)
        _map(ax, model, x, y, shares[pr], f"turbulent share of heat flux, Pr = {pr:g}", cmap="coolwarm",
             vmin=0, vmax=1)
    fig.suptitle("Which mechanism carries the heat? 0 = molecular conduction only, 1 = turbulent mixing only",
                 fontsize=10)
    fig.tight_layout()
    out = Path(out or Path(cfg["paths"]["plots_dir"]) / "xai_energy_budget_maps.png")
    fig.savefig(out, dpi=130)
    plt.show()
    plt.close(fig)

    # 2. along the DNS unit-cell path
    px, py, xi = unit_cell_boundary_path(model)
    with torch.no_grad():
        _, nut_p, _ = model.momentum(_col(px), _col(py), torch.full((len(px), 1), re))
    nut_p = nut_p.numpy().ravel()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for pr in prs:
        at = nut_p / model.pr_t
        ax.plot(xi / cfg["geometry"]["dh_dns"], at / (nu / pr + at), label=f"Pr = {pr:g}")
    for v in (0.105, 1.194):
        ax.axvline(v, color="0.6", ls="--", lw=0.8)
    ax.text(0.02, 1.02, "gap", fontsize=8)
    ax.text(1.0, 1.02, "subchannel centre", fontsize=8)
    ax.set_xlabel("xi / Dh  (rod in gap -> gap centre -> subchannel centre -> rod at 45 deg)", fontsize=9)
    ax.set_ylabel("turbulent share of heat flux")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out).with_name("xai_energy_budget_path.png"), dpi=130)
    plt.show()
    plt.close(fig)

    # 3. table: area-averaged share per region
    rows = []
    for pr in prs:
        row = {"Pr": f"{pr:g}"}
        for reg in ["near wall (< 1 mm)", "narrow gap", "subchannel centre", "bulk"]:
            m = lab == reg
            row[reg] = float(shares[pr][m].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    print("Average turbulent share of the heat flux by region (0 = all conduction, 1 = all turbulence):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return table


def sensor_importance(results, reference="lines PINN", out=None):
    """How much the errors grow when one measurement line is removed from the
    lines-only PINN, as a multiple of the error with all lines - computed per
    seed against the reference run with the same seed, then averaged (the
    range over seeds is shown). results: list/DataFrame of experiment results."""
    import re
    df = pd.DataFrame(results).set_index("run")

    def seed_of(run):
        m = re.search(r"\(seed (\d+)\)$", run)
        return int(m.group(1)) if m else 0

    refs = {seed_of(r): df.loc[r] for r in df.index if re.fullmatch(re.escape(reference) + r"( \(seed \d+\))?", r)}
    rows = []
    for run in df.index:
        m = re.fullmatch(re.escape(reference) + r" without (\w+)( \(seed \d+\))?", run)
        if not m or seed_of(run) not in refs:
            continue
        base = refs[seed_of(run)]
        rows.append({"removed line": m.group(1), "seed": seed_of(run),
                     "velocity": df.loc[run, "test_err_w"] / base["test_err_w"],
                     "temperature": df.loc[run, "test_err_T"] / base["test_err_T"],
                     "worst Nu error (%)": 100 * df.loc[run, "Nu_err_max_Pr<=2"],
                     "Nu reference (%)": 100 * base["Nu_err_max_Pr<=2"]})
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        print("No sensor-importance runs found.")
        return per_seed
    order = {name: i for i, name in enumerate(["seg1", "seg2", "seg3", "line15"])}
    g = per_seed.groupby("removed line")
    table = pd.DataFrame({
        "seeds": g.size(),
        "velocity error x (mean)": g["velocity"].mean(), "velocity min": g["velocity"].min(),
        "velocity max": g["velocity"].max(),
        "temperature error x (mean)": g["temperature"].mean(), "temperature min": g["temperature"].min(),
        "temperature max": g["temperature"].max(),
        "worst Nu error % (mean)": g["worst Nu error (%)"].mean(),
    }).reset_index()
    table = table.sort_values("removed line", key=lambda c: c.map(order)).reset_index(drop=True)
    print(f"Reference ({reference}, all 4 lines), mean over {len(refs)} seed(s): worst Nu error "
          f"{np.mean([r['Nu_err_max_Pr<=2'] for r in refs.values()]):.1%}")
    print("Error when one line is removed, as a multiple of the error with all lines (> 1 = the line matters):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs = np.arange(len(table))
    for off, key, lab in ((-0.2, "velocity", "velocity"), (0.2, "temperature", "temperature")):
        mean = table[f"{key} error x (mean)"]
        err = [mean - table[f"{key} min"], table[f"{key} max"] - mean]
        ax.bar(xs + off, mean, 0.4, yerr=err, capsize=3, label=lab)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(xs, [f"without {l}" for l in table["removed line"]])
    ax.set_ylabel("error / error with all lines")
    ax.set_title(f"Sensor importance ({int(table['seeds'].max())} seed(s); bars = range)")
    ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    plt.show()
    plt.close(fig)
    return table
