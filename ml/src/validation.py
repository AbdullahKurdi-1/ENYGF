"""Independent validation against the 2023 DNS (Mathur et al.), in three
layers kept separate (docs/research_plan.md):

    CFD vs DNS    - error of the RANS turbulence model (k-omega SST)
    PINN vs CFD   - error of the machine learning
    PINN vs DNS   - total

DNS data: the curves digitized from the paper (digitized_data/dns_*.csv,
made by digitized_data/digitize_dns_figures.py). The DNS is never used in
training.

Figures compared (only quantities a RANS model can predict):
  Fig. 6    wall shear around the rod, tau_w / tau_w,m
  Fig. 7    axial velocity in wall units U+(r+) at 0, 15, 45 deg (local u_tau)
  Fig. 9a   turbulent kinetic energy k+ = (uu+vv+ww)/2 / u_tau^2 along the
            unit-cell path xi (RANS k is the model's estimate of the same k)
  Fig. 11a  iso-temperature wall heat flux phi / phi_m around the rod
  Fig. 12a  iso-temperature excess temperature Theta / Theta_b along xi

The DNS domain has walls at +-90 deg; the DNS normalises Figs. 6 and 11a by
the mean over -45..45 deg (its "effective unit cell"). We compare only
|theta| <= 45 deg, folded onto 0..45 (the rest is influenced by the DNS side
walls, which our infinite-array cell does not have).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import physics as ph

DNS_FILES = {
    "fig6": "dns_fig6_wall_shear.csv",
    "fig7": "dns_fig7_velocity_wall_units.csv",
    "fig9a": "dns_fig9a_normal_stresses.csv",
    "fig11a": "dns_fig11a_wall_heat_flux_isoT.csv",
    "fig12a": "dns_fig12a_temperature_isoT.csv",
}
LINE_ANGLE = {"seg1": 0.0, "line15": 15.0, "seg3": 45.0}


def _col(v):
    return torch.tensor(np.asarray(v, dtype=np.float32)).unsqueeze(1)


def fold(theta_deg):
    """Angle from the nearest narrow gap, 0..45 deg."""
    t = np.abs(np.asarray(theta_deg, float)) % 90.0
    return np.minimum(t, 90.0 - t)


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


def load_dns(digitized_dir):
    out = {}
    for key, name in DNS_FILES.items():
        p = Path(digitized_dir) / name
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                out[key] = df
    return out


class CFD:
    """The CFD results the PINN was trained on, in the DNS's normalisations."""

    def __init__(self, cfg):
        g = cfg["geometry"]
        self.r, self.a = g["rod_radius"], g["half_pitch"]
        self.nu = cfg["flow"]["u_bulk"] * g["dh_dns"] / cfg["flow"]["re_values"][0]
        self.field = pd.read_csv(cfg["paths"]["cfd_field"])
        p = cfg["paths"].get("cfd_profiles")
        self.lines = pd.read_csv(p) if p and Path(p).exists() else None
        wall = self.field[(self.field["kind"] == "wall") & (self.field["quantity"] == "tau_w")]
        self.tau_theta = np.degrees(np.arctan2(wall["y"], wall["x"])).to_numpy()
        self.tau = wall["value"].to_numpy()
        self.u_tau_mean = math.sqrt(self.tau.mean())

    def local_u_tau(self, theta):
        f = fold(self.tau_theta)
        order = np.argsort(f)
        return math.sqrt(np.interp(theta, f[order], self.tau[order]))

    def wall_distribution(self, quantity, pr=None):
        w = self.field[(self.field["kind"] == "wall") & (self.field["quantity"] == quantity)]
        if pr is not None:
            w = w[np.isclose(w["Pr"], pr)]
        th = fold(np.degrees(np.arctan2(w["y"], w["x"])))
        v = w["value"].to_numpy()
        order = np.argsort(th)
        return th[order], v[order] / v.mean()          # faces have equal arc length

    def bulk_excess_temperature(self, pr):
        c = self.field[self.field["kind"] == "cell"]
        w = c[c["quantity"] == "w"].set_index(["x", "y"])["value"]
        t = c[(c["quantity"] == "T") & np.isclose(c["Pr"], pr) & (c["bc"] == "isoT")]
        t = t.set_index(["x", "y"])
        a = t["area"]
        ww = w.reindex(t.index).to_numpy()
        return float((ww * t["value"] * a).sum() / (ww * a).sum())

    def path(self, quantity, case_id=None):
        if self.lines is None:
            return None, None
        g = self.lines[self.lines["line"].isin(["seg1", "seg2", "seg3"]) & (self.lines["quantity"] == quantity)]
        if case_id is not None:
            g = g[g["case_id"] == case_id]
        g = g.sort_values("xi")
        return g["xi"].to_numpy(), g["value"].to_numpy()

    def wall_line(self, line):
        """(distance from the rod, w) along a wall-normal sampled line."""
        g = self.lines[(self.lines["line"] == line) & (self.lines["quantity"] == "w")].sort_values("s")
        d = np.hypot(g["x"], g["y"]).to_numpy() - self.r
        return d, g["value"].to_numpy()


def _pinn_path_k(model, re, dh):
    from evaluate import unit_cell_boundary_path
    x, y, xi = unit_cell_boundary_path(model)
    with torch.no_grad():
        _, _, k = model.momentum(_col(x), _col(y), torch.full((len(x), 1), re))
    return xi / dh, k.numpy().ravel()


def validate(cfg, model=None, plots=None, show=False):
    """Returns a table of RMSE per figure/case: CFD vs DNS, PINN vs DNS, PINN vs CFD."""
    from evaluate import ray_points, unit_cell_boundary_path, wall_shear_distribution
    dns = load_dns(cfg["paths"]["digitized_dir"])
    if not dns:
        print("No digitized DNS data found in", cfg["paths"]["digitized_dir"])
        return pd.DataFrame()
    cfd = CFD(cfg)
    re = float(cfg["flow"]["re_values"][0])
    dh = cfg["geometry"]["dh_dns"]
    plots = Path(plots or cfg["paths"]["plots_dir"])
    plots.mkdir(parents=True, exist_ok=True)
    rows = []

    def row(fig, case, x_dns, y_dns, x_cfd, y_cfd, x_pinn=None, y_pinn=None, note=""):
        order = np.argsort(x_cfd)
        c_at = np.interp(x_dns, np.asarray(x_cfd)[order], np.asarray(y_cfd)[order], left=np.nan, right=np.nan)
        r = {"figure": fig, "case": case, "n_dns": len(x_dns), "CFD_vs_DNS": rmse(c_at, y_dns)}
        if x_pinn is not None:
            o = np.argsort(x_pinn)
            p_at = np.interp(x_dns, np.asarray(x_pinn)[o], np.asarray(y_pinn)[o], left=np.nan, right=np.nan)
            p_cfd = np.interp(np.asarray(x_cfd)[order], np.asarray(x_pinn)[o], np.asarray(y_pinn)[o])
            r.update(PINN_vs_DNS=rmse(p_at, y_dns), PINN_vs_CFD=rmse(p_cfd, np.asarray(y_cfd)[order]))
        r["note"] = note
        rows.append(r)

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=130)
        if show:
            plt.show()
        plt.close(fig)

    style = dict(dns=dict(color="k", marker="o", ms=3, ls="none", label="DNS (Mathur 2023)"),
                 cfd=dict(color="tab:blue", lw=1.8, label="CFD (RANS k-omega SST)"),
                 pinn=dict(color="tab:red", lw=1.2, ls="--", label="PINN"))

    # Fig. 6 - wall shear
    if "fig6" in dns:
        d = dns["fig6"][dns["fig6"]["x"].abs() <= 45]
        th_c, v_c = cfd.wall_distribution("tau_w")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(fold(d["x"]), d["y"], **style["dns"])
        ax.plot(th_c, v_c, **style["cfd"])
        xp = yp = None
        if model is not None:
            xp, yp, _ = wall_shear_distribution(model, re, "cpu")
            xp, yp = fold(xp), yp
            ax.plot(xp, yp, **style["pinn"])
        ax.set(xlabel="angle from narrow gap [deg]", ylabel="tau_w / tau_w,m", title="Wall shear (DNS Fig. 6)")
        ax.legend(fontsize=7)
        finish(fig, "dns_fig6_wall_shear.png")
        row("6 wall shear", "-", fold(d["x"]), d["y"], th_c, v_c, xp, yp)

    # Fig. 7 - U+ vs r+ with the local friction velocity
    if "fig7" in dns and cfd.lines is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (line, ang) in zip(axes, LINE_ANGLE.items()):
            d = dns["fig7"][dns["fig7"]["case_id"] == ang]
            dist, w = cfd.wall_line(line)
            if d.empty or len(dist) < 3:
                ax.set_visible(False)
                continue
            ut = cfd.local_u_tau(ang)
            rp, up = dist * ut / cfd.nu, w / ut
            keep = rp > 0.3
            ax.semilogx(d["x"], d["y"], **style["dns"])
            ax.semilogx(rp[keep], up[keep], **style["cfd"])
            xp = yp = None
            if model is not None:
                dd, x, y = ray_points(model, ang)
                with torch.no_grad():
                    wp, _, _ = model.momentum(_col(x), _col(y), torch.full((len(x), 1), re))
                utp = math.sqrt(max(ph.wall_shear(model, re, [math.radians(ang)], "cpu").item(), 1e-12))
                xp, yp = dd * utp / cfd.nu, wp.numpy().ravel() / utp
                k = (xp > 0.3) & (xp < rp.max())
                xp, yp = xp[k], yp[k]
                ax.semilogx(xp, yp, **style["pinn"])
            ax.set(xlabel="r+", ylabel="U+", title=f"U+ at {ang:g} deg (DNS Fig. 7)")
            ax.legend(fontsize=7)
            row("7 U+(r+)", f"{ang:g} deg", np.log10(d["x"]), d["y"], np.log10(rp[keep]), up[keep],
                None if xp is None else np.log10(xp), yp, note=f"CFD u_tau({ang:g})={ut:.4f}")
        finish(fig, "dns_fig7_velocity_wall_units.png")

    # Fig. 9a - turbulent kinetic energy along the unit-cell path
    if "fig9a" in dns and cfd.lines is not None:
        p = dns["fig9a"].pivot_table(index="x", columns="case_id", values="y").dropna()
        k_dns = 0.5 * p.sum(axis=1)
        xi, k = cfd.path("k")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(k_dns.index, k_dns.values, **style["dns"])
        ax.plot(xi / dh, k / cfd.u_tau_mean**2, **style["cfd"])
        xp = yp = None
        if model is not None:
            xp, kp = _pinn_path_k(model, re, dh)
            utp = math.sqrt(ph.pressure_gradient_from_wall_shear(model, re, "cpu") * model.dh / 4)
            yp = kp / utp**2
            ax.plot(xp, yp, **style["pinn"])
        for v in (0.105, 1.194):
            ax.axvline(v, color="0.7", ls=":", lw=0.8)
        ax.set(xlabel="xi / Dh", ylabel="k+ = k / u_tau^2", title="Turbulent kinetic energy (DNS Fig. 9a)")
        ax.legend(fontsize=7)
        finish(fig, "dns_fig9a_tke.png")
        row("9a k+", "-", k_dns.index.to_numpy(), k_dns.values, xi / dh, k / cfd.u_tau_mean**2, xp, yp,
            note="k+ with each model's own mean u_tau")

    # Fig. 11a - iso-temperature wall heat flux
    if "fig11a" in dns:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, (pr, d) in enumerate(dns["fig11a"].groupby("case_id")):
            d = d[d["x"].abs() <= 45]
            th_c, v_c = cfd.wall_distribution("q_w", pr)
            c = f"C{i}"
            ax.plot(fold(d["x"]), d["y"], "o", color=c, ms=3, label=f"DNS Pr={pr:g}")
            ax.plot(th_c, v_c, "-", color=c, lw=1.8, label=f"CFD Pr={pr:g}")
            xp = yp = None
            if model is not None:
                ang = np.linspace(0, math.pi / 4, 91)
                _, fl = ph.wall_temperature_data(model, re, pr, 0.0, ang, "cpu")
                fl = fl.numpy().ravel()
                xp, yp = np.degrees(ang), fl / fl.mean()
                ax.plot(xp, yp, "--", color=c, lw=1.1, label=f"PINN Pr={pr:g}")
            row("11a wall heat flux", f"Pr={pr:g}", fold(d["x"]), d["y"], th_c, v_c, xp, yp)
        ax.set(xlabel="angle from narrow gap [deg]", ylabel="phi / phi_m (iso-T)",
               title="Wall heat flux (DNS Fig. 11a)")
        ax.legend(fontsize=6, ncol=3)
        finish(fig, "dns_fig11a_wall_heat_flux.png")

    # Fig. 12a - excess temperature along the unit-cell path
    if "fig12a" in dns and cfd.lines is not None:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for i, (pr, d) in enumerate(dns["fig12a"].groupby("case_id")):
            case = f"Pr{pr:g}_isoT".replace(".", "p")
            xi, t = cfd.path("T", case)
            if len(xi) < 3:
                continue
            tb = cfd.bulk_excess_temperature(pr)
            c = f"C{i}"
            ax.plot(d["x"], d["y"], "o", color=c, ms=2.5, label=f"DNS Pr={pr:g}")
            ax.plot(xi / dh, t / tb, "-", color=c, lw=1.8, label=f"CFD Pr={pr:g}")
            xp = yp = None
            if model is not None:
                x, y, xip = unit_cell_boundary_path(model)
                n = len(x)
                with torch.no_grad():
                    tp = model.temperature(_col(x), _col(y), torch.full((n, 1), re), torch.full((n, 1), pr),
                                           torch.zeros(n, 1)).numpy().ravel()
                tbp = ph.bulk_temperature(model, re, pr, 0.0, 50000, "cpu")
                xp, yp = xip / dh, tp / tbp
                ax.plot(xp, yp, "--", color=c, lw=1.1, label=f"PINN Pr={pr:g}")
            row("12a Theta/Theta_b", f"Pr={pr:g}", d["x"], d["y"], xi / dh, t / tb, xp, yp)
        for v in (0.105, 1.194):
            ax.axvline(v, color="0.7", ls=":", lw=0.8)
        ax.set(xlabel="xi / Dh", ylabel="Theta / Theta_b (iso-T)", title="Excess temperature (DNS Fig. 12a)")
        ax.legend(fontsize=6, ncol=3)
        finish(fig, "dns_fig12a_temperature.png")

    table = pd.DataFrame(rows)
    print("\n=== Against the DNS figures (RMSE in the figure's own units; DNS never used in training) ===")
    print("CFD_vs_DNS = turbulence-model error, PINN_vs_CFD = machine-learning error, PINN_vs_DNS = total")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


def key_points(cfg):
    """Headline numbers, CFD vs DNS, at the places the papers discuss."""
    dns = load_dns(cfg["paths"]["digitized_dir"])
    cfd = CFD(cfg)
    dh = cfg["geometry"]["dh_dns"]
    out = []
    if "fig6" in dns:
        d = dns["fig6"][dns["fig6"]["x"].abs() <= 45]
        th, v = cfd.wall_distribution("tau_w")
        for ang in (0, 45):
            dv = np.interp(ang, *zip(*sorted(zip(fold(d["x"]), d["y"]))))
            out.append(("wall shear tau/tau_m", f"{ang} deg", np.interp(ang, th, v), dv))
    if "fig9a" in dns and cfd.lines is not None:
        p = dns["fig9a"].pivot_table(index="x", columns="case_id", values="y").dropna()
        k_dns = 0.5 * p.sum(axis=1)
        xi, k = cfd.path("k")
        for x, lab in ((0.105, "gap centre"), (0.6, "xi/Dh=0.6"), (1.194, "subchannel centre")):
            out.append(("k+", lab, np.interp(x * dh, xi, k) / cfd.u_tau_mean**2, np.interp(x, k_dns.index, k_dns.values)))
    if "fig12a" in dns and cfd.lines is not None:
        for pr, d in dns["fig12a"].groupby("case_id"):
            xi, t = cfd.path("T", f"Pr{pr:g}_isoT".replace(".", "p"))
            tb = cfd.bulk_excess_temperature(pr)
            for x, lab in ((0.105, "gap centre"), (1.194, "subchannel centre")):
                out.append((f"Theta/Theta_b Pr={pr:g}", lab, np.interp(x * dh, xi, t) / tb, np.interp(x, d["x"], d["y"])))
    out.append(("mean u_tau [m/s]", "-", cfd.u_tau_mean, 0.0637))
    t = pd.DataFrame(out, columns=["quantity", "where", "CFD", "DNS"])
    t["CFD/DNS"] = t["CFD"] / t["DNS"]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return t


def physics_checks(cfg):
    """Sanity checks of the CFD against conservation laws and textbook
    correlations. Nothing here is fitted; the correlations are for round tubes
    or generic geometries, so ~10% differences are expected for a tight
    bundle - the point is the order of magnitude and the trends."""
    cfd = CFD(cfg)
    g = cfg["geometry"]
    re = cfg["flow"]["re_values"][0]
    nu = cfd.nu
    rows = []
    c = cfd.field[cfd.field["kind"] == "cell"]
    w = c[c["quantity"] == "w"]
    ub = float((w["value"] * w["area"]).sum() / w["area"].sum())
    rows.append(("bulk velocity U_b", ub, 1.0, "mass conservation (imposed)"))
    area = g["half_pitch"] ** 2 - math.pi * g["rod_radius"] ** 2 / 4
    G = cfd.tau.mean() * (math.pi * g["rod_radius"] / 2) / area
    f_cfd = 8 * cfd.u_tau_mean ** 2 / ub ** 2
    rows.append(("friction factor f = 8 u_tau^2/U_b^2", f_cfd, 0.316 * re ** -0.25, "Blasius (smooth tube)"))
    rows.append(("  same, 2023 DNS", 8 * 0.0637 ** 2, 0.316 * re ** -0.25, "Blasius (smooth tube)"))
    rows.append(("pressure gradient from wall shear", G, None, "compare with the solver log"))
    # viscous sublayer and log law on the 45 deg line (largest r+ range)
    if cfd.lines is not None:
        dist, wl = cfd.wall_line("seg3")
        ut = cfd.local_u_tau(45.0)
        rp, up = dist * ut / nu, wl / ut
        o = np.argsort(rp)
        for r_target in (2.0,):
            rows.append((f"U+ at r+ = {r_target:g} (45 deg)", np.interp(r_target, rp[o], up[o]), r_target,
                         "viscous sublayer U+ = r+"))
        for r_target in (50.0, 150.0):
            rows.append((f"U+ at r+ = {r_target:g} (45 deg)", np.interp(r_target, rp[o], up[o]),
                         math.log(r_target) / 0.41 + 5.2, "log law U+ = ln(r+)/0.41 + 5.2"))
    # Nusselt numbers vs correlations
    p = cfg["paths"].get("cfd_nusselt")
    if p and Path(p).exists():
        nus = pd.read_csv(p)
        for r in nus.itertuples():
            pe = re * r.Pr
            if r.Pr < 0.1:
                ref, name = (5.0 + 0.025 * pe ** 0.8, "Seban-Shimazaki (liquid metal, uniform T_w)") if r.bc == "isoT" \
                    else (7.0 + 0.025 * pe ** 0.8, "Lyon (liquid metal, uniform q)")
            else:
                ref, name = 0.023 * re ** 0.8 * r.Pr ** 0.4, "Dittus-Boelter (tube)"
            rows.append((f"Nu Pr={r.Pr:g} {r.bc}", r.Nu_CFD, ref, name))
    t = pd.DataFrame(rows, columns=["check", "CFD", "reference", "reference is"])
    t["CFD/reference"] = t["CFD"] / t["reference"]
    print("Physics sanity checks (CFD vs conservation laws and textbook correlations):")
    with pd.option_context("display.width", 160):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return t
