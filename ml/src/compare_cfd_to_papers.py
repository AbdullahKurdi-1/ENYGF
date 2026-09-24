"""Compare the CFD training data with the two reference papers at a few
key points along the DNS unit-cell boundary path (xi).

The paper values below were READ BY EYE from the published figures
(roughly +-5-10 %). They are for a first sanity check only; the proper
comparison uses digitized curves (digitized_data/README.md) and the trained
PINN (evaluate.py).

2023 DNS velocities are only given in wall units (Fig. 7), so they are
converted with the local friction velocity from Fig. 6:
  gap centre:       U+ ~ 15.3 at r+ ~ 55, tau/tau_m(0 deg) ~ 0.82 -> w ~ 0.85 U_b
  subchannel centre: U+ ~ 19.5 at r+ ~ 380 on the 45 deg line   -> w ~ 1.36 U_b
DNS k = 0.5 * (<uu>+ + <vv>+ + <ww>+) * u_tau^2 from Fig. 9(a), u_tau = 0.0637.

Usage (from ml/):  python3 src/compare_cfd_to_papers.py
Notebook:          import compare_cfd_to_papers; compare_cfd_to_papers.main(cfg)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

U_TAU_DNS = 0.0637

# (label, 2023 DNS, 2018 URANS) - None where the paper has no comparable number
REFERENCE = {
    "w_gap": ("axial velocity at narrow-gap centre [U_b]", 0.85, 0.55),
    "w_centre": ("axial velocity at subchannel centre [U_b]", 1.36, None),
    "tau_ratio": ("wall shear, gap (0 deg) / 45 deg", 0.82 / 1.19, None),
    # No 2018 value: its Fig. 12 gradient gives u_tau/U_b ~ 0.045, identical to
    # its own rod-averaged Re_tau/Re (439/9800), so it can't be read as a gap value.
    "ut_gap": ("friction velocity at gap wall / U_b", U_TAU_DNS * 0.82**0.5, None),
    "k_gap_wall": ("k, near-wall peak in the gap [m2/s2]", 3.15 * U_TAU_DNS**2, None),
    "k_gap": ("k at narrow-gap centre [m2/s2]", 2.48 * U_TAU_DNS**2, None),
    "k_mid": ("k at xi/Dh = 0.6 [m2/s2]", 2.43 * U_TAU_DNS**2, None),
    "k_centre": ("k at subchannel centre [m2/s2]", 1.03 * U_TAU_DNS**2, None),
    "k_45_wall": ("k, near-wall peak at 45 deg [m2/s2]", 5.4 * U_TAU_DNS**2, None),
}
# iso-T: Theta(gap centre) / Theta(subchannel centre), Fig. 12(a);
# wall heat flux q(0 deg) / q(45 deg), Fig. 11(a)
THETA_RATIO_DNS = {0.025: 0.18 / 2.32, 1.0: 0.88 / 1.32, 2.0: 0.93 / 1.22}
FLUX_RATIO_DNS = {0.025: 0.40 / 1.50, 1.0: 0.80 / 1.17, 2.0: 0.80 / 1.16}


def profile(path, quantity, case_id=None):
    g = path[path["quantity"] == quantity]
    if case_id is not None:
        g = g[g["case_id"] == case_id]
    g = g.sort_values("xi")
    return g["xi"].to_numpy(float), g["value"].to_numpy(float)


def wall_slopes(xi, v):
    """|dv/dn| at the gap wall (first points) and the 45 deg wall (last points)."""
    return abs((v[1] - v[0]) / (xi[1] - xi[0])), abs((v[-2] - v[-1]) / (xi[-1] - xi[-2]))


def main(cfg):
    geo = cfg["geometry"]
    r, a = geo["rod_radius"], geo["half_pitch"]
    dh = geo["dh_dns"]
    nu = cfg["flow"]["u_bulk"] * geo["dh_cell"] / cfg["flow"]["re_values"][0]
    xi_gap, xi_centre = a - r, a - r + a

    df = pd.read_csv(cfg["paths"]["cfd_profiles"])
    path = df[df["line"].isin(["seg1", "seg2", "seg3"])]

    xi, w = profile(path, "w")
    _, k = profile(path, "k")
    s = xi / dh
    tau_gap, tau_45 = (nu * g for g in wall_slopes(xi, w))
    cfd = {
        "w_gap": np.interp(xi_gap, xi, w),
        "w_centre": np.interp(xi_centre, xi, w),
        "tau_ratio": tau_gap / tau_45,
        "ut_gap": tau_gap**0.5,
        "k_gap_wall": k[s < 0.07].max(),
        "k_gap": np.interp(xi_gap, xi, k),
        "k_mid": np.interp(0.6 * dh, xi, k),
        "k_centre": np.interp(xi_centre, xi, k),
        "k_45_wall": k[s > 1.5].max(),
    }
    rows = []
    for key, (label, dns, urans) in REFERENCE.items():
        rows.append((label, cfd[key], dns, urans, cfd[key] / dns))

    for pr in sorted(path["Pr"].dropna().unique()):
        case = f"Pr{pr:g}_isoT".replace(".", "p")
        xi_t, t = profile(path, "T", case)
        if len(t) == 0:
            continue
        ratio = np.interp(xi_gap, xi_t, t) / np.interp(xi_centre, xi_t, t)
        q_gap, q_45 = wall_slopes(xi_t, t)
        dns_t, dns_q = THETA_RATIO_DNS.get(pr), FLUX_RATIO_DNS.get(pr)
        rows.append((f"iso-T Pr={pr:g}: Theta gap centre / subchannel centre", ratio, dns_t, None,
                     ratio / dns_t if dns_t else None))
        rows.append((f"iso-T Pr={pr:g}: wall heat flux gap / 45 deg", q_gap / q_45, dns_q, None,
                     q_gap / q_45 / dns_q if dns_q else None))

    table = pd.DataFrame(rows, columns=["quantity", "our CFD", "2023 DNS", "2018 URANS", "CFD / DNS"])
    with pd.option_context("display.max_colwidth", 60, "display.width", 140, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False, na_rep="-"))
    print("\nPaper values read by eye from the figures (+-5-10 %). Wall slopes at 45 deg use "
          "sample spacing of ~3.5 wall units, so heat-flux ratios for Pr >= 2 are rough.")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(yaml.safe_load(Path(parser.parse_args().config).read_text()))
