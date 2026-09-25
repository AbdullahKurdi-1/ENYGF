"""Data loading.

Training data, written by cfd/openfoam/extract_profiles.py into
digitized_data/cfd_generated/ (physical units, 2023 DNS normalisation U_b = 1,
rho*cp = 1):
  cfd_field.csv     every mesh cell (preferred): w, nu_t, k and all 8 T fields
  cfd_profiles.csv  the sampled lines only (older pipeline / plotting)
A fixed random fraction of the cells is held out of training to measure the
PINN's error on points it has never seen.

Validation data: points digitized from the 2023 DNS paper's figures
(digitized_data/dns_*.csv). These are deliberately NOT used for training -
they are the independent check.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BC_CODE = {"isoT": 0.0, "isoFlux": 1.0}

DNS_FILES = {
    "fig6_wall_shear": "dns_fig6_wall_shear.csv",
    "fig7_velocity_wall_units": "dns_fig7_velocity_wall_units.csv",
    "fig11a_wall_heat_flux_isoT": "dns_fig11a_wall_heat_flux_isoT.csv",
    "fig12a_temperature_isoT": "dns_fig12a_temperature_isoT.csv",
}


def load_cfd_profiles(path, device, lines=None):
    """Returns {quantity: dict of (N, 1) tensors x, y, re, pr, bc, value} plus the raw frame.
    lines: optional list of line names to keep (e.g. ["seg1", "seg2"])."""
    path = Path(path)
    if not path.exists():
        return {}, None
    df = pd.read_csv(path)
    if lines is not None:
        df = df[df["line"].isin(lines)]
    if df.empty:
        return {}, df
    out = {}
    for q, g in df.groupby("quantity"):
        col = lambda s: torch.tensor(np.asarray(s, dtype=np.float32), device=device).unsqueeze(1)
        pr = g["Pr"].fillna(1.0)
        bc = g["bc"].map(BC_CODE).fillna(0.0)
        out[q] = {
            "x": col(g["x"]), "y": col(g["y"]), "re": col(g["Re"]),
            "pr": col(pr), "bc": col(bc), "value": col(g["value"]),
        }
    return out, df


def _tensors(g, device):
    col = lambda v: torch.tensor(np.asarray(v, dtype=np.float32), device=device).unsqueeze(1)
    return {
        "x": col(g["x"]), "y": col(g["y"]), "re": col(g["Re"]),
        "pr": col(g["Pr"].fillna(1.0)), "bc": col(g["bc"].map(BC_CODE).fillna(0.0)),
        "value": col(g["value"]),
    }


def load_cfd_field(path, device, holdout_fraction=0.2, seed=0, data_fraction=1.0):
    """Returns (train, test, frame). train/test: {quantity: tensors}.

    The same cells are held out for every quantity, and the held-out set does
    not depend on data_fraction, so runs with different amounts of data are
    tested on identical points. data_fraction < 1 keeps only that fraction of
    all cell locations (and of the rod-surface locations) for training - the
    sparse-measurement experiment; the rest is marked 'unused'."""
    path = Path(path)
    if not path.exists():
        return {}, {}, None
    df = pd.read_csv(path)
    cells = df[df["kind"] == "cell"][["x", "y"]].drop_duplicates().round(9)
    rng = np.random.default_rng(seed)
    test_idx = rng.random(len(cells)) < holdout_fraction
    test_keys = set(map(tuple, cells[test_idx].to_numpy()))

    pick = np.random.default_rng(seed + 1)
    pool = cells[~test_idx].to_numpy()
    n_keep = min(len(pool), int(round(data_fraction * len(cells))))
    keep = set(map(tuple, pool[pick.choice(len(pool), n_keep, replace=False)])) if n_keep else set()
    walls = df[df["kind"] == "wall"][["x", "y"]].drop_duplicates().round(9).to_numpy()
    n_wall = int(round(data_fraction * len(walls)))
    keep |= set(map(tuple, walls[pick.choice(len(walls), n_wall, replace=False)])) if n_wall else set()

    key = list(map(tuple, df[["x", "y"]].round(9).to_numpy()))
    df["split"] = ["test" if (k in test_keys and kind == "cell") else ("train" if k in keep else "unused")
                   for k, kind in zip(key, df["kind"])]
    train, test = {}, {}
    for (q, split), g in df.groupby(["quantity", "split"]):
        if split != "unused":
            (train if split == "train" else test)[q] = _tensors(g, device)
    return train, test, df


def temperature_scales(df, pr_values):
    """(n_pr, 2) array of the CFD temperature range per case, iso-T / iso-flux,
    rows in sorted-Pr order - used to put every case on an O(1) scale. Pass
    only the training rows. NaN where a case has fewer than 2 points."""
    out = np.full((len(pr_values), 2), np.nan)
    t = df[df["quantity"] == "T"]
    for i, pr in enumerate(sorted(pr_values)):
        for j, bc in enumerate(("isoT", "isoFlux")):
            g = t[np.isclose(t["Pr"], pr) & (t["bc"] == bc)]["value"]
            if len(g) >= 2:
                out[i, j] = max(float(g.max() - g.min()), 1e-12)
    return out


def load_dns_digitized(digitized_dir):
    """Returns {name: DataFrame(case_id, x, y)} for every non-empty DNS file present."""
    out = {}
    for name, fname in DNS_FILES.items():
        p = Path(digitized_dir) / fname
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                out[name] = df
    return out
