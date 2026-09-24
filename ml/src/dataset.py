"""Data loading.

Training data: CFD line profiles written by cfd/openfoam/extract_profiles.py
(digitized_data/cfd_generated/cfd_profiles.csv), in physical units with the
2023 DNS normalisation (U_b = 1, rho*cp = 1).

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


def load_cfd_profiles(path, device):
    """Returns {quantity: dict of (N, 1) tensors x, y, re, pr, bc, value} plus the raw frame."""
    path = Path(path)
    if not path.exists():
        return {}, None
    df = pd.read_csv(path)
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
