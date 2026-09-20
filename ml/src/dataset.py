"""Loads paper_reference tables + digitized_data/cfd_generated CSVs into a
single unified set of (x, y, Re, Pr, bc_flag) -> target tensors.

Every profile CSV (whether hand-digitized from the paper's figures or
extracted from your own OpenFOAM run by extract_profiles.py) shares the same
three-column schema: case_id, x, y - where x is the along-line coordinate in
[0, 1] and y is the physical quantity value. This module is what maps that
1D line coordinate back onto the real 2D (x, y) plane of the unit cell, using
the same Line 1 / Line 2 endpoints as cfd/openfoam/unit_cell/system/sampleDict.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FILE_KIND_PATTERNS = {
    "velocity_line1": re.compile(r"velocity_line1"),
    "tke_line1": re.compile(r"tke_line1"),
    "nut_line1": re.compile(r"nut_line1"),
    "temperature_line1_constT": re.compile(r"temperature_line1_constT"),
    "temperature_line2_constT": re.compile(r"temperature_line2_constT"),
    "temperature_line1_constq": re.compile(r"temperature_line1_constq"),
    "temperature_line2_constq": re.compile(r"temperature_line2_constq"),
}


@dataclass
class Sample:
    x: float
    y: float
    re: float
    pr: float
    bc: int  # 0 = constant temperature, 1 = constant heat flux
    quantity: str  # "u_mag", "k", "theta"
    target: float


def _line_point(line_start, line_end, t):
    x0, y0 = line_start
    x1, y1 = line_end
    return x0 + t * (x1 - x0), y0 + t * (y1 - y0)


def _lookup_re(case_id: str, table2: pd.DataFrame) -> float:
    row = table2[table2["case"] == case_id]
    if len(row):
        return float(row.iloc[0]["Re_bulk"])
    # CFD-generated velocity/tke files use case_id "unit_cell" - that run is
    # fixed at the finalized Table 5 Reynolds number.
    return 9800.0


def _lookup_pr(case_id: str) -> float:
    """Handles both digitized_data's convention (case_id IS the Pr value,
    e.g. "2" or "0.025") and cfd_generated's convention (case_id contains
    "Pr<value>", e.g. "case16_Pr1_constT")."""
    m = re.search(r"Pr([0-9.]+)", case_id)
    if m:
        return float(m.group(1))
    try:
        return float(case_id)
    except ValueError:
        raise ValueError(f"Could not parse Prandtl number from case_id={case_id!r}")


def load_profiles(digitized_dir: Path, table2_path: Path, geometry: dict):
    table2 = pd.read_csv(table2_path)
    samples = []

    def iter_files(base: Path):
        if not base.exists():
            return
        yield from sorted(base.glob("*.csv"))
        cfd_dir = base / "cfd_generated"
        if cfd_dir.exists():
            yield from sorted(cfd_dir.glob("*.csv"))

    for path in iter_files(Path(digitized_dir)):
        name = path.name
        kind = next((k for k, pat in FILE_KIND_PATTERNS.items() if pat.search(name)), None)
        if kind is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue

        if kind == "velocity_line1":
            line = (geometry["line1_start"], geometry["line1_end"])
            quantity = "u_mag"
            # momentum is Pr/BC-independent (paper Section 4.3); filler values,
            # MomentumNet in model.py never reads these two columns.
            pr, bc = 1.0, 0
        elif kind == "tke_line1":
            line = (geometry["line1_start"], geometry["line1_end"])
            quantity = "k"
            pr, bc = 1.0, 0
        elif kind == "nut_line1":
            line = (geometry["line1_start"], geometry["line1_end"])
            quantity = "nut"
            pr, bc = 1.0, 0
        else:
            line1 = "line1" in kind
            line = (geometry["line1_start"], geometry["line1_end"]) if line1 else (
                geometry["line2_start"], geometry["line2_end"]
            )
            quantity = "theta"
            bc = 0 if "constT" in kind else 1
            pr = None

        for _, row in df.iterrows():
            case_id = str(row["case_id"])
            t = float(row["x"])
            xp, yp = _line_point(line[0], line[1], t)
            re = _lookup_re(case_id, table2)
            this_pr = pr if pr is not None else _lookup_pr(case_id)
            samples.append(
                Sample(x=xp, y=yp, re=re, pr=this_pr, bc=bc, quantity=quantity, target=float(row["y"]))
            )

    return samples


def samples_to_arrays(samples):
    """Returns dict[quantity] -> (inputs[N,5], targets[N,1]) as float32 numpy arrays.
    inputs columns: x, y, re, pr, bc
    """
    by_quantity = {}
    for s in samples:
        by_quantity.setdefault(s.quantity, {"inputs": [], "targets": []})
        by_quantity[s.quantity]["inputs"].append([s.x, s.y, s.re, s.pr, s.bc])
        by_quantity[s.quantity]["targets"].append([s.target])
    out = {}
    for q, d in by_quantity.items():
        out[q] = (
            np.asarray(d["inputs"], dtype=np.float32),
            np.asarray(d["targets"], dtype=np.float32),
        )
    return out
