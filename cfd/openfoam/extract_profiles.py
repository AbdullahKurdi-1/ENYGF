"""Collect the CFD results for the PINN, into digitized_data/cfd_generated/:

  cfd_profiles.csv  sampled line profiles (DNS unit-cell path + 15 deg line)
  cfd_field.csv     every cell of the mesh (and the iso-flux wall temperatures)
  cfd_nusselt.csv   the CFD's own Nusselt numbers, defined as in the 2023 DNS

Columns: case_id, line, s, xi, x, y, quantity, value, Re, Pr, bc
  quantity is one of: w (axial velocity), k, nut, T
  s  = distance along the sampled line [m]
  xi = distance along the 2023 DNS unit-cell-boundary path [m] (seg1-3 only)

Iso-flux temperatures are only defined up to a constant (Neumann wall BC +
balanced sink), so they are shifted to T = 0 at the rod surface in the narrow
gap (first point of seg1). The PINN uses the same gauge.

Also reports how much each profile changed between the last two write times,
as a convergence check.

cfd_field.csv columns: case_id, kind, x, y, area, quantity, value, Re, Pr, bc
  kind = cell (cell centre, area = cell cross-section area) or
         wall (on the rod surface, area = 0): quantity tau_w (wall shear
         stress / rho), q_w (iso-temperature wall heat flux) or T (iso-flux
         wall temperature). Wall gradients use the first cell, which sits at
         y+ < 1.

Usage:
    python3 extract_profiles.py
"""
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

from case_geometry import (
    DH_CELL, DH_DNS, NU, PR_VALUES, RE, ROD_RADIUS, SAMPLE_LINES, SINK_ISO_T, U_BULK,
    WALL_HEAT_FLUX, field_name,
)
from foam_io import Mesh, read_internal_field, read_patch_value

HERE = Path(__file__).resolve().parent
BCS = ["isoT", "isoFlux"]
VECTOR_FIELDS = {"U"}


def time_dirs(func_dir: Path):
    out = []
    for p in func_dir.iterdir():
        try:
            out.append((float(p.name), p))
        except ValueError:
            pass
    return [p for _, p in sorted(out)]


def expand(fields):
    cols = []
    for f in fields:
        cols += [f"{f}_x", f"{f}_y", f"{f}_z"] if f in VECTOR_FIELDS else [f]
    return cols


def split_field_names(text, known_fields):
    """'T_Pr1_isoT_k' -> ['T_Pr1_isoT', 'k']: field names may contain '_',
    so match the requested names greedily instead of splitting on '_'."""
    if not text:
        return list(known_fields)
    out, pos = [], 0
    by_length = sorted(known_fields, key=len, reverse=True)
    while pos < len(text):
        for f in by_length:
            end = pos + len(f)
            if text.startswith(f, pos) and (end == len(text) or text[end] == "_"):
                out.append(f)
                pos = end + 1
                break
        else:
            raise SystemExit(f"Can't recognise field names in '{text}' (expected some of {known_fields}).")
    return out


def resolve_truncated(names, requested_fields, path):
    """OpenFOAM v13 shortens long column names in raw headers, e.g.
    'T_Pr0p025_isoFlux' -> 'T_Pr0p025_is...'. Map each shortened name to the
    one requested field it can be: same prefix, and not already present in
    full elsewhere in the header. Refuse to guess if that isn't unique."""
    full = set(names)
    out = []
    for n in names:
        if not n.endswith("..."):
            out.append(n)
            continue
        prefix = n[:-3]
        candidates = [c for c in expand(requested_fields) if c.startswith(prefix) and c not in full]
        if len(candidates) != 1:
            raise SystemExit(
                f"Column '{n}' in {path} is shortened and matches {candidates or 'no'} requested "
                "fields - can't resolve it safely. Use shorter field names."
            )
        out.append(candidates[0])
        full.add(candidates[0])
    return out


def read_set_files(time_dir: Path, requested_fields):
    """Return {set_name: {column_name: [values]}} for every raw file.

    Handles both layouts OpenFOAM has used: one file per set with a '#'
    header naming the columns, or one file per set and field group whose
    name encodes the fields (e.g. seg1_k_nut.xy, seg1_U.xy).
    """
    sets = {}
    for f in sorted(time_dir.iterdir()):
        if not f.is_file():
            continue
        lines = f.read_text().splitlines()
        header = [ln for ln in lines if ln.lstrip().startswith("#")]
        rows = [ln.split() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        if not rows:
            continue
        stem = f.name.split(".")[0]
        set_name = stem.split("_")[0]
        if header:
            names = resolve_truncated(header[-1].lstrip("#").split(), requested_fields, f)
        else:
            fields = split_field_names(stem[len(set_name) + 1:], requested_fields)
            names = ["distance"] + expand(fields)
        if len(names) != len(rows[0]):
            raise SystemExit(
                f"Can't map columns in {f}: header/inferred names {names} "
                f"but rows have {len(rows[0])} values. Paste this message and the "
                f"first 3 lines of that file to get the parser fixed."
            )
        cols = sets.setdefault(set_name, {})
        for i, n in enumerate(names):
            cols[n] = [float(r[i]) for r in rows]
    return sets


def column(cols, *candidates):
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): k for k in cols}
    for c in candidates:
        key = norm.get(re.sub(r"[^a-z0-9]", "", c.lower()))
        if key is not None:
            return cols[key]
    return None


def point_on_line(line, s):
    (x0, y0), (x1, y1), _ = SAMPLE_LINES[line]
    L = math.hypot(x1 - x0, y1 - y0)
    return x0 + (x1 - x0) * s / L, y0 + (y1 - y0) * s / L


def convergence_report(func_dir: Path, requested_fields, label):
    ts = time_dirs(func_dir)
    if len(ts) < 2:
        print(f"[{label}] only one write time - can't check convergence yet.")
        return
    a = read_set_files(ts[-2], requested_fields)
    b = read_set_files(ts[-1], requested_fields)
    worst, worst_col, skipped = 0.0, None, set()
    for s in b:
        for col, vb in b[s].items():
            if col.lower() == "distance" or col not in a.get(s, {}):
                continue
            va = a[s][col]
            scale = max(abs(v) for v in list(va) + list(vb))
            # Columns that are zero to round-off (e.g. U_x, U_y: no secondary
            # flow with k-omega SST) have meaningless relative changes.
            if scale < 1e-10:
                skipped.add(col)
                continue
            change = max(abs(x - y) for x, y in zip(va, vb)) / scale
            if change > worst:
                worst, worst_col = change, f"{col} on {s}"
    flag = "OK" if worst < 0.01 else "NOT CONVERGED - run longer"
    print(f"[{label}] max relative change between t={ts[-2].name} and t={ts[-1].name}: "
          f"{worst:.2e} ({worst_col})  {flag}")
    if skipped:
        print(f"[{label}] ignored (zero to round-off): {sorted(skipped)}")


def latest_time(case: Path) -> Path:
    times = []
    for p in case.iterdir():
        try:
            if p.is_dir() and float(p.name) > 0:
                times.append((float(p.name), p))
        except ValueError:
            pass
    if not times:
        raise SystemExit(f"No results (time directory > 0) in {case}.")
    return max(times)[1]


def export_field(unit_cell: Path, thermal: Path, field_csv: Path, nusselt_csv: Path):
    """Every cell of the mesh, plus the CFD's Nusselt numbers (2023 DNS Eq. 4-5)."""
    mesh = Mesh(unit_cell)
    n = mesh.n_cells
    flow_t = latest_time(unit_cell)
    x, y = mesh.cell_xy[:, 0], mesh.cell_xy[:, 1]
    area = mesh.cell_area
    w = read_internal_field(flow_t / "U", n)[:, 2]
    rows = []

    def add(case_id, kind, xs, ys, areas, q, vals, pr="", bc=""):
        rows.extend((case_id, kind, a, b, c, q, v, RE, pr, bc) for a, b, c, v in zip(xs, ys, areas, vals))

    for q, name in (("w", None), ("k", "k"), ("nut", "nut")):
        vals = w if name is None else read_internal_field(flow_t / name, n)
        add("flow", "cell", x, y, area, q, vals)

    # Driving pressure gradient from the wall shear: G * A = sum(tau_w * L).
    rod = mesh.patch_faces("rod")
    owner = mesh.owner[rod]
    length = np.linalg.norm(mesh.face_area[rod], axis=1) / mesh.lz
    normal = mesh.face_area[rod, :2] / (length * mesh.lz)[:, None]
    wall_dist = np.abs(((mesh.face_centre[rod, :2] - mesh.cell_xy[owner]) * normal).sum(1))
    tau_w = NU * w[owner] / wall_dist
    G = (tau_w * length).sum() / area.sum()
    print(f"[field] {n} cells; flow from {flow_t}; pressure gradient from wall shear = {G:.5f} "
          f"(compare with the 'pressure gradient' in unit_cell/log.foamRun)")

    wall_xy = mesh.face_centre[rod, :2] * (ROD_RADIUS / np.linalg.norm(mesh.face_centre[rod, :2], axis=1))[:, None]
    zeros = np.zeros(len(rod))
    add("flow", "wall", wall_xy[:, 0], wall_xy[:, 1], zeros, "tau_w", tau_w)
    gap_face = np.argmin(np.minimum(np.abs(np.arctan2(wall_xy[:, 1], wall_xy[:, 0])),
                                    np.abs(np.arctan2(wall_xy[:, 0], wall_xy[:, 1]))))
    nusselt = []
    if not thermal.exists():
        print(f"[field] no thermal case at {thermal} - flow only.")
    else:
        thermal_t = latest_time(thermal)
        for pr in PR_VALUES:
            lam = NU / pr
            for bc in BCS:
                f = field_name(pr, bc)
                T = read_internal_field(thermal_t / f, n)
                Tb = (w * T * area).sum() / (w * area).sum()
                if bc == "isoT":
                    Tw = np.zeros(len(rod))
                    flux = lam * (Tw - T[owner]) / wall_dist
                    phi_m = (flux * length).sum() / length.sum()
                    # steady state: all heat removed by the sink enters through the rod
                    balance = phi_m / (SINK_ISO_T * area.sum() / length.sum()) - 1
                    gauge = 0.0
                else:
                    Tw = read_patch_value(thermal_t / f, "rod", len(rod))
                    if Tw is None:
                        # OpenFOAM v13 writes only the gradient of a fixedGradient
                        # patch; its wall value is T_cell + gradient * distance.
                        grad = read_patch_value(thermal_t / f, "rod", len(rod), key="gradient")
                        Tw = T[owner] + grad * wall_dist
                    phi_m, balance = WALL_HEAT_FLUX, 0.0
                    gauge = Tw[gap_face]
                Tw_m = (Tw * length).sum() / length.sum()
                nu_cfd = phi_m * DH_DNS / (lam * (Tw_m - Tb))
                nusselt.append((pr, bc, nu_cfd, balance))
                add(f[2:], "cell", x, y, area, "T", T - gauge, pr, bc)
                if bc == "isoFlux":
                    add(f[2:], "wall", wall_xy[:, 0], wall_xy[:, 1], zeros, "T", Tw - gauge, pr, bc)
                else:
                    add(f[2:], "wall", wall_xy[:, 0], wall_xy[:, 1], zeros, "q_w", flux, pr, bc)
        print(f"[field] temperatures from {thermal_t}")

    field_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(field_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case_id", "kind", "x", "y", "area", "quantity", "value", "Re", "Pr", "bc"])
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {field_csv}")

    if nusselt:
        with open(nusselt_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["Pr", "bc", "Nu_CFD", "wall_heat_balance_error"])
            wr.writerows(nusselt)
        print(f"CFD Nusselt numbers (Dh = {DH_DNS} m, as in the DNS) -> {nusselt_csv}")
        for pr, bc, nu_cfd, bal in nusselt:
            note = f"   (heat balance error {bal:+.1e})" if bc == "isoT" else ""
            print(f"    Pr={pr:<6g}{bc:8s} Nu = {nu_cfd:7.2f}{note}")


def export_lines(unit_cell: Path, thermal: Path, out_csv: Path):
    rows = []

    flow_dir = unit_cell / "postProcessing" / "sampleFlow"
    if not flow_dir.exists():
        print(f"{flow_dir} not found - skipping line profiles.")
        return
    flow_fields = ["U", "k", "nut"]
    convergence_report(flow_dir, flow_fields, "flow")
    flow_sets = read_set_files(time_dirs(flow_dir)[-1], flow_fields)
    print("flow columns found:", {s: list(c) for s, c in flow_sets.items()})
    for line, cols in flow_sets.items():
        if line not in SAMPLE_LINES:
            continue
        xi0 = SAMPLE_LINES[line][2]
        dist = column(cols, "distance")
        w = column(cols, "U_z", "Uz", "U2")
        for qname, vals in (("w", w), ("k", column(cols, "k")), ("nut", column(cols, "nut"))):
            if vals is None:
                print(f"WARNING: no '{qname}' column in flow set {line}")
                continue
            for s, v in zip(dist, vals):
                x, y = point_on_line(line, s)
                xi = "" if xi0 is None else xi0 + s
                rows.append(("flow", line, s, xi, x, y, qname, v, RE, "", ""))

    thermal_dir = thermal / "postProcessing" / "sampleThermal"
    if thermal_dir.exists():
        t_fields = [field_name(pr, bc) for pr in PR_VALUES for bc in BCS]
        convergence_report(thermal_dir, t_fields, "thermal")
        t_sets = read_set_files(time_dirs(thermal_dir)[-1], t_fields)
        for pr in PR_VALUES:
            for bc in BCS:
                f = field_name(pr, bc)
                gauge = 0.0
                if bc == "isoFlux":
                    seg1 = column(t_sets.get("seg1", {}), f)
                    gauge = seg1[0] if seg1 else 0.0
                for line, cols in t_sets.items():
                    if line not in SAMPLE_LINES:
                        continue
                    vals = column(cols, f)
                    if vals is None:
                        print(f"WARNING: no '{f}' column in thermal set {line}")
                        continue
                    xi0 = SAMPLE_LINES[line][2]
                    for s, v in zip(column(cols, "distance"), vals):
                        x, y = point_on_line(line, s)
                        xi = "" if xi0 is None else xi0 + s
                        rows.append((f[2:], line, s, xi, x, y, "T", v - gauge, RE, pr, bc))
    else:
        print(f"No thermal results at {thermal_dir} (run generate_thermal_case.py + thermal/Allrun) - flow only.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case_id", "line", "s", "xi", "x", "y", "quantity", "value", "Re", "Pr", "bc"])
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")


def main(unit_cell: Path, thermal: Path, out_dir: Path):
    print(f"(case constants: U_b={U_BULK}, nu={NU:.4e}, Dh_DNS={DH_DNS}, Dh_cell={DH_CELL:.6f})")
    export_lines(unit_cell, thermal, out_dir / "cfd_profiles.csv")
    export_field(unit_cell, thermal, out_dir / "cfd_field.csv", out_dir / "cfd_nusselt.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", default=str(HERE / "unit_cell"))
    parser.add_argument("--thermal", default=str(HERE / "thermal"))
    parser.add_argument("--out-dir", default=str(HERE.parent.parent / "digitized_data" / "cfd_generated"))
    args = parser.parse_args()
    main(Path(args.unit_cell), Path(args.thermal), Path(args.out_dir))
