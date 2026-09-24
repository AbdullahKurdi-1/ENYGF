"""Collect the sampled line profiles from the flow and thermal cases into
one tidy CSV for the PINN: digitized_data/cfd_generated/cfd_profiles.csv

Columns: case_id, line, s, xi, x, y, quantity, value, Re, Pr, bc
  quantity is one of: w (axial velocity), k, nut, T
  s  = distance along the sampled line [m]
  xi = distance along the 2023 DNS unit-cell-boundary path [m] (seg1-3 only)

Iso-flux temperatures are only defined up to a constant (Neumann wall BC +
balanced sink), so they are shifted to T = 0 at the rod surface in the narrow
gap (first point of seg1). The PINN uses the same gauge.

Also reports how much each profile changed between the last two write times,
as a convergence check.

Usage:
    python3 extract_profiles.py
"""
import argparse
import csv
import math
import re
from pathlib import Path

from case_geometry import NU, PR_VALUES, RE, SAMPLE_LINES, U_BULK, DH_CELL, field_name

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
            names = header[-1].lstrip("#").split()
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


def main(unit_cell: Path, thermal: Path, out_csv: Path):
    rows = []

    flow_dir = unit_cell / "postProcessing" / "sampleFlow"
    if not flow_dir.exists():
        raise SystemExit(f"{flow_dir} not found - has unit_cell/Allrun finished?")
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
    print(f"(case constants: U_b={U_BULK}, nu={NU:.4e}, Dh_cell={DH_CELL:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", default=str(HERE / "unit_cell"))
    parser.add_argument("--thermal", default=str(HERE / "thermal"))
    parser.add_argument(
        "--out", default=str(HERE.parent.parent / "digitized_data" / "cfd_generated" / "cfd_profiles.csv")
    )
    args = parser.parse_args()
    main(Path(args.unit_cell), Path(args.thermal), Path(args.out))
