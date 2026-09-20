"""Reformat OpenFOAM `sample` .xy output into the same CSV schema used for
the paper's digitized figures (digitized_data/), so ml/src/dataset.py can
load CFD-generated and hand-digitized profiles identically.

Looks for files named line1_<field>.xy / line2_<field>.xy anywhere under
each case directory (robust to OpenFOAM version differences in where
`sample`/postProcess write their output).

Usage:
    python3 extract_profiles.py --unit-cell ../unit_cell \
        --scalar-cases cases --out ../../digitized_data/cfd_generated
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_xy(path: Path, n_value_cols: int):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 1 + n_value_cols:
            continue
        distance = float(parts[0])
        values = [float(v) for v in parts[1 : 1 + n_value_cols]]
        rows.append((distance, values))
    return rows


def normalize_distance(rows):
    distances = [d for d, _ in rows]
    d_min, d_max = min(distances), max(distances)
    span = (d_max - d_min) or 1.0
    return [((d - d_min) / span, v) for d, v in rows]


def find_first(case_dir: Path, pattern: str):
    matches = sorted(case_dir.rglob(pattern))
    return matches[-1] if matches else None  # rglob is unsorted-by-time; last is fine as a pick, verify manually if multiple times exist


def velocity_magnitude(case_dir: Path, line_name: str, out_rows: list, case_id: str):
    f = find_first(case_dir, f"{line_name}_U.xy")
    if f is None:
        return
    for y_star, (ux, uy, uz) in normalize_distance(read_xy(f, 3)):
        mag = (ux**2 + uy**2 + uz**2) ** 0.5
        out_rows.append((case_id, y_star, mag))


def scalar_field(case_dir: Path, line_name: str, field: str, out_rows: list, case_id: str):
    f = find_first(case_dir, f"{line_name}_{field}.xy")
    if f is None:
        return
    for y_star, (val,) in normalize_distance(read_xy(f, 1)):
        out_rows.append((case_id, y_star, val))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "x", "y"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main(unit_cell_dir: Path, scalar_cases_dir: Path, out_dir: Path):
    velocity_rows, tke_rows, nut_rows = [], [], []
    velocity_magnitude(unit_cell_dir, "line1", velocity_rows, "unit_cell")
    scalar_field(unit_cell_dir, "line1", "k", tke_rows, "unit_cell")
    # nu_t is OpenFOAM's own turbulence-model output (k-omega SST), extracted
    # here specifically so the PINN's own learned eddy-viscosity has a real
    # target to match, instead of being constrained only indirectly through
    # the momentum residual + velocity data (see conversation on why that's
    # an under-constrained way to identify it).
    scalar_field(unit_cell_dir, "line1", "nut", nut_rows, "unit_cell")
    write_csv(out_dir / "velocity_line1.csv", velocity_rows)
    write_csv(out_dir / "tke_line1.csv", tke_rows)
    write_csv(out_dir / "nut_line1.csv", nut_rows)

    if scalar_cases_dir.exists():
        for case_dir in sorted(scalar_cases_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            for line_name, out_name in (("line1", "line1"), ("line2", "line2")):
                rows = []
                scalar_field(case_dir, line_name, "T", rows, case_id)
                if rows:
                    write_csv(
                        out_dir / f"temperature_{out_name}_{case_id}.csv", rows
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", default=str(HERE / "unit_cell"))
    parser.add_argument("--scalar-cases", default=str(HERE / "cases"))
    parser.add_argument(
        "--out", default=str(HERE.parent.parent / "digitized_data" / "cfd_generated")
    )
    args = parser.parse_args()
    main(Path(args.unit_cell), Path(args.scalar_cases), Path(args.out))
