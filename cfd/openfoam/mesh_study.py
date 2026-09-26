"""Mesh-refinement (grid-convergence) study of the CFD case.

Three meshes, each twice as fine as the previous one in both directions:
  coarse  30 x 20 x 2 blocks =  1 200 cells   (mesh_study/coarse)
  medium  60 x 40 x 2 blocks =  4 800 cells   (the existing unit_cell + thermal)
  fine   120 x 80 x 2 blocks = 19 200 cells   (mesh_study/fine)
The wall grading (last/first radial cell = 8) is kept, so every cell shrinks
by the same factor 2 - the condition for a Richardson / GCI analysis.

Reported for every mesh: pressure gradient, friction velocity, wall-shear
ratio in the gap, first-cell y+, and the 8 Nusselt numbers; then the
observed order of convergence p, the Richardson-extrapolated value and the
Grid Convergence Index GCI_fine (Celik et al. 2008, J. Fluids Eng. 130,
078001), i.e. the numerical uncertainty of the medium and fine results.

Usage (Ubuntu terminal, OpenFOAM loaded, venv active):
    cd ~/ENYGF/cfd/openfoam
    python3 mesh_study.py setup      # creates mesh_study/ and mesh_study/Allrun
    ./mesh_study/Allrun              # flow + thermal on coarse and fine (1-3 h)
    python3 mesh_study.py report     # table + GCI -> mesh_study/mesh_study.csv
"""
import argparse
import math
import re
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE / "mesh_study"
LEVELS = {"coarse": (30, 20, 20000), "fine": (120, 80, 20000)}   # nRadial, nAzim, max iterations


def setup():
    base = HERE / "unit_cell"
    for name, (n_rad, n_azi, end_time) in LEVELS.items():
        case = STUDY / name
        if case.exists():
            raise SystemExit(f"{case} exists - delete mesh_study/ to start again.")
        case.mkdir(parents=True)
        for item in ("0", "constant", "system", "Allrun", "Allclean"):
            src = base / item
            if src.is_dir():
                shutil.copytree(src, case / item, ignore=shutil.ignore_patterns("polyMesh"))
            else:
                shutil.copy(src, case / item)
        bmd = case / "system" / "blockMeshDict"
        text = bmd.read_text()
        text = re.sub(r"^nRadial\s+\d+;", f"nRadial {n_rad};", text, flags=re.M)
        text = re.sub(r"^nAzim\s+\d+;", f"nAzim   {n_azi};", text, flags=re.M)
        bmd.write_text(text)
        cd = case / "system" / "controlDict"
        # Finer meshes need more iterations; the SIMPLE residual control in
        # fvSolution stops the run as soon as it has converged.
        cd.write_text(re.sub(r"^endTime\s+\d+;", f"endTime         {end_time};", cd.read_text(), flags=re.M))
        print(f"created {case}  ({n_rad} x {n_azi} per block, up to {end_time} iterations)")
    allrun = STUDY / "Allrun"
    allrun.write_text("""#!/bin/sh
# Flow, then the 8 temperature fields, on the coarse and the fine mesh.
cd "${0%/*}" || exit 1
for m in coarse fine
do
    echo "=== $m: flow ==="
    (cd $m && ./Allrun) || exit 1
    grep "pressure gradient" $m/log.foamRun | tail -1
    echo "=== $m: thermal ==="
    python3 ../generate_thermal_case.py --unit-cell $m --out ${m}_thermal || exit 1
    (cd ${m}_thermal && ./Allrun) || exit 1
done
echo "Done. Now run:  python3 mesh_study.py report"
""")
    allrun.chmod(0o755)
    print("\nNext:  ./mesh_study/Allrun   then   python3 mesh_study.py report")


def gci(f1, f2, f3, r=2.0):
    """Celik et al. (2008): f1 fine, f2 medium, f3 coarse, constant ratio r."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return dict(p=float("nan"), f_ext=f1, gci_fine=0.0, gci_medium=0.0, note="identical")
    ratio = e32 / e21
    note = "monotonic" if ratio > 0 else "oscillatory"
    p = abs(math.log(abs(ratio))) / math.log(r)
    p = min(max(p, 0.5), 4.0)          # guard against unphysical orders
    f_ext = f1 + (f1 - f2) / (r**p - 1)
    g_fine = 1.25 * abs((f1 - f2) / f1) / (r**p - 1)
    g_med = 1.25 * abs((f2 - f3) / f2) / (r**p - 1)
    return dict(p=p, f_ext=f_ext, gci_fine=g_fine, gci_medium=g_med, note=note)


def report():
    import extract_profiles as ex
    levels = {"coarse": (STUDY / "coarse", STUDY / "coarse_thermal"),
              "medium": (HERE / "unit_cell", HERE / "thermal"),
              "fine": (STUDY / "fine", STUDY / "fine_thermal")}
    res = {}
    for name, (flow, thermal) in levels.items():
        if not (flow / "constant" / "polyMesh").exists():
            raise SystemExit(f"{flow} has no mesh - run mesh_study/Allrun first.")
        print(f"\n--- {name}: {flow}")
        out = STUDY / "results" / name
        res[name] = ex.export_field(flow, thermal, out / "cfd_field.csv", out / "cfd_nusselt.csv")

    rows = []
    for key, label in (("G", "pressure gradient dp/dz"), ("u_tau", "mean friction velocity"),
                       ("tau_gap_over_mean", "wall shear gap / mean"), ("y_plus_max", "first-cell y+ (max)")):
        rows.append([label] + [res[m][key] for m in ("coarse", "medium", "fine")])
    for i, (pr, bc, _, _) in enumerate(res["fine"]["nusselt"]):
        rows.append([f"Nu Pr={pr:g} {bc}"] + [res[m]["nusselt"][i][2] for m in ("coarse", "medium", "fine")])
    table = pd.DataFrame(rows, columns=["quantity", "coarse", "medium", "fine"])
    for c in ("p", "f_ext", "gci_fine", "gci_medium", "note"):
        table[c] = None
    for i, r in table.iterrows():
        if r["quantity"].startswith("first-cell"):
            continue
        g = gci(r["fine"], r["medium"], r["coarse"])
        for c in ("p", "f_ext", "gci_fine", "gci_medium", "note"):
            table.at[i, c] = g[c]
    table["cells"] = None
    table.loc[0, "cells"] = f"{res['coarse']['n_cells']}/{res['medium']['n_cells']}/{res['fine']['n_cells']}"
    table.to_csv(STUDY / "mesh_study.csv", index=False)
    with pd.option_context("display.width", 160):
        print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\ngci_medium = numerical uncertainty of the medium-mesh (production) result; below ~2-3% "
          "means the mesh is fine enough for the RANS-vs-DNS differences (13-45%) to be model error, not "
          "mesh error.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["setup", "report"])
    args = parser.parse_args()
    setup() if args.action == "setup" else report()
