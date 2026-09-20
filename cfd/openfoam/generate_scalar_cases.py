"""Spin up the 8 Pr x wall-BC scalarTransportFoam cases from Table 4.

Reuses the converged unit_cell mesh and (frozen) velocity field, and adds a
passive-scalar temperature equation per case - exactly the paper's own
justification in Section 4.3 for treating temperature as a passive scalar
(momentum is the same across all four Prandtl numbers; only the diffusivity
and wall BC change).

Run this *after* `unit_cell/Allrun` has produced a converged simpleFoam
solution (a numeric time directory beyond `0/`).

Usage:
    python3 generate_scalar_cases.py [--unit-cell ../unit_cell] [--out cases]
"""
import argparse
import csv
import shutil
from pathlib import Path

AIR_RHO = 1.2       # kg/m3
AIR_CP = 1005.0     # J/kg/K
AIR_NU = 1.5e-05    # m2/s, matches unit_cell/constant/transportProperties

HERE = Path(__file__).resolve().parent
TABLE4 = HERE.parent.parent / "paper_reference" / "table4_thermal_cases.csv"


def read_table4(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def latest_time_dir(unit_cell_dir: Path) -> Path:
    times = []
    for p in unit_cell_dir.iterdir():
        if p.is_dir():
            try:
                times.append((float(p.name), p))
            except ValueError:
                continue
    if not times:
        raise SystemExit(
            f"No time directories found under {unit_cell_dir}. "
            "Run unit_cell/Allrun first (needs a converged simpleFoam solution)."
        )
    times.sort(key=lambda t: t[0])
    latest_value, latest_path = times[-1]
    if latest_value == 0.0:
        raise SystemExit(
            f"Only 0/ exists under {unit_cell_dir} - simpleFoam hasn't produced "
            "a converged solution yet."
        )
    return latest_path


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def foam_header(class_, object_):
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_};\n"
        f"    object      {object_};\n"
        "}\n\n"
    )


def control_dict():
    return foam_header("dictionary", "controlDict") + (
        "application     scalarTransportFoam;\n\n"
        "startFrom       startTime;\n"
        "startTime       0;\n\n"
        "stopAt          endTime;\n"
        "endTime         500;\n\n"
        "deltaT          1;\n\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n\n"
        "purgeWrite      2;\n\n"
        "writeFormat     ascii;\n"
        "writePrecision  6;\n"
        "writeCompression off;\n\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n\n"
        "runTimeModifiable true;\n"
    )


def fv_schemes():
    return foam_header("dictionary", "fvSchemes") + (
        "ddtSchemes\n{\n    default steadyState;\n}\n\n"
        "gradSchemes\n{\n    default Gauss linear;\n}\n\n"
        "divSchemes\n{\n"
        "    default            none;\n"
        "    div(phi,T)         bounded Gauss upwind;\n"
        "}\n\n"
        "laplacianSchemes\n{\n    default Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default linear;\n}\n\n"
        "snGradSchemes\n{\n    default corrected;\n}\n"
    )


def fv_solution():
    return foam_header("dictionary", "fvSolution") + (
        "solvers\n{\n"
        "    T\n    {\n"
        "        solver          smoothSolver;\n"
        "        smoother        GaussSeidel;\n"
        "        tolerance       1e-9;\n"
        "        relTol          0.1;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 1;\n"
        "    residualControl\n    {\n        T 1e-6;\n    }\n"
        "}\n\n"
        "relaxationFactors\n{\n"
        "    equations\n    {\n        T 0.7;\n    }\n"
        "}\n"
    )


def transport_properties(DT):
    return foam_header("dictionary", "transportProperties") + (
        "transportModel  Newtonian;\n\n"
        f"DT              DT [0 2 -1 0 0 0 0] {DT:.6e};\n"
        f"nu              nu [0 2 -1 0 0 0 0] {AIR_NU:.6e};\n"
    )


def temperature_field(bc_type, wall_value_or_gradient, internal_value):
    header = foam_header("volScalarField", "T")
    dims = "dimensions      [0 0 0 1 0 0 0];\n\n"
    internal = f"internalField   uniform {internal_value};\n\n"
    if bc_type == "constT":
        rod_bc = (
            "    rod\n    {\n"
            "        type            fixedValue;\n"
            f"        value           uniform {wall_value_or_gradient};\n"
            "    }\n"
        )
    else:
        rod_bc = (
            "    rod\n    {\n"
            "        type            fixedGradient;\n"
            f"        gradient        uniform {wall_value_or_gradient:.6e};\n"
            "    }\n"
        )
    boundary = (
        "boundaryField\n{\n"
        + rod_bc
        + "    xMin  { type symmetryPlane; }\n"
        "    xMax  { type symmetryPlane; }\n"
        "    yMin  { type symmetryPlane; }\n"
        "    yMax  { type symmetryPlane; }\n"
        "    front { type cyclic; }\n"
        "    back  { type cyclic; }\n"
        "}\n"
    )
    return header + dims + internal + boundary


def sample_dict():
    return (
        "interpolationScheme cellPoint;\n"
        "setFormat           raw;\n\n"
        "sets\n(\n"
        "    line1\n    {\n"
        "        type    uniform;\n"
        "        axis    distance;\n"
        "        start   (0.07      0         0.005);\n"
        "        end     (0.0775    0         0.005);\n"
        "        nPoints 50;\n"
        "    }\n"
        "    line2\n    {\n"
        "        type    uniform;\n"
        "        axis    distance;\n"
        "        start   (0.0494975 0.0494975 0.005);\n"
        "        end     (0.0775    0.0775    0.005);\n"
        "        nPoints 50;\n"
        "    }\n"
        ");\n\n"
        "fields (T);\n"
    )


def generate(unit_cell_dir: Path, out_dir: Path, table4_path: Path):
    rows = read_table4(table4_path)
    latest = latest_time_dir(unit_cell_dir)
    print(f"Using converged fields from {latest}")

    for row in rows:
        pr = float(row["Pr"])
        is_const_t = bool(row["wall_temperature_K"].strip())
        bc_type = "constT" if is_const_t else "constq"
        DT = AIR_NU / pr

        case_name = f"case{row['case']}_Pr{row['Pr']}_{bc_type}"
        case_dir = out_dir / case_name

        # mesh: symlink to avoid duplicating a potentially large polyMesh
        constant_dir = case_dir / "constant"
        constant_dir.mkdir(parents=True, exist_ok=True)
        poly_mesh_link = constant_dir / "polyMesh"
        if not poly_mesh_link.exists():
            poly_mesh_link.symlink_to(
                (unit_cell_dir / "constant" / "polyMesh").resolve(),
                target_is_directory=True,
            )
        write(constant_dir / "transportProperties", transport_properties(DT))

        write(case_dir / "system" / "controlDict", control_dict())
        write(case_dir / "system" / "fvSchemes", fv_schemes())
        write(case_dir / "system" / "fvSolution", fv_solution())
        write(case_dir / "system" / "sampleDict", sample_dict())

        zero_dir = case_dir / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)
        for field in ("U", "phi"):
            src = latest / field
            if src.exists():
                shutil.copy(src, zero_dir / field)

        if bc_type == "constT":
            wall_T = float(row["wall_temperature_K"])
            temp_content = temperature_field("constT", wall_T, wall_T + 20.0)
        else:
            q = float(row["wall_heat_flux_W_m2"])
            alpha = DT
            grad = q / (AIR_RHO * AIR_CP * alpha)
            temp_content = temperature_field("constq", grad, 295.0)
        write(zero_dir / "T", temp_content)

        print(f"wrote {case_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", default=str(HERE / "unit_cell"))
    parser.add_argument("--out", default=str(HERE / "cases"))
    parser.add_argument("--table4", default=str(TABLE4))
    args = parser.parse_args()

    generate(Path(args.unit_cell), Path(args.out), Path(args.table4))
