"""Build the thermal case from a converged unit_cell flow solution (OpenFOAM v13).

One case solves all 8 temperature fields (Pr = 0.025/1/2/7 x iso-temperature /
iso-flux) as passive scalars on the frozen flow field - the same approach as
the 2023 DNS, which solved all its temperature fields alongside a single
momentum solution. Uses v13's `foamRun` with `solver functions`, which runs
the scalarTransport function object (this replaced scalarTransportFoam).

Thermal set-up mirrors the 2023 DNS so results are directly comparable:
  - rho*cp = 1, diffusivity D = nu/Pr + nut/Pr_t
  - iso-temperature: T = 0 at the rod ("excess temperature"), uniform sink of 1
  - iso-flux: rod heat flux q = 1, uniform sink q*P/A = 50.96 (DNS: 50.96)

Usage (after unit_cell/Allrun has converged):
    python3 generate_thermal_case.py
    cd thermal && ./Allrun
"""
import argparse
import shutil
from pathlib import Path

from case_geometry import (
    NU, PR_TURBULENT, PR_VALUES, SAMPLE_LINES, SINK_ISO_FLUX, SINK_ISO_T,
    WALL_HEAT_FLUX, field_name,
)

HERE = Path(__file__).resolve().parent
BCS = ["isoT", "isoFlux"]


def header(cls, obj, location=None):
    loc = f"    location    \"{location}\";\n" if location else ""
    return (
        "FoamFile\n{\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def latest_time_dir(case: Path) -> Path:
    times = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                times.append((float(p.name), p))
            except ValueError:
                pass
    times.sort()
    if not times or times[-1][0] == 0:
        raise SystemExit(f"No converged time directory in {case} - run unit_cell/Allrun first.")
    return times[-1][1]


def temperature_field(pr, bc):
    if bc == "isoT":
        rod = "        type            fixedValue;\n        value           uniform 0;\n"
    else:
        # snGrad along the outward normal (fluid -> rod). Heat flows into the
        # fluid, so T rises towards the wall: dT/dn = q / (rho*cp*nu/Pr) > 0.
        # The wall value uses molecular diffusivity only; nut is ~0 at the wall
        # because the mesh resolves the viscous sublayer (y+ < 1).
        grad = WALL_HEAT_FLUX * pr / NU
        rod = f"        type            fixedGradient;\n        gradient        uniform {grad:.8e};\n"
    return (
        header("volScalarField", field_name(pr, bc))
        + "dimensions      [0 0 0 1 0 0 0];\n\n"
        + "internalField   uniform 0;\n\n"
        + "boundaryField\n{\n    #includeEtc \"caseDicts/setConstraintTypes\"\n\n"
        + "    rod\n    {\n" + rod + "    }\n}\n"
    )


def fv_models():
    lines = []
    for pr in PR_VALUES:
        for bc in BCS:
            sink = SINK_ISO_T if bc == "isoT" else SINK_ISO_FLUX
            lines.append(
                f"        {field_name(pr, bc)}\n        {{\n"
                f"            explicit    {-sink:.6f};\n            implicit    0;\n        }}\n"
            )
    return (
        header("dictionary", "fvModels", "constant")
        + "// Uniform volumetric heat sinks balancing the wall heat input (2023 DNS set-up).\n"
        + "heatSinks\n{\n    type            semiImplicitSource;\n\n    cellZone        all;\n\n"
        + "    volumeMode      specific;\n\n    sources\n    {\n"
        + "".join(lines)
        + "    }\n}\n"
    )


def functions():
    blocks = []
    for pr in PR_VALUES:
        for bc in BCS:
            f = field_name(pr, bc)
            blocks.append(
                f"{f}\n{{\n"
                "    type            scalarTransport;\n"
                "    libs            (\"libsolverFunctionObjects.so\");\n"
                f"    field           {f};\n"
                "    schemesField    T;\n"
                "    diffusivity     viscosity;\n"
                f"    alphal          {1.0 / pr:.8g};    // 1/Pr\n"
                f"    alphat          {1.0 / PR_TURBULENT:.8g};    // 1/Pr_t\n"
                "    writeControl    writeTime;\n"
                "}\n\n"
            )
    fields = " ".join(field_name(pr, bc) for pr in PR_VALUES for bc in BCS)
    sets = []
    for name, (start, end, _) in SAMPLE_LINES.items():
        sets.append(
            f"        {name}\n        {{\n"
            "            type    lineUniform;\n            axis    distance;\n"
            f"            start   ({start[0]:.6f} {start[1]:.6f} 0.0005);\n"
            f"            end     ({end[0]:.6f} {end[1]:.6f} 0.0005);\n"
            "            nPoints 100;\n        }\n"
        )
    blocks.append(
        "sampleThermal\n{\n"
        "    type            sets;\n    libs            (\"libsampling.so\");\n"
        "    writeControl    writeTime;\n    interpolationScheme cellPoint;\n"
        "    setFormat       raw;\n\n"
        f"    fields          ({fields});\n\n    sets\n    {{\n"
        + "".join(sets)
        + "    }\n}\n"
    )
    return header("dictionary", "functions", "system") + "".join(blocks)


CONTROL_DICT = header("dictionary", "controlDict", "system") + """\
// Passive-scalar transport on the frozen converged flow (v13 replacement
// for scalarTransportFoam). Pseudo-transient Euler marching to steady state.
solver          functions;

subSolver       incompressibleFluid;

startFrom       latestTime;

startTime       0;

subSolverTime   0;

stopAt          endTime;

endTime         100;

deltaT          0.1;

writeControl    timeStep;

writeInterval   100;

purgeWrite      3;

writeFormat     ascii;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable yes;
"""

FV_SCHEMES = header("dictionary", "fvSchemes", "system") + """\
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,T)      Gauss linearUpwind grad(T);
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""

FV_SOLUTION = header("dictionary", "fvSolution", "system") + """\
solvers
{
    // Shared by all 8 temperature fields via schemesField T
    T
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0;
    }
}

PIMPLE
{
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}
"""

ALLRUN = """#!/bin/sh
cd "${0%/*}" || exit 1

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

runApplication foamRun
"""


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(unit_cell: Path, out: Path):
    latest = latest_time_dir(unit_cell)
    print(f"Using converged flow from {latest}")

    if out.exists():
        raise SystemExit(f"{out} already exists - delete it first to regenerate.")

    (out / "constant").mkdir(parents=True)
    (out / "constant" / "polyMesh").symlink_to(
        (unit_cell / "constant" / "polyMesh").resolve(), target_is_directory=True
    )
    for name in ("physicalProperties", "momentumTransport"):
        shutil.copy(unit_cell / "constant" / name, out / "constant" / name)
    write(out / "constant" / "fvModels", fv_models())

    write(out / "system" / "controlDict", CONTROL_DICT)
    write(out / "system" / "fvSchemes", FV_SCHEMES)
    write(out / "system" / "fvSolution", FV_SOLUTION)
    write(out / "system" / "functions", functions())

    zero = out / "0"
    zero.mkdir()
    missing = []
    for field in ("U", "phi", "p", "k", "omega", "nut"):
        src = latest / field
        if src.exists():
            shutil.copy(src, zero / field)
        else:
            missing.append(field)
    if missing:
        print(f"WARNING: {missing} not found in {latest}; the thermal run may fail without them.")
    for pr in PR_VALUES:
        for bc in BCS:
            write(zero / field_name(pr, bc), temperature_field(pr, bc))

    allrun = out / "Allrun"
    write(allrun, ALLRUN)
    allrun.chmod(0o755)
    print(f"Wrote thermal case to {out}  ->  cd {out} && ./Allrun")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-cell", default=str(HERE / "unit_cell"))
    parser.add_argument("--out", default=str(HERE / "thermal"))
    args = parser.parse_args()
    main(Path(args.unit_cell), Path(args.out))
