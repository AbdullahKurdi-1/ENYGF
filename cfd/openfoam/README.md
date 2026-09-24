# OpenFOAM cases (OpenFOAM v13, openfoam.org)

Built on the syntax of v13's own tutorial `incompressibleFluid/ductSecondaryFlow`
(periodic duct, symmetry planes, `meanVelocityForce`), which is structurally
the same problem. Still untested on a real OpenFOAM install from this side -
expect to paste back an error or two on the first run.

## What it models

The **quarter unit cell** of an infinite square rod array (P/D = 1.107,
D = 0.14 m): rod at the origin, cell spanning 0..a in x and y (a = P/2),
symmetry planes on all four sides, one cell thick and periodic in the
streamwise direction z. This is the same "effective unit cell" the 2023 DNS
averages its statistics over, and the one its figures are plotted on.

Normalisation follows the 2023 DNS: U_b = 1 m/s, rho = 1, rho*cp = 1, so
Re_h = 9800 fixes nu = Dh/Re_h = 8.0099e-6 m²/s (Dh = 0.078497 m for the
unit cell).

Mesh: structured two-block O-grid from `blockMesh`, 2 x 60 x 40 cells,
clustered at the rod (first-cell y+ below 1, so near-wall eddy viscosity is ~0).

## Pipeline

```bash
cd cfd/openfoam/unit_cell
./Allrun                          # blockMesh -> checkMesh -> foamRun (steady RANS, k-omega SST)
cd ..
python3 generate_thermal_case.py  # builds thermal/ from the converged flow
cd thermal && ./Allrun && cd ..   # 8 temperature fields on the frozen flow
python3 extract_profiles.py       # -> digitized_data/cfd_generated/cfd_profiles.csv
```

- `unit_cell/` — flow. `foamRun` with `solver incompressibleFluid` (v13
  replaced `simpleFoam`); steady via `steadyState` + `SIMPLE`. Line
  profiles of U, k, nut are sampled at every write (`system/functions`).
- `thermal/` (generated) — all 8 temperature fields (Pr = 0.025/1/2/7 x
  iso-temperature/iso-flux) solved at once as passive scalars with
  `solver functions` + the `scalarTransport` function object (v13 replaced
  `scalarTransportFoam`). Diffusivity = nu/Pr + nut/Pr_t (Pr_t = 0.9).
  Wall BCs and uniform heat sinks mirror the 2023 DNS exactly: iso-temperature
  T = 0 at the rod with sink 1 W/m³; iso-flux q = 1 W/m² with sink
  4q/Dh = 50.96 W/m³ - the same value the DNS reports.
- `extract_profiles.py` — reads the sampled lines, checks convergence
  (change between the last two write times), writes one tidy CSV.

## Sampled lines (`case_geometry.py`)

`seg1` -> `seg2` -> `seg3` together trace the DNS's unit-cell-boundary
coordinate xi (rod in the narrow gap -> gap centre -> subchannel centre ->
rod at 45°); `line15` is the 15° wall-normal line of DNS Fig. 7.

## Disclosed simplifications

- Infinite-array unit cell with symmetry planes, not the DNS's confined
  six-rod domain with gap walls. The central unit cell of that domain is
  geometrically identical to ours; the difference is the influence of the
  outer walls.
- Steady RANS: no gap vortex street / flow pulsations (no time axis; the
  symmetry planes also suppress the antisymmetric gap oscillation). The DNS
  pulsation frequency (3.7 Hz, St = 0.52) is out of reach by construction.
- k-omega SST is a linear eddy-viscosity model: it predicts essentially no
  secondary flow in the cross-section (the DNS finds it weak). An RSM such
  as LRR (used by the v13 `ductSecondaryFlow` tutorial) is the upgrade path.
- Constant Pr_t = 0.9. Known to be crude for liquid metals (Pr = 0.025);
  compare the Pr = 0.025 Nusselt number against the DNS before trusting it.
- Pr = 7 has no DNS counterpart (the 2023 DNS ran Pr = 0.025, 1, 2 only).
