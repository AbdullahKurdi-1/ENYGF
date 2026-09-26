# OpenFOAM cases (OpenFOAM v13, openfoam.org)

Built on the syntax of v13's own tutorial `incompressibleFluid/ductSecondaryFlow`
(periodic duct, symmetry planes, `meanVelocityForce`), which is structurally
the same problem. Run and converged on OpenFOAM v13. Cross-checked with an
independent copy: the flow on OpenFOAM v1912 gives the same pressure
gradient (0.1883 vs 0.1881), and the temperatures from a separate
finite-volume solve on that flow match the v13 ones to within 0.5% (both
checks done with the earlier viscosity, nu = 8.0099e-6).

## What it models

The **quarter unit cell** of an infinite square rod array (P/D = 1.107,
D = 0.14 m): rod at the origin, cell spanning 0..a in x and y (a = P/2),
symmetry planes on all four sides, one cell thick and periodic in the
streamwise direction z. This is the same "effective unit cell" the 2023 DNS
averages its statistics over, and the one its figures are plotted on.

Normalisation follows the 2023 DNS: U_b = 1 m/s, rho = 1, rho*cp = 1, and
its Reynolds-number definition Re_h = U_b*Dh/nu = 9800 with its Dh = 0.0712 m,
so nu = 7.2653e-6 m²/s - the same fluid as the DNS. Nusselt numbers also use
Dh = 0.0712 m, as the DNS does. (The unit cell's own Dh is 0.078497 m; it is
only used where the geometry needs it, e.g. the sink 4q/Dh = P/A.)

Mesh: structured two-block O-grid from `blockMesh`, 2 x 60 x 40 cells,
clustered at the rod (first-cell y+ below 1, so near-wall eddy viscosity is ~0).

## Pipeline

```bash
cd cfd/openfoam/unit_cell
./Allrun                          # blockMesh -> checkMesh -> foamRun (steady RANS, k-omega SST)
cd ..
python3 generate_thermal_case.py  # builds thermal/ from the converged flow
cd thermal && ./Allrun && cd ..   # 8 temperature fields on the frozen flow
python3 extract_profiles.py       # -> digitized_data/cfd_generated/ (field, Nusselt, line CSVs)
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
- `extract_profiles.py` — writes three CSVs to `digitized_data/cfd_generated/`:
  `cfd_field.csv` (every mesh cell: w, nu_t, k, 8 temperatures; plus wall
  shear, iso-T wall heat flux and iso-flux wall temperature on the rod),
  `cfd_nusselt.csv` (the CFD's own Nusselt numbers, defined exactly as in the
  DNS, Eq. 4-5, with Dh = 0.0712 m) and `cfd_profiles.csv` (the sampled
  lines). Also checks convergence (change between the last two write times)
  and prints the pressure gradient recomputed from the wall shear - it must
  match the one in `unit_cell/log.foamRun`.
- `foam_io.py` — small reader for OpenFOAM ASCII meshes and fields (no
  OpenFOAM installation needed to run the extractor).

## Looking at the results in ParaView

`paraFoam` has to be started from **inside** a case folder - it looks for
`./constant`, and anywhere else it stops with
`FATAL ERROR: Mesh constant does not exist`. The helper script does the
`cd` for you:

```bash
cd ~/ENYGF/cfd/openfoam
./view_in_paraview.sh                 # unit_cell: U, p, k, omega, nut
./view_in_paraview.sh thermal         # the 8 temperature fields (T_Pr..._isoT / _isoFlux)
./view_in_paraview.sh mesh_study/fine # a mesh-study case
./view_in_paraview.sh unit_cell -builtin   # if ParaView says the OpenFOAM reader is missing
```

(Equivalent by hand: `cd ~/ENYGF/cfd/openfoam/unit_cell && paraFoam`.)

In ParaView: click **Apply**, pick the last time step (the "Last Frame"
button), colour by a field (e.g. `U` - Z component, or `nut`). Useful filters:
- **Plot Over Line** - e.g. from (0.070001, 0.000001, 0.0005) to
  (0.077499, 0.000001, 0.0005) is `seg1`, the narrow gap (all sampled lines
  are listed in `unit_cell/system/functions`);
- **Slice** with normal (0,0,1) at z = 0.0005 gives a clean 2D view (the
  mesh is one cell thick in z);
- **Surface With Edges** representation shows the mesh itself.
The temperature fields live in `thermal/` (last time step, e.g. 100); the
velocity there is the frozen flow copied from `unit_cell`.

## Mesh-refinement study (`mesh_study.py`)

Reviewers expect evidence that the results do not depend on the mesh.
Three meshes, each 2x finer in both directions with the same wall grading:
coarse 30x20 (1 200 cells), medium 60x40 (4 800, the production mesh),
fine 120x80 (19 200).

```bash
cd ~/ENYGF/cfd/openfoam
python3 mesh_study.py setup        # creates mesh_study/coarse and mesh_study/fine
./mesh_study/Allrun                # flow + thermal on both (roughly 1-3 h)
python3 mesh_study.py report       # table + Grid Convergence Index -> mesh_study/mesh_study.csv
```

The report gives, for every quantity (pressure gradient, friction velocity,
gap wall shear, 8 Nusselt numbers), the value on each mesh, the observed
order of convergence, the Richardson-extrapolated value and the GCI
(Celik et al. 2008) - the numerical uncertainty of the production results.

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
