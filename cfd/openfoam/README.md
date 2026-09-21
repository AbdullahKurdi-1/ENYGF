# OpenFOAM unit-cell case (v0 scaffold)

Targets **OpenFOAM.org v11** naming conventions (`constant/momentumTransport`,
`constant/fvModels`). If you're on an ESI/OpenCFD version instead, rename
`momentumTransport` -> `turbulenceProperties` and `fvModels` -> the equivalent
entries in `system/fvOptions`, and adjust the `RAS.model` key to `RASModel`.

This was authored without a local OpenFOAM install to test against (this
session's sandbox doesn't have OpenFOAM) — treat it as a first-pass scaffold,
not a validated case. Expect to run `blockMesh`, `surfaceCheck` on the STL,
`snappyHexMesh`, and `checkMesh`, and to fix whatever those complain about.
That's a normal part of standing up any new CFD case, not a sign something
here is fundamentally wrong.

## What it models

A single square-pitch unit cell around one rod (see
`../../paper_reference/NOTES.md` for why — the paper's full 6-rod
wall-bounded domain isn't numerically reconstructable from the extracted PDF
text alone). Streamwise direction is `z`, made periodic with a `cyclic`
patch pair and a `meanVelocityForce` momentum source targeting the Re=9800
bulk velocity — this reproduces the paper's own periodic-BC philosophy
(Section 3.1) without needing an artificially long domain. The four outer
cross-section faces are `symmetryPlane` (this is what makes it an *infinite
array* unit cell rather than a 6-rod bounded bundle).

## What this case can and cannot validate against the paper

**Cannot reproduce, structurally, not just approximately:** the gap vortex
street / axial flow pulsations that are the paper's central subject. Two
independent reasons, not one:

1. `simpleFoam` is steady-state — it converges to a time-averaged mean
   field. There is no time axis for an oscillation to exist on.
2. The gap vortex street is an *antisymmetric* oscillation between adjacent
   rod gaps. A `symmetryPlane` boundary mathematically enforces mirror
   symmetry across itself, which suppresses exactly that mode - regardless
   of solver. This isn't a guess: the paper itself (Section 3.1, citing
   Cardoso de Souza et al. 2015) reports that reduced-domain
   simplifications like this one caused "unwanted numerical errors" for
   anything beyond basic mean-flow topology, which is why the paper used
   its full wall-bounded domain instead.

Reproducing the pulsation itself would need a real rebuild: a small
periodic rod cluster (e.g. 2x2, cyclic in both cross-section directions
instead of symmetryPlane) run with `pimpleFoam` (unsteady URANS) instead of
`simpleFoam`. That's a materially bigger mesh and a transient run, not a
tweak to this case - out of scope here on purpose (see project decision
log / conversation history for the tradeoff discussion).

**Can validate:** mean-flow trends only - relative velocity/TKE/temperature
profile shapes, and how they order across the Reynolds and Prandtl sweeps.
State the claim at this scope in any write-up ("validates mean-flow trends
against the paper's published profiles") - not "reproduces the paper's rod
bundle simulation," which this case cannot do and was never going to.

## Pipeline

1. `geometry/make_rod_stl.py` — writes `rod.stl`, a cylinder along z
   representing the solid rod (removed from the fluid domain by
   snappyHexMesh).
2. `unit_cell/Allrun` — `blockMesh` (background box) -> `snappyHexMesh`
   (cuts the rod out, refines near its surface) -> `checkMesh` ->
   `simpleFoam` (steady RANS, k-omega SST) to convergence.
3. Once `simpleFoam` converges, run `generate_scalar_cases.py` to spin up
   the 8 Pr x BC combinations from `paper_reference/table4_thermal_cases.csv`,
   each reusing the converged `U` field and solving a passive-scalar
   temperature equation with `scalarTransportFoam` (diffusivity = nu/Pr,
   matching the paper's own passive-scalar treatment in Section 4.3 — this
   is valid because momentum is Pr-independent, exactly as the paper states).
4. `extract_profiles.py` samples Line 1 / Line 2 profiles from each
   converged case into the same CSV schema as `digitized_data/`, so the ML
   pipeline in `ml/` can't tell your CFD data from the paper's digitized
   figures — same loader, same format.

## Known simplifications (disclose these in your write-up)

- Unit cell (infinite array), not the paper's finite wall-bounded 6-rod
  domain — matches the paper's own precedent for "reduced domain" URANS
  studies, disclosed as insufficient for LES/DNS-grade fidelity but fine
  for RANS-level surrogate training data.
- No boundary-layer inflation (`addLayers false`) in the v0 snappyHexMesh —
  add layers once the base mesh checks out, or wall-function accuracy will
  be poor. Left off here so there's one fewer thing to debug on the first
  meshing pass.
- Air only in `unit_cell/` (needed to fix Re via bulk velocity); the other
  three Prandtl numbers only ever appear as a passive-scalar diffusivity in
  the `scalarTransportFoam` step, never as a different fluid.
