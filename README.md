# ENYGF 2026 — AI & Nuclear Symbiosis: PINN Surrogate for a Closely-Spaced Rod Bundle

Physics-informed ML surrogate for velocity, TKE, and temperature profiles in
a P/D=1.107 bare rod bundle, built against Shams & Kwiatkowski (2018),
*Annals of Nuclear Energy* 121:146-161 — see `paper_reference/NOTES.md` for
exactly what that paper does and does not give us, and why the pipeline is
structured the way it is below.

## Why this structure

The paper is a URANS calibration study, not a released DNS dataset — no
public full-field rod-bundle dataset with heat transfer exists (confirmed:
even Merzari et al. 2020's DNS follow-up work on a 5x5 bundle says so
explicitly). So "using the paper as a benchmark" means two things at once,
not a data download:

1. **Generate real CFD ground truth yourself**, from the paper's own
   finalized case spec (`paper_reference/table5_finalized_params.csv`), via
   OpenFOAM — `cfd/openfoam/`.
2. **Validate against the paper's own published figures**, by digitizing
   the line profiles it actually reports (Figs. 11-13, 20, 22) — into
   `digitized_data/`.

Both feed the same PyTorch RANS-PINN in `ml/`, through one shared CSV
schema, so the model can't tell your CFD run from the paper's digitized
curves.

## Pipeline, in order

```
1. cfd/openfoam/geometry/make_rod_stl.py     -> rod.stl
2. cfd/openfoam/unit_cell/Allrun             -> converged RANS flow field (simpleFoam, k-omega SST)
3. cfd/openfoam/generate_scalar_cases.py     -> 8 Pr x BC scalarTransportFoam cases (Table 4)
4. (run each case's scalarTransportFoam)
5. cfd/openfoam/extract_profiles.py          -> digitized_data/cfd_generated/*.csv
6. digitize Figs. 11-13/20/22 by hand        -> digitized_data/*.csv  (see its README)
7. ml/src/train.py                           -> trains the PINN on everything above
8. ml/src/evaluate.py                        -> prediction-vs-reference plots + RMSE
```

Steps 1-5 need OpenFOAM (your machine, not this repo's dev environment).
Steps 7-8 need only `ml/requirements.txt` and run anywhere, even with zero
CFD data — the pipeline degrades gracefully to training on whatever's in
`digitized_data/` (empty templates by default; fill them in per its
README before results mean anything).

## What the model actually is

Two coupled sub-networks (`ml/src/model.py`), matching the paper's own
physics claim that momentum is Pr/BC-independent (Section 4.3):

- `MomentumNet(x, y, Re) -> u, v, p, nu_t, k` — steady RANS with a
  Boussinesq eddy-viscosity closure (`nu_t` is a learned field, not a
  transported k-omega quantity — see `ml/src/physics.py` docstring for why).
- `ThermalNet(x, y, Re, Pr, bc, u, v) -> theta` — passive-scalar energy
  equation, one-way coupled (temperature never feeds back into momentum,
  exactly like running `scalarTransportFoam` on a frozen velocity field).

Losses: data-fit against whatever's in `digitized_data/` (paper figures +
your CFD), PDE residuals via autograd at random collocation points
(continuity, momentum, energy), and no-slip / symmetryPlane BC losses.

## Known, disclosed simplifications

Put these in your write-up rather than letting a reviewer find them first:

- OpenFOAM case models a single-rod **unit cell** (infinite square array via
  symmetryPlane BCs), not the paper's finite 6-rod, wall-bounded domain —
  the paper's own dimensioned domain drawing (Figs. 1-3) isn't
  reconstructable from extracted PDF text alone. See `cfd/openfoam/README.md`.
- Turbulence closure is Boussinesq/eddy-viscosity inside the PINN, not a
  transported k-omega model — `k` is predicted but is a data-fit output
  only, with no PDE residual of its own.
- The paper's Line 1 / Line 2 non-dimensionalization arrived through PDF
  text extraction as a garbled formula; treated as min-max normalization.
  Verify once you have the actual figures in front of you.
- The OpenFOAM case in `cfd/openfoam/unit_cell` was written without a local
  OpenFOAM install to test against — expect a normal amount of mesh/solver
  iteration on your machine, it's a first-pass scaffold.

## Setup

```
pip install -r ml/requirements.txt
cd ml && python3 -m pytest tests/test_smoke.py -q   # pipeline sanity check, no CFD needed
```
