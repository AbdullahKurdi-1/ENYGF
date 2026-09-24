# ENYGF — AI & Nuclear Symbiosis: PINN Surrogate for a Closely-Spaced Rod Bundle

Physics-informed neural network (PINN) for fully developed turbulent flow and
heat transfer in a bare rod bundle (P/D = 1.107, Re = 9800), built against
two published papers from the NRG/NCBJ programme:

- Shams & Kwiatkowski (2018), *Ann. Nucl. Energy* 121 — URANS calibration
  study that designed the case (geometry, Re, Prandtl numbers).
- Mathur, Kwiatkowski, Potempski & Komen (2023), *Int. J. Heat Mass Transfer*
  211 — the DNS of that case. **The independent validation reference.**

See `paper_reference/` for what each paper does and does not provide.

## Approach

No public full-field dataset exists for this case, so:

1. **Training data** comes from our own lightweight CFD: an OpenFOAM v13
   steady RANS (k-omega SST) of the quarter unit cell, with 8 passive-scalar
   temperature fields (Pr = 0.025/1/2/7 x iso-temperature/iso-flux). Set up
   with the same normalisation, wall BCs and heat sinks as the DNS.
2. **Validation data** comes from the DNS and is **never used in training**:
   its Nusselt numbers (Table 1, exact values) and, once digitized, its
   wall shear, near-wall velocity (U+ vs r+), wall heat flux and temperature
   profiles.

## Pipeline

```bash
# 1. CFD (needs OpenFOAM v13 - see cfd/openfoam/README.md)
cd cfd/openfoam/unit_cell && ./Allrun && cd ..
python3 generate_thermal_case.py
cd thermal && ./Allrun && cd ..
python3 extract_profiles.py          # -> digitized_data/cfd_generated/cfd_profiles.csv

# 2. Digitize DNS figures (optional but recommended) - digitized_data/README.md

# 3. PINN - either the notebook (recommended: ml/notebooks/pinn_walkthrough.ipynb)
#    or the same code from the terminal:
cd ../../ml
python3 src/compare_cfd_to_papers.py # optional: CFD vs. key values read from both papers
python3 src/train.py                 # ~6000 epochs, ~30 min on a laptop CPU
python3 src/evaluate.py              # CFD fit + DNS comparison, plots in ml/outputs/plots/
```

## The model (`ml/src/`)

Cross-section coordinates (x, y) of the quarter unit cell; U_b = 1,
nu = Dh/Re, rho*cp = 1.

- **Flow network** -> axial velocity w, eddy viscosity nu_t, TKE k; plus a
  learned driving pressure gradient G(Re). Axial RANS momentum
  0 = G + div((nu + nu_t) grad w), with the bulk velocity held at U_b. The
  eddy viscosity is supervised directly by the CFD's nu_t: without it the
  momentum equation alone cannot pin it down.
- **Thermal network** -> T per (Pr, wall BC). Energy
  0 = div((nu/Pr + nu_t/Pr_t) grad T) - S, with the DNS's uniform heat sinks.
  One-way coupled: temperature never trains the flow network.
- Exact wall conditions built into the network (w = nu_t = k = 0 and
  iso-temperature T = 0 at the rod); symmetry planes, iso-flux wall heat
  flux and bulk velocity enforced as losses. Multi-scale wall-distance input
  features and self-normalising loss weights address the usual PINN
  convergence problem of very thin near-wall layers.

Tests: `cd ml && python3 -m pytest tests -q` - `test_physics.py` checks the
PDE operators against hand-derived results; `test_smoke.py` runs the whole
pipeline on fake data.

## Disclosed limitations

State these in any write-up:

- Unit cell with symmetry planes, not the DNS's confined six-rod domain
  (its central unit cell is geometrically identical; outer-wall influence
  is left out).
- Steady RANS training data: no gap vortex street / flow pulsations (the DNS
  measures 3.7 Hz, St = 0.52 - unreachable by construction). Validation
  claims are about mean quantities only.
- Linear eddy-viscosity closure (k-omega SST in the CFD, nu_t in the PINN):
  no secondary flow in the cross-section. Constant Pr_t = 0.9, known to be
  crude for liquid metals.
- The Nusselt comparison uses the unit-cell Dh (0.0785 m) vs. the DNS's
  whole-domain Dh (0.0712 m).
- Pr = 7 has no DNS counterpart.
- Single Reynolds number (9800) unless you add CFD runs at others
  (`ml/configs/default.yaml: flow.re_values`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
cd ml && python3 -m pytest tests -q
```
