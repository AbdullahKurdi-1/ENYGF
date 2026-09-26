# ENYGF — AI & Nuclear Symbiosis: PINN Surrogate for a Closely-Spaced Rod Bundle

Physics-informed neural network (PINN) for fully developed turbulent flow and
heat transfer in a bare rod bundle (P/D = 1.107, Re = 9800), built against
two published papers from the NRG/NCBJ programme:

- Shams & Kwiatkowski (2018), *Ann. Nucl. Energy* 121 — URANS calibration
  study that designed the case (geometry, Re, Prandtl numbers).
- Mathur, Kwiatkowski, Potempski & Komen (2023), *Int. J. Heat Mass Transfer*
  211 — the DNS of that case. **The independent validation reference.**

See `paper_reference/` for what each paper does and does not provide.

**Research question, plan and current results:** `docs/research_plan.md`.
**Session-by-session record:** `docs/sessions/`.
**CFD and PINN vs the DNS, with explanations:** `docs/dns_comparison.md`.

## Approach

No public full-field dataset exists for this case, so:

1. **Training data** comes from our own lightweight CFD: an OpenFOAM v13
   steady RANS (k-omega SST) of the quarter unit cell, with 8 passive-scalar
   temperature fields (Pr = 0.025/1/2/7 x iso-temperature/iso-flux). Same
   normalisation, Reynolds-number definition, wall BCs and heat sinks as the
   DNS. Every mesh cell is exported (plus wall shear and wall heat flux on the
   rod); 20% of the cells are held out to measure the PINN's test error.
2. **Validation data** comes from the DNS and is **never used in training**:
   its Nusselt numbers (Table 1, exact values) and its figures, digitized
   reproducibly from the PDF (`digitized_data/digitize_dns_figures.py`):
   wall shear (Fig. 6), U+ vs r+ (Fig. 7), turbulent kinetic energy (Fig. 9a),
   wall heat flux (Fig. 11a) and excess temperature (Fig. 12a).

Three levels of comparison, kept separate in every result table:
PINN vs. the CFD it learned from (the machine-learning error), CFD vs. DNS
(the RANS model error), and PINN vs. DNS (the total).

## Pipeline

```bash
# 1. CFD (needs OpenFOAM v13 - see cfd/openfoam/README.md)
cd cfd/openfoam/unit_cell && ./Allrun && cd ..
python3 generate_thermal_case.py
cd thermal && ./Allrun && cd ..
python3 extract_profiles.py          # -> digitized_data/cfd_generated/: cfd_field.csv,
                                     #    cfd_nusselt.csv (CFD's own Nu), cfd_profiles.csv

# 2. DNS figures are already digitized (digitized_data/dns_*.csv); to redo it:
#    python3 digitized_data/digitize_dns_figures.py <path to the 2023 DNS PDF>

# 3. PINN - either the notebook (recommended: ml/notebooks/pinn_walkthrough.ipynb)
#    or the same code from the terminal:
cd ../../ml
python3 src/compare_cfd_to_papers.py # optional: CFD vs. DNS at the key points (digitized DNS data)
python3 src/train.py                 # 6000 epochs, roughly 30-60 min on a laptop CPU
python3 src/evaluate.py              # CFD fit + DNS comparison, plots in ml/outputs/plots/
```

## The model (`ml/src/`)

Cross-section coordinates (x, y) of the quarter unit cell; U_b = 1,
nu = Dh/Re, rho*cp = 1.

- **Flow network** -> axial velocity w, eddy viscosity nu_t, TKE k; plus a
  learned driving pressure gradient G(Re). Axial RANS momentum
  0 = G + div((nu + nu_t) grad w), with the bulk velocity held at U_b and the
  overall force balance G * area = wall shear * rod perimeter.
- **Thermal network** -> one output per (Pr, wall BC) case. Energy
  0 = div((nu/Pr + nu_t/Pr_t) grad T) - S, with the DNS's uniform heat sinks.
  One-way coupled: temperature never trains the flow network.
- Exact wall conditions built into the network (w = nu_t = k = 0 and
  iso-temperature T = 0 at the rod); symmetry planes, iso-flux wall heat
  flux and bulk velocity enforced as losses. Multi-scale wall-distance input
  features handle the very thin near-wall layers.
- **Two-phase training**: fit the CFD data first, then ramp in the PDE
  residuals. Switching the PDEs on from a random start lets the energy
  residual drive every temperature to the trivial T = 0 solution.
- **What the physics buys** (sparse-data experiment, `src/experiments.py`,
  results in `docs/research_plan.md`, measured on a copy of the CFD case):
  with dense data a plain network fits the CFD 2-3x more closely, but its
  fields violate the equations ~10x more; with 1% of the cells the PINN has
  about half the plain network's error and a ~3x smaller worst-case Nusselt
  error (3 random seeds); with only the sampled lines it reconstructs the
  velocity field ~7x better, but both miss the Nusselt numbers by up to ~18%.

- **Explainability** (`src/xai.py`, notebook Step 7-8): residual maps (where
  the network breaks the equations), the energy budget (share of the heat
  carried by turbulence vs conduction, from the network's own eddy viscosity)
  and sensor importance (error growth when one measurement line is removed).

Tests: `cd ml && python3 -m pytest tests -q` - `test_physics.py` checks the
PDE operators against hand-derived results; `test_smoke.py` runs the whole
pipeline on fake data.

## Disclosed limitations

State these in any write-up:

- Infinite-array unit cell with symmetry planes, not the DNS's confined
  six-rod Hooper domain, where three of each subchannel's four gaps are
  closed by walls (`docs/figures/domain_comparison.png`). Only the open-gap
  side (|theta| <= 45 deg) is compared.
- Steady RANS training data: no gap vortex street / flow pulsations (the DNS
  measures 3.7 Hz, St = 0.52 - unreachable by construction). Validation
  claims are about mean quantities only.
- Linear eddy-viscosity closure (k-omega SST in the CFD, nu_t in the PINN):
  no secondary flow in the cross-section. Constant Pr_t = 0.9, known to be
  crude for liquid metals.
- The DNS's Reynolds number and Nusselt numbers are defined with its
  whole-domain Dh (0.0712 m); we use the same definitions (so the same
  fluid viscosity), but our cell's bulk velocity is 1 by construction while
  the DNS's central cell's local bulk velocity is not published.
- The PINN predicts temperature only at the trained Prandtl numbers.
- Pr = 7 has no DNS counterpart.
- Single Reynolds number (9800) unless you add CFD runs at others
  (`ml/configs/default.yaml: flow.re_values`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
cd ml && python3 -m pytest tests -q
```
