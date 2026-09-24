# Mathur, Kwiatkowski, Potempski & Komen (2023), IJHMT 211, 124226

"Direct numerical simulation of flow and heat transport in a closely-spaced
bare rod bundle"

## What this paper actually is

The real DNS the 2018 calibration paper (`NOTES.md`) was designing toward -
run with the spectral-element code Nek5000, not a URANS/RANS approximation.
**625.9 million grid points** (7th-order polynomial elements on a 1.22M
element base mesh), **9,984 processors**, 7.65 million time-steps, at
NCBJ's Świerk Computing Centre. This is the number to point at whenever
anyone asks why a laptop CFD run isn't expected to reproduce this paper -
it's not a fidelity gap, it's a 5-6 order of magnitude compute gap.

**Data availability (checked directly, last page of the PDF): "Data will
be made available on request."** No public download. Same situation as the
2018 paper - what we get is richer published figures, not a dataset.

## Parameters (supersedes/refines the 2018 paper's numbers where they differ)

| Parameter | Value | Note |
|---|---|---|
| P/D | 1.107 | same as 2018 paper |
| D (rod diameter) | 0.14 m | same |
| Re_h (bulk velocity, hydraulic diameter) | 9800 | same finalized case |
| **Dh (hydraulic diameter)** | **0.0712 m** | explicitly stated here - use this, not the 2018 paper's back-calculated 0.0714 m or this project's idealized-unit-cell 0.0785 m (see `NOTES.md`) |
| Domain streamwise length | 32 x Dh (periodic) | *not* the 2018 paper's L/4 = 2.285 m / 32 Dh in this paper's own normalization - both reduced from the original 128 Dh Hooper test section, converging on the same order |
| Re_tau (friction Reynolds number) | 624 | based on rod-surface-averaged u_tau |
| u_tau (friction velocity) | 0.0637 m/s | averaged over -45 to +45 degrees around the central rod |
| **Pr values simulated** | **0.025, 1.0, 2.0** | **Pr=7 from the 2018 calibration paper was dropped - never actually DNS'd** |
| Thermal BCs | iso-temperature and iso-flux | both run for all 3 Pr = 6 thermal fields total |
| Dominant pulsation frequency | 3.7 Hz | from PSD of horizontal velocity at narrow-gap centre |
| Strouhal number (St = fD/Ub) | 0.52 | cf. Lai et al. (2019) at higher Re: St=0.56 |
| Strouhal number (St_tau^-1, gap-width based) | 0.12 | cf. Möller's correlation: 0.14 |

## Geometry: confirms this project's disclosed simplification, doesn't change it

This paper's actual DNS domain is the **real confined two-subchannel
geometry with straight gap walls** (Fig. 1) - not an idealized infinite
periodic array. The paper explicitly notes that averaging over the
theoretical "one-eighth" infinite-array unit cell is *not valid* here
because "the presence of gap walls and the resulting asymmetry" breaks
that symmetry; they instead average over a folded one-quarter "effective
unit cell" specific to their bounded domain.

The effective unit cell the DNS averages over is one quarter of a
subchannel - geometrically the same as this project's quarter unit cell
(`cfd/openfoam/unit_cell`). So the DNS statistics are directly comparable
with ours; the remaining difference is the influence of the DNS's outer gap
walls on that central cell, which our symmetry planes leave out.

## The DNS set-up this project mirrors (Section 2)

- U_b = 1 m/s, rho = 1, nu = Dh*U_b/Re_h, rho*cp = 1, alpha = nu/Pr.
- Temperatures are passive scalars, several solved alongside one momentum
  solution.
- Iso-temperature: excess temperature Theta = T - T_w = 0 at the rod, with
  a uniform volumetric sink of 1 W/m³ driving heat from the rod to the bulk.
- Iso-flux: rod heat flux 1 W/m², with a uniform sink of 50.96 W/m³
  (= flux x rod area / volume). Our quarter cell gives 4q/Dh = 50.957 W/m³ -
  the same number, a useful consistency check on the geometry.
- Gap walls in the DNS are adiabatic (we have symmetry planes instead).

## Quantitative targets

**Table 1, Nusselt numbers - no digitizing needed** (`table_dns_2023_nusselt.csv`):

| Pr | Pe | Nu iso-T | Nu iso-flux | El-Genk correlation |
|---|---|---|---|---|
| 0.025 | 245 | 7.54 | 8.85 | 11.54 |
| 1.0 | 9800 | 38.74 | 40.19 | 38.98 |
| 2.0 | 19,600 | 52.00 | 52.77 | 49.00 |

Nu = phi_m*Dh / (lambda*(T_w,m - T_b)), T_b = integral(u*T dA)/integral(u dA),
wall means over -45° to 45° of the central rod (Eq. 4-5).
`ml/src/evaluate.py` computes the same quantity from the PINN. Caveat: the
PINN uses the unit-cell Dh (0.0785 m), the DNS its whole-domain Dh (0.0712 m).

**Figures worth digitizing** (templates in `digitized_data/`):

| Figure | Content | How it's used |
|---|---|---|
| Fig. 6 | Wall shear around the central rod, / mean over -45..45° | vs. PINN wall shear from dw/dn |
| Fig. 7(b)-(d) | U+ vs r+ at 0, 15, 45° from the gap (log axis) | vs. PINN in wall units, using its own local u_tau |
| Fig. 11(a) | Iso-temperature wall heat flux, / mean | vs. PINN wall heat flux |
| Fig. 12(a) | Iso-temperature Theta/Theta_b along the unit-cell boundary xi | vs. PINN along the same path |

The unit-cell-boundary coordinate xi (Fig. 8) runs: rod surface in the narrow
gap -> narrow-gap centre (xi/Dh ~ 0.1) -> subchannel centre -> back to the
rod at 45° (xi/Dh ~ 1.75). Our sampled lines `seg1-seg3` follow exactly this.

Not usable for our model: Fig. 9-10 and 13-17 (Reynolds stresses, anisotropy,
temperature fluctuations, turbulent heat fluxes) are second-order turbulence
statistics a RANS eddy-viscosity model doesn't produce; Fig. 12(b) (iso-flux)
has an ambiguous temperature reference; Fig. 19 (pulsation spectrum) is out
of reach of a steady model by construction.
