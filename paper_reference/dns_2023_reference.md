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

This directly confirms what `NOTES.md` already disclosed about
`cfd/openfoam/unit_cell`: our single-rod, symmetryPlane, infinite-array
simplification is a real, acknowledged departure from both papers' actual
geometry, not just from the 2018 paper's. Nothing to change in the
OpenFOAM case as a result - the simplification was already correctly
flagged - but cite this paper's Section 2 (not just the 2018 paper) when
justifying it in a write-up.

## Best available digitization targets (better than the 2018 paper's Figs. 11-13/20-22)

| Figure | Content | Use for |
|---|---|---|
| Fig. 6 | Wall shear stress distribution around central rod (normalized by mean) | Direct check on our momentum solution's wall-shear prediction |
| Fig. 7(b)-(d) | Mean streamwise velocity in wall units (U+ vs r+) at 0, 15, 45 degrees from the gap | The single best velocity validation target available - already in wall units, already compares against experiment (Hooper) and higher-Re DNS (Lai et al. 2019) |
| Fig. 9 | Normal Reynolds stress components along the unit-cell boundary | No direct PINN output to compare (we don't resolve individual Reynolds stress components, only eddy-viscosity nu_t) - use qualitatively: confirms turbulence suppression in the gap vs. subchannel, a trend our nu_t field should also show |
| Figs. 14-17 | Temperature statistics and turbulent heat flux, by Pr and thermal BC | Direct check on theta/temperature predictions - note Pr=7 has no entry here (see Pr gap above) |
| Fig. 19 | PSD and Strouhal number of flow pulsations | Not reproducible by this project's steady RANS case (disclosed limitation) - cite as the concrete number that illustrates the gap, don't attempt to match it |

Add digitized points from this paper into the same `digitized_data/*.csv`
files as the 2018 paper's figures where the quantity matches (e.g. Fig. 7's
velocity profiles go into `velocity_line1_by_Re.csv` if you pick a case_id
convention that avoids colliding with the 2018 paper's `Rn` names - e.g.
prefix with `DNS_`).
