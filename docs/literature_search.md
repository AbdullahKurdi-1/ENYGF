# Literature search: has a PINN been applied to the Hooper case?

Date: 26 Sep 2026. Method: web search (about 15 queries: PINN / physics-informed
+ rod bundle, Hooper, tight lattice, subchannel, fuel assembly, liquid metal,
Nek5000 DNS, Mathur/Kwiatkowski/Shams). Limits: several publishers
(AIP, Frontiers, the Chinese J. Nuclear Power Engineering site) could not be
opened from here, so some items are judged from abstracts only; Chinese-
language journals and conference proceedings (NURETH, ICONE) are poorly
indexed. **Confirm with Google Scholar / Scopus before claiming novelty.**

## Result

No published PINN (or other ML model) of the Hooper rod bundle or of the
Mathur et al. 2023 DNS was found. The only search hit combining a PINN with
P/D = 1.107 and Re = 9800 is this project's own public GitHub repository.

## Closest work

| Work | What it is | Difference from this project |
|---|---|---|
| "Deep Learning Solution Technology for Sparse Data of Multi-Channel Flow Field of PWR Rod Bundle", J. Nucl. Power Eng., 2024 (doi 10.13832/j.jnpe.2024.080039) | Equation-constrained deep learning from sparse points in a PWR rod bundle (60 points = 7.8% of data, R^2 > 0.95); adaptive correction of the governing equations | Closest in spirit (sparse data + equations). PWR geometry, flow only as far as the abstract says; no DNS validation, no heat transfer / Pr sweep, no explainability. Full text not read. |
| POD-RBFNN surrogate for 5x5 fuel rod bundles, Nucl. Eng. Des. 2024 | Data-driven surrogate (reduced-order model) | Not physics-informed; different geometry |
| Multi-fidelity ROM for cross-scale flow reconstruction in fuel rod bundles, Phys. Fluids 37, 097164 (2025) | POD + neural network, multi-fidelity | Not a PINN |
| Data-driven turbulence modelling in peripheral subchannels, Phys. Fluids 36, 025141 (2024) | ML correction of a RANS model, trained on DNS (three rods in a channel) | Improves the RANS model itself (uses DNS for training); we keep DNS for evaluation only |
| PSO / neural-network optimised RANS for fuel bundles with spacer grids (Springer proceedings, 2025) | Turbulence-model tuning from PIV data | Not a PINN |
| PINN for temperature fields in fuel-heat-pipe assemblies (Energy, 2025) | PINN, heat conduction in a solid assembly | No coolant flow |
| Physics-informed ML framework (PIMLAF) for DNB in rod bundles | ML on correlation residuals | Not field reconstruction |

General PINN work on turbulent RANS flows and sparse-data reconstruction
exists (channels, airfoils, porous media, flames, Rayleigh-Benard), but none
on a tight-lattice rod bundle with heat transfer at several Prandtl numbers.

## What this means for our claim

Defensible: "a PINN for turbulent flow and heat transfer in a tight-lattice
rod bundle at the conditions of the Hooper / Mathur et al. 2023 DNS benchmark
(P/D 1.107, Re 9800, Pr 0.025-2), validated against that DNS" - to our
knowledge the first. Not defensible with the current CFD: "a PINN of the
Hooper case", because our domain is the infinite-array unit cell, not the
six-rod Hooper domain (`docs/figures/domain_comparison.png`).
