# Research plan: sparse-data PINN reconstruction with explainability

Agreed 25 September 2026. This file is the reference for what the project
claims, how it is tested, and what it must not do.

## Research question

> Can a physics-informed neural network reconstruct the velocity and
> temperature fields and the Nusselt number of a tight rod bundle from
> sparse measurements, and how does it compare with a purely data-driven
> network?

Why it matters for nuclear engineering: a reactor never provides a full
field, only a handful of thermocouples and pressure taps. The value of a
PINN is how little data it needs, not how well it fits data we already have.

## Why data is needed at all

The eddy viscosity nu_t is not governed by any equation inside the PINN, so
it is not identifiable from physics alone. A data-free PINN would need the
full k-omega SST equations inside the network - a research project of its
own and notoriously unstable. The question is therefore how little data, not
whether to use data.

## Main experiment: data fraction x physics

Train on a subset of the CFD, evaluate on all cells and on the CFD Nusselt
numbers.

| Training data                  | Plain NN (no physics) | PINN (data + physics) |
|--------------------------------|-----------------------|-----------------------|
| 100% of cells (upper bound)    |                       |                       |
| 10% of cells                   |                       |                       |
| 1% of cells (~50 points)       |                       |                       |
| 4 sampled lines only           |                       |                       |
| sensor-like: a few wall + bulk points |                |                       |

Metrics per cell: held-out RMSE/range for w, nu_t, k, T (per case); learned
dp/dz vs CFD; Nusselt PINN vs CFD (all 8 cases); PDE residual RMS.
Repeat each run with 2-3 random seeds and report the spread.

Expected before testing: physics keeps accuracy as data shrinks while the
plain NN degrades. If physics does not help at low data, that is a valid
result and is reported as such.

### First results (independent test copy of the CFD case, 25 Sep 2026)

Held-out error = RMSE / range on the same 998 held-out cells for every run.
Nu error = worst |PINN/CFD - 1| over the six DNS cases (Pr <= 2). dp/dz from
each network's own wall shear (CFD: 0.1822). One seed unless stated.

| Training data                | Model    | w      | nu_t   | k      | T      | worst Nu error | dp/dz  |
|------------------------------|----------|--------|--------|--------|--------|----------------|--------|
| 100% (3840 cells + wall)     | PINN     | 0.49%  | 1.1%   | 1.7%   | 1.4%   | 1.6%           | 0.1825 |
|                              | plain NN | 0.17%  | 0.7%   | 1.3%   | 0.5%   | 1.3%           | 0.1820 |
| 10% (480 cells, 8 wall)      | PINN     | 0.63%  | 1.1%   | 1.8%   | 1.3%   | 4.3%           | 0.1803 |
|                              | plain NN | 0.20%  | 0.9%   | 1.3%   | 0.6%   | 1.7%           | 0.1813 |
| 1% (48 cells, 1 wall), 3 seeds | PINN   | 1.1-1.5% | 3.6-8.5% | 4.5-10.8% | 1.6-2.5% | 2.0-8.8% (mean 5.2%) | 0.169-0.177 |
|                              | plain NN | 1.7-2.4% | 4.1-20% | 6.3-8.3% | 1.9-3.4% | 10.6-33.7% (mean 18.4%) | 0.169-0.191 |
| 4 lines only (no wall data)  | PINN     | 3.7%   | 13%    | 21%    | 9.7%   | 18%            | 0.147  |
|                              | plain NN | 24.7%  | 40%    | 20%    | 16%    | 17.5%          | 0.121  |

PDE residual (RMS): PINN 0.15-0.27 x G (momentum) and ~0.5 x S (energy) in
every row; plain NN 1.8-9 x G and 7-15 x S.

What it shows:

- **Dense data: physics is not needed for accuracy.** The plain network fits
  the CFD 2-3x more closely; the PINN's advantage is physical consistency
  (equations satisfied ~10x better), not lower error.
- **1% of the data: the PINN wins in every seed** on velocity, temperature
  and the worst Nusselt error (mean 5% vs 18%). This is the regime the
  research question is about.
- **Lines only: the PINN reconstructs the flow ~7x better**, but neither
  model gets the Nusselt numbers right (PINN under-predicts all by 12-18%):
  four lines do not carry enough thermal information near the rod.
- The crossover between "physics hurts slightly" and "physics helps clearly"
  lies between 10% and 1% of the cells.

Correction to an earlier claim: an early comparison said that without the
physics phase "dp/dz is 36% off and the iso-flux Nusselt numbers are
meaningless". That compared the plain network's untrained pressure-gradient
parameter and computed iso-flux Nu from the network's own wall flux. With
fair metrics (dp/dz from the wall shear; iso-flux Nu with the imposed flux,
as the DNS defines it) the plain network is fine when data is dense.

Still to do: seeds for the lines-only and 10% rows; the same table on the
student's own CFD data (notebook Step 6).

### Explainability, first results (test copy, full-data PINN)

Energy budget - average share of the heat flux carried by turbulence:

| Pr    | near wall (<1 mm) | narrow gap | subchannel centre | bulk |
|-------|-------------------|------------|-------------------|------|
| 0.025 | 0.00              | 0.06       | 0.50              | 0.29 |
| 1     | 0.06              | 0.63       | 0.98              | 0.89 |
| 2     | 0.10              | 0.76       | 0.99              | 0.93 |
| 7     | 0.22              | 0.91       | 1.00              | 0.98 |

This is the model's own explanation of the CFD-vs-DNS pattern: for liquid
metal the gap heat transport is 94% conduction, so the RANS gap-turbulence
error barely matters (Nu within 2% of the DNS); for Pr >= 1 turbulence
carries most of the heat even in the gap, so the missing gap mixing shows up
(Nu 13-45% low).

Residual maps - RMS residual relative to the driving term: momentum 0.13-0.21
everywhere; energy 0.2-0.6 for Pr = 0.025, up to ~1 near the wall for Pr = 1
and ~4.6 near the wall for Pr = 7. The thin near-wall thermal layer at high
Pr is where the PINN is least physically consistent.

Sensor importance: implemented (notebook Step 8), not yet run at full length.

## Results on the student's own CFD data (26 Sep 2026)

Note (26 Sep, 2nd session): these were made before the nu_t loss fix; the model default
changed, so the notebook re-trains everything and these numbers will be
replaced by the next full run. Keep them as the "before" record.


These supersede the test-copy numbers above for the paper. Same held-out
cells (998) for every run; Nu error = worst |PINN/CFD - 1| over the six DNS
cases (Pr <= 2); dp/dz from each network's wall shear (CFD 0.1820).

| Training data | Model | held-out w | held-out T | worst Nu error | dp/dz |
|---|---|---|---|---|---|
| 100% | PINN | 0.5% | 1.4% | **1.3%** | 0.1823 |
| 100% | plain NN | **0.2%** | **0.5%** | 2.4% | 0.1818 |
| 10% | PINN | 0.6% | 1.3% | 4.0% | 0.1801 |
| 10% | plain NN | **0.2%** | **0.6%** | **2.5%** | 0.1811 |
| 1%, 3 seeds | PINN | **1.1-1.5%** | **1.6-2.6%** | **1.8-9.5% (mean 5.4%)** | 0.169-0.177 |
| 1%, 3 seeds | plain NN | 1.7-2.4% | 1.9-3.4% | 10.7-34.0% (mean 18.5%) | 0.169-0.191 |
| 4 lines | PINN | **3.8%** | **9.8%** | 18.0% | 0.147 |
| 4 lines | plain NN | 24.8% | 16.3% | 18.9% | 0.118 |

Conclusions (they match the test copy):

1. **Dense data (100%, 10%)**: the plain network reproduces the CFD fields
   2-3x more closely; the PINN's fields satisfy the equations ~10x better.
   At 100% the PINN has the better Nusselt number (1.3% vs 2.4%).
2. **1% of the data (48 cells)**: the PINN wins in all three seeds on
   velocity, temperature and Nusselt number (mean worst Nu error 5.4% vs
   18.5%). **This is the answer to the research question.**
3. **Lines only**: the PINN reconstructs the flow ~6.5x better (3.8% vs
   24.8%) and the temperature 1.7x better, but neither gets Nu below ~18%:
   the lines carry too little near-wall thermal information, and neither
   model gets the pressure gradient right (-20% / -35%).
4. The crossover lies between 10% and 1% of the cells: physics costs a
   little fit accuracy when data is plentiful and pays off when it is scarce.

Explainability on the student's full-data PINN (identical to the test copy
to 2 decimals):

- **Energy budget** - turbulent share of the heat flux, narrow gap / subchannel
  centre: Pr 0.025: 0.06 / 0.50; Pr 1: 0.63 / 0.98; Pr 2: 0.76 / 0.99;
  Pr 7: 0.90 / 1.00. Explains the CFD-vs-DNS pattern: liquid-metal gap heat
  transport is 94% conduction, so the RANS gap-turbulence error barely
  matters (Nu within 2% of DNS); for Pr >= 1 turbulence dominates even in the
  gap, so the missing gap mixing shows (Nu 13-45% low).
- **Residual maps** - RMS relative to the driving term: momentum 0.13-0.21
  everywhere; energy 0.21-0.60 (Pr 0.025), 0.33-1.09 (Pr 1), 0.95-4.6
  (Pr 7, worst in the near-wall layer). Trust is lowest in the thin thermal
  wall layer at high Pr.
- **Sensor importance** (lines-only PINN, one line removed, one seed):

  | removed line | velocity error x | temperature error x | worst Nu error |
  |---|---|---|---|
  | seg1 (narrow gap) | 0.96 | **1.36** | 16.9% |
  | seg2 (gap centreline to subchannel centre) | 1.01 | 1.00 | 15.9% |
  | seg3 (subchannel centre to rod at 45 deg) | **2.03** | 1.28 | **29.2%** |
  | line15 (wall-normal at 15 deg) | 1.23 | 0.94 | 24.9% |

  seg3 is the most valuable measurement (velocity error doubles, Nu error
  18% -> 29% without it): it crosses the fastest flow and the 45-deg wall
  layer. seg1 (the gap) is the most valuable for temperature (the gap hot
  spot). seg2 adds nothing once the others are present - redundant.
  Differences below ~10% are within run-to-run noise (single seed).

## Explainability (XAI)

Standard XAI tools (SHAP, LIME) are built for models whose inputs are
features such as temperature or power. Here the inputs are coordinates
(x, y), so "which input mattered" has little physical meaning. The
explanations that do mean something for a PINN:

1. **Physics-residual maps** - where in the cross-section the network
   violates the momentum and energy equations. Shows where not to trust it.
2. **Energy-budget decomposition** - split the learned heat transport into
   molecular conduction (nu/Pr grad T) and turbulent diffusion
   (nu_t/Pr_t grad T), mapped over the cell. Explains, from the model itself,
   why liquid metal (conduction-dominated) matches the DNS and Pr 1-2 do not.
3. **Sensor importance** - in the sparse-data runs, remove one measurement
   location at a time and measure how much the Nusselt error grows. Answers
   "where should thermocouples go?", which is directly useful to reactor
   instrumentation.
4. **Uncertainty (deep ensemble)** - train 5 networks with different seeds;
   their spread is a confidence map. Should be largest where data is sparse
   and in the narrow gap.

Priority: 1 and 2 are cheap (a few minutes on a trained model); 3 comes free
with the sparse-data runs; 4 costs 5x training time and is optional.

## Rules (to keep the work honest)

- DNS data is never used for training in this study - evaluation only.
- ML hyperparameters are tuned only against CFD data (held-out cells, CFD
  dp/dz, CFD Nusselt), never against the DNS. All tuning runs are reported.
- Physical inputs (nu, Pr_t, mesh, geometry) are changed only with a written
  physical justification, and the effect of the change is reported
  (e.g. nu: 8.0099e-6 -> 7.2653e-6, see the session report).
- The final numbers come from runs on the student's machine; results from
  the independent test copy are labelled as such.
- A negative result (physics does not help, an XAI method shows nothing) is
  reported, not dropped.

## Order of work

1. Done - full-data PINN on the student's machine (baseline).
2. Done - sparse-data runs tested on the independent copy.
3. Done - sparse-data settings and notebook Step 6; full table run on the
   student's data (26 Sep).
4. Done - XAI 1 (residual maps), 2 (energy budget), 3 (sensor importance).
   Not done: 4 (uncertainty ensemble, optional).
5. Done (code, 26 Sep, 2nd session) - extra seeds 1-2 for the 10%, 1%, lines and sensor
   rows; sensor importance averaged over seeds. To run: student's machine.
6. Done on the copy (26 Sep, 2nd session) - mesh-refinement study with GCI
   (`docs/dns_comparison.md` section 2). To run: `cfd/openfoam/mesh_study.py`
   on the student's machine.
7. Done (26 Sep, 2nd session) - DNS Figs. 6, 7, 9a, 11a, 12a digitized reproducibly;
   three-layer validation (CFD vs DNS, PINN vs CFD, PINN vs DNS) in
   `ml/src/validation.py`; explanations in `docs/dns_comparison.md`.
8. Done (26 Sep, 2nd session) - nu_t gap fix: data error relative to nu + nu_t (gap-centre
   nu_t error +79% -> +13% on the copy). Requires retraining everything.
9. Open - student reruns the notebook (all steps, ~2 nights) with the new
   default and seeds; then the final tables replace the 26 Sep numbers.
10. Open - update the old project-plan PDF and professor docx; paper draft.

Optional later: reverse PINN (learn a nu_t correction from DNS data), only
with a strict split - train on some DNS quantities, test on others.
