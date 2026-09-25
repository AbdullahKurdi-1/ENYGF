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

1. Full-data PINN on the student's machine (baseline; in progress).
2. Test sparse-data runs on the independent copy (check the idea works).
3. Add `data_fraction` / sensor-selection settings and a notebook section;
   run the table on the student's machine.
4. XAI 1 and 2 on the best model; XAI 3 from the sparse runs; 4 if time.
5. Mesh-refinement check of the CFD (needed for any CFD paper).
6. Digitize DNS Figs 6, 7, 11a, 12a for independent validation.

Optional later: reverse PINN (learn a nu_t correction from DNS data), only
with a strict split - train on some DNS quantities, test on others.
