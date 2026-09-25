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

Expected (not yet tested): physics keeps accuracy as data shrinks while the
plain NN degrades. If physics does not help at low data, that is a valid
result and is reported as such.

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
