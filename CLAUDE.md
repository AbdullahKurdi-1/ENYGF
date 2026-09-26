# Working rules for this project

ENYGF conference project (AI & Nuclear Symbiosis): PINN reconstruction of flow
and heat transfer in a tight rod bundle from sparse CFD data, validated
against the Mathur et al. 2023 DNS. Plan and current results:
`docs/research_plan.md`. History: `docs/sessions/`.

## How to work with the student

- End every reply with numbered steps: what exactly to do, and why.
- The student is new to ML and runs everything on a laptop (WSL Ubuntu,
  OpenFOAM v13, Jupyter at localhost:8890, kernel "enygf"). Give exact
  commands and say which terminal to use.
- Speak plainly and honestly; say when something is uncertain or was wrong.

## Session log (required)

At the end of every session, add `docs/sessions/<date>_session.md` (date of
the session, or a range) with: starting point, what was done in order,
key results with numbers, decisions, limitations found, next steps. Add a
row to `docs/sessions/README.md`, and update the status in
`docs/research_plan.md`. Commit and push.

## Research rules

- DNS data is never used for training - evaluation only.
- ML hyperparameters are tuned only against CFD data (held-out cells, CFD
  dp/dz, CFD Nusselt numbers). Report every tuning run.
- Physical inputs (nu, Pr_t, mesh, geometry) change only with a written
  physical justification, and the effect is reported.
- Keep three errors separate: PINN vs CFD, CFD vs DNS, PINN vs DNS.
- Report negative results and corrections.
- Test changes on an independent copy of the CFD case before asking the
  student to spend laptop hours; final numbers come from the student's runs.

## Code

- CFD: `cfd/openfoam/` (v13 syntax). `extract_profiles.py` writes
  `digitized_data/cfd_generated/` (field, Nusselt, line CSVs).
- ML: `ml/src/` - `model.py`, `physics.py`, `train.py`, `evaluate.py`,
  `experiments.py` (sparse-data table), `xai.py` (explainability).
  Notebook: `ml/notebooks/pinn_walkthrough.ipynb` (Run All is safe; finished
  runs are reused).
- Tests: `cd ml && python3 -m pytest tests -q`.
