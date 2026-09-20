# Digitized profile data from Shams & Kwiatkowski (2018)

These CSVs are empty except for headers. Fill them in using
[WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) (free, browser-based)
against the actual PDF figures, page by page:

| File | Source figure(s) | x-axis | y-axis | Sweep |
|---|---|---|---|---|
| `velocity_line1_by_Re.csv` | Fig. 11 | y_star (line 1, [0,1]) | U/U_bulk | 11 Re cases, Table 2 |
| `tke_line1_by_Re.csv` | Fig. 13 | y_star (line 1, [0,1]) | k/U_bulk^2 (or as-plotted units) | 11 Re cases, Table 2 |
| `temperature_line1_constT_by_Pr.csv` | Fig. 20 (top) | y_star (line 1) | theta = (T-Tw)/(Tb-Tw) | 4 Pr, Table 4 rows 15-18 |
| `temperature_line2_constT_by_Pr.csv` | Fig. 20 (bottom) | y_star (line 2) | theta | 4 Pr, Table 4 rows 15-18 |
| `temperature_line1_constq_by_Pr.csv` | Fig. 22 | y_star (line 1) | theta (Neumann form) | 4 Pr, Table 4 rows 19-22 |
| `temperature_line2_constq_by_Pr.csv` | Fig. 22 | y_star (line 2) | theta (Neumann form) | 4 Pr, Table 4 rows 19-22 |

## Workflow per figure

1. Open the PDF page in WebPlotDigitizer, calibrate the two axes.
2. Digitize each curve (one curve = one Re or one Pr value) separately.
3. Export as CSV, then reshape into this project's schema:
   `case_id,x,y` where `case_id` is the Re case name (e.g. `R5`) or the Pr
   value (e.g. `2`), matching the identifiers used in
   `paper_reference/table2_reynolds_scaling.csv` /
   `table4_thermal_cases.csv`.
4. Append rows to the matching file here — don't rename the headers, the
   loader in `ml/src/dataset.py` depends on them.

This is manual, unglamorous work, but it's what turns "we used this paper as
inspiration" into "we validated against the actual published data" — the
difference that matters for a conference submission.

If you'd rather not hand-digitize every curve, digitize just 2-3
representative cases per figure (e.g. R1, R5, R32 for the Re sweep; Pr=1 and
Pr=7 for the thermal sweep) — the ML pipeline works with partial coverage,
it just validates less densely.
