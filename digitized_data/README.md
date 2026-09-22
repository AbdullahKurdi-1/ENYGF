# Digitized profile data

Two source papers now (see `../paper_reference/NOTES.md` and
`../paper_reference/dns_2023_reference.md`). **Prefer the 2023 DNS paper's
figures where a quantity exists in both** - they're real resolved
turbulence statistics, not URANS, and Fig. 7's velocity profiles are
already in wall units with experimental data overlaid. Use `DNS_` as the
case_id prefix for points from the 2023 paper so they don't collide with
the 2018 paper's `Rn` case names in the same CSV.

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
| `velocity_line1_by_Re.csv` (preferred source) | **2023 DNS paper, Fig. 7(b)-(d)** | r+ (wall units), pick one of 0/15/45 degree lines as "line 1" | U+ (already in wall units) | Re=9800 only - 3 angle cases, prefix `DNS_` |
| `tke_line1_by_Re.csv` (preferred source) | **2023 DNS paper, Fig. 9** | xi/Dh along unit cell boundary | normal Reynolds stress (not exactly k, see caveat below) | Re=9800 only, prefix `DNS_` |
| `temperature_line1_constT_by_Pr.csv` (preferred source) | **2023 DNS paper, Figs. 14-15 (iso-temperature)** | xi/Dh along unit cell boundary | theta or turbulent heat flux, per figure | Pr = 0.025, 1, 2 only (no Pr=7 in the DNS - see `dns_2023_reference.md`), prefix `DNS_` |

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

## Caveat on using DNS Fig. 9 for "k"

The DNS paper reports individual normal Reynolds stress components
(`<u'u'>`, `<v'v'>`, `<w'w'>`), not turbulent kinetic energy directly. If
you digitize Fig. 9, either sum the three components as
`k = 0.5*(<u'u'>+<v'v'>+<w'w'>)` before entering it as `y`, or treat it as
qualitative-only (confirms gap-vs-subchannel turbulence suppression, not a
number our `k` output should hit exactly).

## Why there's no `nut_*_by_Re.csv` here

The paper never reports eddy-viscosity directly — there's nothing to
digitize for it. `nut` only ever comes from your own OpenFOAM run
(`cfd/openfoam/extract_profiles.py` writes `cfd_generated/nut_line1.csv`),
because it's what identifies the PINN's own learned eddy-viscosity field —
see the conversation in the project history for why that's needed at all,
not just nice-to-have.
