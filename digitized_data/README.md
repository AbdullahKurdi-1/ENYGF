# Data for the PINN

## `cfd_generated/cfd_profiles.csv` — training data

Written by `cfd/openfoam/extract_profiles.py` from your OpenFOAM run. Don't
edit by hand. Columns: `case_id, line, s, xi, x, y, quantity, value, Re, Pr, bc`
(quantity = `w` axial velocity, `k`, `nut`, or `T`).

## `dns_*.csv` — independent validation data (never used for training)

Points traced off figures in Mathur et al. (2023), the DNS paper. Fill them
in with [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) (free,
browser-based): calibrate the axes, trace one curve at a time, export, and
paste into the matching file as `case_id,x,y` rows. Leave the header line.

| File | DNS figure | case_id | x | y |
|---|---|---|---|---|
| `dns_fig6_wall_shear.csv` | Fig. 6 | `dns` | angle from narrow gap [deg] | tau_w / tau_w,m |
| `dns_fig7_velocity_wall_units.csv` | Fig. 7(b)-(d) | angle: `0`, `15` or `45` | r+ | U+ |
| `dns_fig11a_wall_heat_flux_isoT.csv` | Fig. 11(a) | Pr: `0.025`, `1` or `2` | angle from narrow gap [deg] | phi / phi_m |
| `dns_fig12a_temperature_isoT.csv` | Fig. 12(a) | Pr: `0.025`, `1` or `2` | xi / Dh (as plotted) | Theta / Theta_b |

Notes:
- Angles may be negative or run past 45°; the evaluation folds them onto the
  0-90° range of the unit cell.
- Fig. 7 is on a log r+ axis - set WebPlotDigitizer's x axis to log scale.
- You don't need every curve. One Pr value per temperature figure and two
  angles for Fig. 7 already give a meaningful check. Start with Fig. 7 and
  Fig. 12(a).
- Fig. 12(b) (iso-flux) is left out on purpose: its temperature reference is
  not clearly defined in the text, so any comparison would be ambiguous.

Also available without any digitizing: the DNS Nusselt numbers (Table 1),
already in `../paper_reference/table_dns_2023_nusselt.csv` and compared
automatically by `ml/src/evaluate.py`.

The 2018 calibration paper's figures are no longer used: they are URANS
results, so validating a RANS-trained model against them would say little
now that DNS data for the same case exists.
