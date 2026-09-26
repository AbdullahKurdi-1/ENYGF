# Data for the PINN

## `cfd_generated/cfd_profiles.csv` — training data

Written by `cfd/openfoam/extract_profiles.py` from your OpenFOAM run. Don't
edit by hand. Columns: `case_id, line, s, xi, x, y, quantity, value, Re, Pr, bc`
(quantity = `w` axial velocity, `k`, `nut`, or `T`).

## `dns_*.csv` — independent validation data (never used for training)

Curves of Mathur et al. (2023), the DNS paper, digitized **reproducibly** by
`digitize_dns_figures.py` (run it with the path to the paper PDF). It reads
each figure's embedded image at full resolution, finds the axes frame,
calibrates the axes from the printed limits (checked against the gridlines:
every gridline lands within ~0.3% of its value) and picks each curve by
colour, column by column. Accuracy ~0.5% of the axis range; steep near-wall
parts are captured less well than plateaus.

| File | DNS figure | case_id | x | y |
|---|---|---|---|---|
| `dns_fig6_wall_shear.csv` | Fig. 6 (present DNS, red) | `dns` | angle from narrow gap [deg] | tau_w / tau_w,m |
| `dns_fig7_velocity_wall_units.csv` | Fig. 7(b)-(d) | angle `0`, `15`, `45` | r+ (local u_tau) | U+ |
| `dns_fig9a_normal_stresses.csv` | Fig. 9(a) | `uu`, `vv`, `ww` | xi / Dh | Reynolds normal stress in wall units |
| `dns_fig11a_wall_heat_flux_isoT.csv` | Fig. 11(a) | Pr `0.025`, `1`, `2` | angle [deg] (every 5 deg, the markers) | phi / phi_m |
| `dns_fig12a_temperature_isoT.csv` | Fig. 12(a) | Pr `0.025`, `1`, `2` | xi / Dh | Theta / Theta_b |

Self-checks (printed by the script): Figs. 6 and 11(a) are normalised by their
own mean over -45..45 deg, and the digitized curves average to 1.004 and
1.001 there. In Fig. 7 the published DNS curve itself starts at U+ ~ 0.85 at
r+ = 1 (below U+ = r+ and below the other DNS drawn in that figure) - a
property of the figure, reproduced faithfully.

Not digitized, on purpose: Fig. 11(b) and 12(b) (iso-flux temperatures -
their temperature reference is not defined in the text, so any comparison
would be ambiguous), Figs. 9(b), 10, 13-17 (quantities a RANS model does not
produce), Fig. 19 (pulsation spectrum; a steady model has none).

Only |theta| <= 45 deg of Figs. 6 and 11(a) is compared: that is the DNS's
own averaging cell; beyond it the DNS side walls, which an infinite-array
unit cell does not have, change the flow.

Also available without any digitizing: the DNS Nusselt numbers (Table 1),
already in `../paper_reference/table_dns_2023_nusselt.csv` and compared
automatically by `ml/src/evaluate.py`.

The 2018 calibration paper's figures are no longer used: they are URANS
results, so validating a RANS-trained model against them would say little
now that DNS data for the same case exists.
