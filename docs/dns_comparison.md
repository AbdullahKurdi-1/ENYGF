# CFD and PINN vs the 2023 DNS - numbers and explanations

Reference: Mathur, Kwiatkowski, Potempski & Komen (2023), IJHMT 211, 124226.
DNS curves digitized reproducibly by `digitized_data/digitize_dns_figures.py`
(checks: axis calibration within ~0.3% of every gridline; Figs. 6 and 11a
average to 1.004 / 1.001 over -45..45 deg, as their definition requires).
Numbers below are from the independent copy of the CFD case (medium mesh,
viscosity 0.0712/9800), which reproduces the student's v13 CFD to 0.1-0.5%;
the student's own run of `validation.validate()` (notebook Step 5) gives the
final values. Nothing in the CFD or PINN was tuned to the DNS.

## 1. Conservation and textbook checks (`validation.physics_checks`)

| Check | CFD | Reference | Ratio |
|---|---|---|---|
| Bulk velocity | 1.000 | 1 (imposed) | 1.000 |
| Pressure gradient from wall shear vs solver log | 0.18223 | 0.18223 | 1.000 |
| Iso-T heat balance (wall heat in = sink) | - | - | error 2e-8 |
| U+ at r+ = 2 (45 deg) | 2.02 | U+ = r+ | 1.01 |
| U+ at r+ = 50 / 150 (45 deg) | 15.4 / 18.3 | log law 14.7 / 17.4 | 1.04 / 1.05 |
| Friction factor 8 u_tau^2 / U_b^2 | 0.0286 | Blasius 0.0318 (DNS: 0.0325) | 0.90 |
| Nu iso-T, Pr = 1 / 2 / 7 | 33.4 / 45.4 / 74.2 | Dittus-Boelter 35.9 / 47.3 / 78.1 | 0.93 / 0.96 / 0.95 |
| Nu iso-T, Pr = 0.025 | 7.71 | Seban-Shimazaki 7.04 | 1.10 |
| Nu iso-flux, Pr = 0.025 | 4.88 | Lyon 9.04 (DNS 8.85) | 0.54 |

The CFD is physically sound: exact near-wall behaviour, log law, conservation.
The one clear outlier is iso-flux heat transfer, explained in section 4.

## 2. Mesh (numerical) uncertainty (`cfd/openfoam/mesh_study.py`)

Coarse 1 200 / medium 4 800 (production) / fine 19 200 cells, ratio 2,
monotonic convergence in every quantity. GCI after Celik et al. (2008).

| Quantity | coarse | medium | fine | extrapolated | medium vs fine | GCI medium |
|---|---|---|---|---|---|---|
| dp/dz | 0.1774 | 0.1822 | 0.1854 | 0.1917 | -1.7% | 6.5% |
| gap wall shear tau/tau_m | 0.471 | 0.465 | 0.464 | 0.463 | +0.3% | 0.5% |
| Nu Pr 0.025 iso-T | 7.73 | 7.71 | 7.68 | 7.64 | +0.4% | 0.4% |
| Nu Pr 1 iso-T | 32.4 | 33.4 | 34.0 | 35.1 | -1.8% | 6.5% |
| Nu Pr 1 iso-flux | 25.0 | 25.4 | 25.7 | 25.9 | -0.9% | 2.2% |
| Nu Pr 2 iso-T | 43.9 | 45.4 | 46.5 | 49.2 | -2.4% | 9.7% |
| Nu Pr 7 iso-T | 71.8 | 74.2 | 76.6 | 82.5 | -3.2% | 9.6% |

First-cell y+ (max): 1.8 / 0.9 / 0.5. The observed order of convergence is
low (0.5-1) for the high-Pr Nusselt numbers: the thin thermal wall layer
(conduction sublayer ~2-3 wall units at Pr = 7) is resolved by only a few
cells on the medium mesh. Consequence for the comparison below: for iso-T
Pr 1-2 part of the Nu deficit vs the DNS (roughly a third to a half) is
discretisation error; the gap and iso-flux differences are far larger than
the mesh uncertainty and are model error.

## 3. Against the DNS figures (`validation.validate`)

RMSE in each figure's own units. PINN = full-data PINN (test copy).

| DNS figure | case | CFD vs DNS | PINN vs CFD | PINN vs DNS |
|---|---|---|---|---|
| 6 wall shear tau/tau_m | - | 0.184 | 0.004 | 0.181 |
| 7 U+(r+) | 0 deg | 0.37 | 0.22 | 0.50 |
| 7 U+(r+) | 15 deg | 0.60 | 0.07 | 0.59 |
| 7 U+(r+) | 45 deg | 0.56 | 0.06 | 0.48 |
| 9a k+ | - | 0.90 | 0.04 | 0.89 |
| 11a phi/phi_m | Pr 0.025 | 0.031 | 0.001 | 0.036 |
| 11a phi/phi_m | Pr 1 | 0.158 | 0.001 | 0.173 |
| 11a phi/phi_m | Pr 2 | 0.156 | 0.002 | 0.170 |
| 12a Theta/Theta_b | Pr 0.025 | 0.32 | 0.02 | 0.33 |
| 12a Theta/Theta_b | Pr 1 | 0.17 | 0.01 | 0.17 |
| 12a Theta/Theta_b | Pr 2 | 0.15 | 0.01 | 0.15 |

The PINN reproduces the CFD closely (PINN vs CFD is 10-100x smaller than CFD
vs DNS in every row except the gap velocity profile): **the disagreement with
the DNS is the RANS model's, not the machine learning's.**

Key points (CFD / DNS): gap wall shear 0.47 / 0.81; wall shear at 45 deg
1.37 / 1.18; k+ at gap centre 0.26 / 2.50; k+ at xi/Dh = 0.6 2.56 / 2.44;
k+ at subchannel centre 1.22 / 1.00; Theta/Theta_b at gap centre
0.18 / 0.18 (Pr 0.025), 0.52 / 0.88 (Pr 1), 0.61 / 0.94 (Pr 2); mean u_tau
0.0598 / 0.0637.

## 4. Why the CFD differs from the DNS

1. **Narrow gap: too little turbulence, too slow, too little wall shear.**
   The DNS has a gap vortex street - large coherent structures crossing the
   gap at 3.7 Hz (St = 0.52) - which carries fast fluid and heat into the gap
   and makes <u'u'> as large as <w'w'> at the gap centre (DNS Fig. 9a). A
   steady RANS model cannot have it (no time), the symmetry plane through
   the gap forbids its antisymmetric motion even in an unsteady run, and a
   linear eddy-viscosity model (k-omega SST) produces k only from local mean
   shear, which is small in the gap. Result: k in the gap 10x too low, gap
   wall shear 0.47 instead of 0.81 of the mean. Two independent confirmations:
   the 2018 URANS with the same model gives the same gap velocity (0.55 U_b),
   and the literature on tight lattices reports that linear eddy-viscosity
   models mispredict the wall-shear distribution there (e.g. Baglietto &
   Ninokata 2005, Nucl. Eng. Des. 235, cited in the 2018 paper - read it
   before citing it for this specific point). The mesh study shows it is not mesh error
   (gap wall shear changes 0.3% from medium to fine).
2. **Iso-flux Nusselt numbers 30-45% low.** With the heat flux imposed
   everywhere, the heat entering the gap wall must be carried out of the gap
   laterally - by the missing gap mixing. The CFD gap wall's excess temperature
   (T_w - T_b) is ~2x the rod average (the DNS wall temperature varies by
   roughly +-15% around its mean in its Fig. 11b, although its temperature
   reference is not stated), which
   raises the mean wall temperature and lowers Nu. The DNS liquid-metal value
   (8.85) agrees with the Lyon correlation (9.04); the CFD's 4.88 does not.
3. **Iso-T Pr 1-2 Nusselt 13-14% low.** Part numerical (mesh study:
   extrapolated values are 5-9% below the DNS), part the gap deficit: the
   energy budget (explainability) shows turbulence carries 63-76% of the heat
   even in the gap at Pr 1-2, so under-predicted gap turbulence means
   under-predicted heat transfer. The wall heat flux distribution (Fig. 11a)
   is too low in the gap for the same reason.
4. **Liquid metal (Pr 0.025) - the good agreement is partly compensating
   errors.** Iso-T Nu (7.71 vs 7.54) and the gap temperature (0.18 vs 0.18)
   match, because 94% of the gap heat transport is conduction (energy
   budget), so the gap turbulence error hardly matters. But the subchannel-
   centre peak Theta/Theta_b is 1.78 vs 2.32. Two sensitivity tests on the
   copy (not used for any result):
   - Kays' turbulent Prandtl number for liquid metals (Pr_t = 0.85 + 0.7/Pe_t
     instead of 0.9) improves the profile shape (gap/centre ratio 0.084 vs
     DNS 0.079; constant 0.9 gives 0.100) but lowers Nu to 6.58 (-13% vs DNS).
   - Pure conduction (no turbulent heat transport at all) still gives a
     centre peak of only 2.00. So the peak shortfall is not a thermal-model
     issue: it comes from the velocity field through the velocity-weighted
     bulk temperature (the DNS puts much more flow through the gap, 0.85 vs
     0.55 U_b), and possibly from the DNS computing Theta_b over its whole
     domain. Conclusion: with Pr_t = 0.9 the thermal-closure error and the
     velocity-field error partly cancel in Nu.
5. **Friction velocity 6% low** (0.0598 vs 0.0637; extrapolated to zero mesh
   size 0.0613, -4%). The remainder is consistent with the redistributed
   wall shear (point 1) and with the DNS's side walls, which a periodic unit
   cell does not have.
6. **Near-wall k peak at 45 deg about half the DNS value.** A known k-omega
   SST trait (it under-predicts the near-wall TKE peak); the mean velocity
   there is still right (U+ matches the log law within 4-5%).

## 5. What agrees

Viscous sublayer and log law; bulk quantities; the shape of the wall shear
and wall heat flux distributions (minimum in the gap, maximum near 45 deg);
TKE away from the gap (xi/Dh = 0.6: 2.56 vs 2.44); liquid-metal iso-T Nu and
gap temperature; heat transfer trends with Pr; Nu within 4-7% of
Dittus-Boelter for Pr >= 1.
