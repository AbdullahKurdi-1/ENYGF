# Two reference papers, same research program

1. Shams & Kwiatkowski (2018), *Annals of Nuclear Energy* 121, 146-161 —
   "Towards the Direct Numerical Simulation of a closely-spaced bare rod
   bundle." The calibration study. Covered below in this file.
2. Mathur, Kwiatkowski, Potempski & Komen (2023), *Int. J. Heat and Mass
   Transfer* 211, 124226 — "Direct numerical simulation of flow and heat
   transport in a closely-spaced bare rod bundle." The actual DNS the 2018
   paper was calibrating toward, run 5 years later. Covered in
   `dns_2023_reference.md` in this folder — **read that one first**, its
   published figures are the better validation target (real resolved
   turbulence statistics, not URANS).

Same geometry, same Re=9800, same NRG/NCBJ team. Reynolds-number-scaling
case-naming context below (`R1`...`R32`) is from the 2018 paper only.

## What the 2018 paper actually is

A URANS/RANS calibration study that designs (but does not itself run) a DNS of a
closely-spaced bare rod bundle. There is no released field dataset (no exported
velocity/temperature volumes). What it does give us:

1. A fully specified geometry and operating envelope (`table5_finalized_params.csv`).
2. An 11-point Reynolds-number sweep used to calibrate the case
   (`table2_reynolds_scaling.csv`), with results shown as line profiles in
   Figs. 8, 9, 11, 12, 13.
3. An 8-case thermal sweep (4 Prandtl numbers x 2 wall BCs) at the finalized
   Re = 9800 (`table4_thermal_cases.csv`), with results in Figs. 18-22.

Case naming convention: `Rn` = original Hooper bulk Reynolds number (49,000)
divided by `n`. R5 (Re=9800) is the case that was ultimately finalized for the
targeted DNS (Table 5), consistent across Tables 3-5.

**Correction**: the extracted PDF text shows the finalized axial length as
"22.85 cm", but Tables 3-4 consistently use 2.285 m for the same L/4-scaled
domain (L/4 of the original 9.14 m Hooper test section). Treated as 2.285 m
(228.5 cm) throughout this project — almost certainly an OCR-dropped digit in
the source PDF table, not a real 10x discrepancy.

## Geometry actually reconstructable from the paper text alone

- Six rods, square array, P/D = 1.107, D = 0.14 m, P = 0.155 m.
- Original Hooper test section: 9.14 m long = 128 hydraulic diameters
  => paper's own implied Dh = 9.14/128 = 0.0714 m.
- The full computational domain in the paper *includes side walls* (Fig. 3) —
  its exact outer dimensions are only given as a dimensioned drawing (Figs.
  1-3), which are images, not text. We do not have those numbers without
  digitizing the figures.

## How this project uses the 2018 paper now

For geometry and case context only: P/D, D, Re = 9800, the Prandtl numbers,
and the Reynolds-scaling background (Table 2). Its figures (URANS line
profiles) are no longer used for validation - the 2023 DNS of the same case
is a far better reference. See `dns_2023_reference.md`.

The OpenFOAM case models the quarter unit cell of an infinite square array
with symmetry planes - the same simplification this paper notes other authors
used for reduced-domain URANS (Cardoso de Souza et al. 2015; Chandra et al.
2010), and which it warns is not sufficient for LES/DNS-grade fidelity. It is
also exactly the "effective unit cell" over which the 2023 DNS averages its
statistics. Case normalisation (U_b = 1, nu = Dh/Re) follows the 2023 DNS;
see `cfd/openfoam/README.md`.
