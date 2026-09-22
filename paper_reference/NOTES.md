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

## Simplification used in `cfd/openfoam/unit_cell`

Because the full 6-rod-with-side-walls cross-section isn't numerically
specified in the extractable text, the OpenFOAM case in this repo instead
models a single square-pitch **unit cell** (one rod, symmetry planes on all
four cell edges) — the same "infinite array" simplification the paper itself
says other authors (Cardoso de Souza et al. 2015; Chandra et al. 2010) used
for reduced-domain URANS studies. The paper's own point is that this
simplification is *not* sufficient for LES/DNS-grade fidelity (missing
gap-to-gap and wall effects) — but it is standard and adequate for a
URANS-level training-data generator, which is all we need here. This is
disclosed, not a silent shortcut: don't claim in a conference poster that this
reproduces the paper's exact domain.

Unit-cell hydraulic diameter (idealized square-pitch, one rod):
Dh = 4*(P^2 - pi*D^2/4)/(pi*D) = 0.0785 m (vs. paper's domain-implied 0.0714 m
— different because the paper's domain includes wall-bounded subchannels, not
an infinite array). The OpenFOAM case matches Re=9800 using this unit-cell Dh
with air (rho=1.2 kg/m3, mu=1.8e-5 Pa.s), giving U_bulk ~= 1.87 m/s, not the
paper's mass flow rate directly (which was for the full multi-rod domain).

## Line 1 / Line 2 profiles (Figs. 11-13, 20, 22)

The figure caption's non-dimensionalization formula came through the PDF
text extraction as `y = (yni + yni)/(2*yn)`, which is almost certainly OCR
corruption of a min-max normalization. This project treats the profile
coordinate as plain min-max: `y_star = (y - y_min)/(y_max - y_min)` in [0,1]
along each line. If you digitize the real figures and the shape doesn't match
this assumption, fix it here first — everything downstream depends on it.

## What "Line 1" and "Line 2" physically are

Not given as coordinates in the extractable text (they're defined visually on
Fig. 3/Fig. 11's diagram) beyond "across line 1 (till the mid of the gap)"
appearing in the Fig. 12 caption. Treat Line 1 as the gap-to-center line
(rod-gap to subchannel center) and Line 2 as the diagonal/secondary line, and
confirm against the actual figures when you digitize them — this is a
reasonable literature-standard convention for rod-bundle profile plots but
is not independently confirmed from text alone.
