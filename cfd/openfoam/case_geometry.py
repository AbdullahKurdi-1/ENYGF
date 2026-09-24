"""Shared geometry and case constants for the OpenFOAM scripts.

Must match unit_cell/system/blockMeshDict, unit_cell/system/functions and
unit_cell/constant/physicalProperties.
"""
import math

ROD_RADIUS = 0.07
HALF_PITCH = 0.0775
QUARTER_AREA = HALF_PITCH**2 - math.pi * ROD_RADIUS**2 / 4
QUARTER_WETTED = math.pi * ROD_RADIUS / 2
DH_CELL = 4 * QUARTER_AREA / QUARTER_WETTED        # 0.078497 m
DH_DNS = 0.0712                                     # 2023 DNS domain

U_BULK = 1.0
RE = 9800.0
# Same fluid and Re definition as the 2023 DNS: Re_h = U_b * Dh_DNS / nu.
NU = U_BULK * DH_DNS / RE                           # 7.2653e-06

PR_VALUES = [0.025, 1.0, 2.0, 7.0]
PR_TURBULENT = 0.9
WALL_HEAT_FLUX = 1.0                                # iso-flux BC, W/m2 (rho*cp = 1)
SINK_ISO_T = 1.0                                    # 2023 DNS: 1 W/m3 for iso-temperature
SINK_ISO_FLUX = WALL_HEAT_FLUX * QUARTER_WETTED / QUARTER_AREA   # = 4q/Dh = 50.96, matches DNS

_EPS = 1e-6
_c = ROD_RADIUS / math.sqrt(2)
_a = HALF_PITCH - _EPS
_deg15 = math.radians(15)

# (start, end) in the cross-section plane; xi_offset = start of the segment
# along the DNS unit-cell-boundary coordinate xi (None if not on that path).
SAMPLE_LINES = {
    "seg1": ((ROD_RADIUS + _EPS, _EPS), (_a, _EPS), 0.0),
    "seg2": ((_a, _EPS), (_a, _a), HALF_PITCH - ROD_RADIUS),
    "seg3": ((_a, _a), (_c * (1 + 1e-5), _c * (1 + 1e-5)), (HALF_PITCH - ROD_RADIUS) + HALF_PITCH),
    "line15": (
        (ROD_RADIUS * (1 + 1e-5) * math.cos(_deg15), ROD_RADIUS * (1 + 1e-5) * math.sin(_deg15)),
        (_a, _a * math.tan(_deg15)),
        None,
    ),
}


def case_name(pr, bc):
    return f"Pr{pr:g}_{bc}".replace(".", "p")


def field_name(pr, bc):
    return "T_" + case_name(pr, bc)
