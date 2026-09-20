"""Generate rod.stl: a cylinder along z representing the solid rod.

Geometry from paper_reference/table5_finalized_params.csv:
  rod diameter D = 0.14 m -> radius = 0.07 m, centered at the unit-cell origin.

The cylinder is extruded slightly beyond the background box's z-extent
(BOX_LZ) on both ends so snappyHexMesh sees it fully penetrating the domain
rather than terminating mid-mesh.
"""
import numpy as np

ROD_RADIUS_M = 0.07
BOX_LZ_M = 0.01
Z_OVERSHOOT_M = 0.2 * BOX_LZ_M
N_SEGMENTS = 96

OUT_PATH = "rod.stl"


def cylinder_facets(radius, z_lo, z_hi, n_segments):
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    facets = []
    for i in range(n_segments):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n_segments]
        p0_lo = (x0, y0, z_lo)
        p1_lo = (x1, y1, z_lo)
        p0_hi = (x0, y0, z_hi)
        p1_hi = (x1, y1, z_hi)
        # two triangles per quad, outward-facing normal (radial)
        facets.append((p0_lo, p1_lo, p1_hi))
        facets.append((p0_lo, p1_hi, p0_hi))
    return facets


def write_stl(path, facets, solid_name="rod"):
    with open(path, "w") as f:
        f.write(f"solid {solid_name}\n")
        for a, b, c in facets:
            ax, ay, az = a
            bx, by, bz = b
            cx, cy, cz = c
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm = (nx**2 + ny**2 + nz**2) ** 0.5
            if norm > 0:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for px, py, pz in (a, b, c):
                f.write(f"      vertex {px:.6e} {py:.6e} {pz:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


if __name__ == "__main__":
    z_lo = -Z_OVERSHOOT_M
    z_hi = BOX_LZ_M + Z_OVERSHOOT_M
    facets = cylinder_facets(ROD_RADIUS_M, z_lo, z_hi, N_SEGMENTS)
    write_stl(OUT_PATH, facets)
    print(f"wrote {len(facets)} facets to {OUT_PATH}")
