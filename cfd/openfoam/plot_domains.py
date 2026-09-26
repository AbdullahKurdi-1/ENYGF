"""Sketch of the DNS (Hooper) domain next to this project's CFD domain.

(a) Mathur et al. 2023 DNS / Hooper experiment: six rods in a square lattice,
    two subchannels, the outer gaps closed by straight gap walls; only the
    central gap is open. Grey: the DNS "effective unit cell" its statistics
    are folded onto (their Fig. 1a).
(b) This project: an infinite square array; the computed quarter cell has
    symmetry planes on all four sides, so every gap is open.
The dashed green square in (a) is where our quarter cell sits in the DNS
domain: its 0-45 deg side (open gap -> subchannel centre) matches; its top
side is a gap wall in the DNS but a symmetry plane in ours.

Usage (from the repo root):  python3 cfd/openfoam/plot_domains.py
Writes docs/figures/domain_comparison.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

D, P = 0.14, 0.155
R, a = D / 2, P / 2
ROD, WALL, SYM, OURS = "#c0392b", "#1f5fbf", "#7f7f7f", "#1a9850"
OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "domain_comparison.png"


def rods(ax, centres, clip):
    for cx, cy in centres:
        c = Circle((cx, cy), R, facecolor="white", edgecolor=ROD, lw=1.8, zorder=3)
        ax.add_patch(c)
        c.set_clip_path(clip)


def dns_panel(ax):
    box = Rectangle((-P, -a), 2 * P, P, facecolor="#eef3fa", edgecolor="none")
    ax.add_patch(box)
    eff = Rectangle((0, 0), P, a, facecolor="#cfcfcf", edgecolor="none", zorder=1)
    ax.add_patch(eff)
    rods(ax, [(x, y) for x in (-P, 0, P) for y in (-a, a)], box)
    gap = P - D
    walls = [((-a - gap / 2, a), (-a + gap / 2, a)), ((a - gap / 2, a), (a + gap / 2, a)),
             ((-a - gap / 2, -a), (-a + gap / 2, -a)), ((a - gap / 2, -a), (a + gap / 2, -a)),
             ((-P, -gap / 2), (-P, gap / 2)), ((P, -gap / 2), (P, gap / 2))]
    for (x0, y0), (x1, y1) in walls:
        ax.plot([x0, x1], [y0, y1], color=WALL, lw=3.5, zorder=4, solid_capstyle="butt")
    ax.add_patch(Rectangle((0, 0), a, a, fill=False, edgecolor=OURS, lw=2, ls="--", zorder=5))
    ax.annotate("open gap\n(vortex street)", (0, 0), (-0.05, -0.035), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("gap wall", (a, -a), (0.115, -0.055), fontsize=8, color=WALL,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=WALL))
    ax.plot([a], [0], "k+", ms=7, zorder=6)
    ax.set_xlim(-P - 0.01, P + 0.01)
    ax.set_ylim(-a - 0.01, a + 0.01)
    ax.set_title("(a) DNS 2023 / Hooper: 6 rods, 2 subchannels,\nouter gaps closed by walls", fontsize=10)


def ours_panel(ax):
    box = Rectangle((-P, -a), 2 * P, P, facecolor="#eef3fa", edgecolor="none")
    ax.add_patch(box)
    rods(ax, [(x, y) for x in (-P, 0, P) for y in (-a, a)], box)
    ax.add_patch(Rectangle((0, 0), a, a, facecolor="#d9f0d3", edgecolor=OURS, lw=2, zorder=1))
    for x in np.arange(-P, P + 1e-9, a):
        ax.plot([x, x], [-a, a], color=SYM, lw=0.6, ls=":", zorder=2)
    for y in (-a, 0, a):
        ax.plot([-P, P], [y, y], color=SYM, lw=0.6, ls=":", zorder=2)
    ax.text(0, -a - 0.006, "array continues on every side (all gaps open)", fontsize=8,
            ha="center", va="top")
    ax.plot([a], [0], "k+", ms=7, zorder=6)
    ax.set_xlim(-P - 0.01, P + 0.01)
    ax.set_ylim(-a - 0.01, a + 0.01)
    ax.set_title("(b) This CFD: infinite square array,\nsymmetry planes (dotted) on every side", fontsize=10)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    dns_panel(axes[0])
    ours_panel(axes[1])
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [plt.Line2D([], [], color=ROD, lw=2, label="rod wall (heated)"),
               plt.Line2D([], [], color=WALL, lw=3.5, label="gap wall (no-slip, adiabatic)"),
               plt.Line2D([], [], color=SYM, lw=1, ls=":", label="symmetry plane"),
               plt.Line2D([], [], color=OURS, lw=2, ls="--", label="our computed quarter cell"),
               Rectangle((0, 0), 1, 1, facecolor="#cfcfcf", label="DNS effective unit cell"),
               plt.Line2D([], [], color="k", marker="+", ls="none", label="subchannel centre")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
