"""Digitize the curves of Mathur et al. (2023) Figs. 6, 7, 9(a), 11(a), 12(a)
from the paper PDF, reproducibly (no hand tracing).

How it works:
  1. Extract each figure's embedded image at its native resolution.
  2. Find each panel's axes frame (long dark-grey lines) inside a crop box.
  3. Map pixels to data with the axis limits printed on the figure (linear,
     or log10 for the r+ axis of Fig. 7); checked against the gridlines.
  4. Pick each curve by its colour (red / blue / black), column by column,
     skipping legends and the black dashed marker lines; the curve value in a
     column is the median of its pixels.

Accuracy is about +-0.5% of the axis range (1 px of ~500-800 px). Steep
near-wall parts of the curves are captured less well than the plateaus.
Self-checks printed at the end: Fig. 6 and 11(a) are normalised by their mean
over -45..45 deg, so their digitized means there must be ~1; Fig. 7 must
follow U+ = r+ in the viscous sublayer.

Usage:
    python3 digitize_dns_figures.py path/to/Mathur_2023_DNS.pdf
Writes dns_fig*.csv next to this script (case_id, x, y).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Image objects (xref) of each figure inside the published PDF.
FIGURE_XREF = {"fig6": 100, "fig7": 98, "fig9": 126, "fig11": 141, "fig12": 159}


def extract_images(pdf):
    import pymupdf as fitz
    doc = fitz.open(pdf)
    out = {}
    for name, xref in FIGURE_XREF.items():
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        out[name] = arr[:, :, :3].astype(int)
    return out


def find_frame(img, box):
    """Pixel positions (left, right, top, bottom) of the axes frame inside box = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    sub = img[y0:y1, x0:x1]
    grey = (sub.max(2) - sub.min(2) < 15) & (sub.mean(2) < 200)
    cols = np.where(grey.mean(0) > 0.5)[0]
    rows = np.where(grey.mean(1) > 0.5)[0]
    if len(cols) < 2 or len(rows) < 2:
        raise RuntimeError(f"axes frame not found in box {box}")
    return x0 + cols.min(), x0 + cols.max(), y0 + rows.min(), y0 + rows.max()


class Panel:
    def __init__(self, img, box, xlim, ylim, xlog=False):
        self.img = img
        self.left, self.right, self.top, self.bottom = find_frame(img, box)
        self.xlim, self.ylim, self.xlog = xlim, ylim, xlog

    def x(self, px):
        f = (np.asarray(px, float) - self.left) / (self.right - self.left)
        if self.xlog:
            lo, hi = np.log10(self.xlim)
            return 10 ** (lo + f * (hi - lo))
        return self.xlim[0] + f * (self.xlim[1] - self.xlim[0])

    def px(self, x):
        if self.xlog:
            lo, hi = np.log10(self.xlim)
            f = (np.log10(x) - lo) / (hi - lo)
        else:
            f = (np.asarray(x, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        return self.left + f * (self.right - self.left)

    def row(self, y):
        f = (np.asarray(y, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.bottom - f * (self.bottom - self.top)

    def y(self, py):
        f = (self.bottom - np.asarray(py, float)) / (self.bottom - self.top)
        return self.ylim[0] + f * (self.ylim[1] - self.ylim[0])

    def trace(self, colour, exclude=(), exclude_cols=(), min_px=1):
        """(x, y) of the curve of a colour: median row of its pixels per column."""
        m = MASKS[colour](self.img)
        xs, ys = [], []
        for c in range(self.left + 3, self.right - 2):
            if any(abs(c - e) <= 5 for e in exclude_cols):
                continue
            rows = np.where(m[self.top + 3:self.bottom - 2, c])[0] + self.top + 3
            rows = [r for r in rows if not any(b[0] <= c <= b[1] and b[2] <= r <= b[3] for b in exclude)]
            if len(rows) >= min_px:
                xs.append(c)
                ys.append(np.median(rows))
        return self.x(np.array(xs)), self.y(np.array(ys))


MASKS = {
    "red": lambda a: (a[:, :, 0] > 170) & (a[:, :, 1] < 110) & (a[:, :, 2] < 110),
    "blue": lambda a: (a[:, :, 2] > 170) & (a[:, :, 0] < 110) & (a[:, :, 1] < 140),
    "black": lambda a: a.max(2) < 90,
}


def resample(x, y, grid):
    """Average the traced points into bins centred on `grid` (drops empty bins)."""
    x, y = np.asarray(x), np.asarray(y)
    half = np.diff(grid).min() / 2
    out = [(g, y[np.abs(x - g) <= half].mean()) for g in grid if (np.abs(x - g) <= half).any()]
    return np.array(out).T if out else (np.array([]), np.array([]))


def despike(x, y, tol):
    """Drop isolated points (legend text, labels) further than tol from the
    median of their 7 neighbours."""
    x, y = np.asarray(x), np.asarray(y)
    if len(y) < 7:
        return x, y
    med = np.array([np.median(y[max(0, i - 3):i + 4]) for i in range(len(y))])
    keep = np.abs(y - med) <= tol
    return x[keep], y[keep]


def rows_for(case_id, x, y):
    return pd.DataFrame({"case_id": case_id, "x": np.round(x, 5), "y": np.round(y, 5)})


def fig6(img):
    p = Panel(img, (100, 950, 0, 700), (-90, 90), (0.3, 1.3))
    legend = (p.px(-50), p.px(65), p.row(0.73), p.row(0.33))   # legend box, lower centre
    x, y = p.trace("red", exclude=[legend])
    gx, gy = despike(*resample(x, y, np.arange(-88, 88.1, 1.0)), tol=0.03)
    return rows_for("dns", gx, gy)


def fig7(img):
    frames = {0: ((980, 1900, 0, 700), (1, 1000), (0, 20)),     # (b) theta = 0
              15: ((0, 950, 820, 1627), (1, 1000), (0, 20)),    # (c) theta = 15
              45: ((950, 1900, 820, 1627), (1, 1000), (0, 25))}  # (d) theta = 45
    out = []
    for ang, (box, xlim, ylim) in frames.items():
        p = Panel(img, box, xlim, ylim, xlog=True)
        legend = (p.px(8), p.right, p.row(ylim[1] * 0.30), p.bottom)   # lower right
        x, y = p.trace("red", exclude=[legend])
        gx, gy = despike(*resample(np.log10(x), y, np.linspace(0, 3, 121)), tol=0.5)
        out.append(rows_for(ang, 10 ** gx, gy))
    return pd.concat(out)


def fig9a(img):
    p = Panel(img, (0, 925, 0, 700), (0, 1.75), (0, 9))
    marks = [p.px(0.105), p.px(1.194)]
    legend = [(p.left, p.px(0.6), p.top, p.row(6.0))]   # upper left
    out = []
    for comp, colour in (("uu", "blue"), ("vv", "black"), ("ww", "red")):
        x, y = p.trace(colour, exclude=legend, exclude_cols=marks)
        gx, gy = despike(*resample(x, y, np.arange(0.0, 1.751, 0.01)), tol=0.3)
        out.append(rows_for(comp, gx, gy))
    return pd.concat(out)


def fig11a(img):
    p = Panel(img, (60, 1000, 0, 700), (-90, 90), (0, 2))
    legend = [(p.left, p.px(-15), p.top, p.row(1.5))]   # upper left
    out = []
    for pr, colour in ((0.025, "red"), (1.0, "black"), (2.0, "blue")):
        x, y = p.trace(colour, exclude=legend)
        # markers every 5 deg from -85 to 85: read the curve there
        gx, gy = resample(x, y, np.arange(-85, 85.1, 5.0))
        out.append(rows_for(pr, gx, gy))
    return pd.concat(out)


def fig12a(img):
    p = Panel(img, (60, 900, 0, 600), (0, 1.75), (0, 2.5))
    marks = [p.px(0.105), p.px(1.194)]
    legend = [(p.left, p.px(0.65), p.top, p.row(1.95))]   # upper left
    out = []
    for pr, colour in ((0.025, "blue"), (1.0, "black"), (2.0, "red")):
        x, y = p.trace(colour, exclude=legend, exclude_cols=marks)
        gx, gy = despike(*resample(x, y, np.arange(0.0, 1.751, 0.01)), tol=0.08)
        out.append(rows_for(pr, gx, gy))
    return pd.concat(out)


def self_checks(df6, df7, df11):
    print("\nSelf-checks (independent of the digitizing):")
    m = df6[df6["x"].abs() <= 45]
    print(f"  Fig. 6: mean tau/tau_m over -45..45 deg = {m['y'].mean():.3f} (definition: 1)")
    for pr, g in df11.groupby("case_id"):
        m = g[g["x"].abs() <= 45]
        print(f"  Fig. 11a Pr={pr:g}: mean phi/phi_m over -45..45 deg = {m['y'].mean():.3f} (definition: 1)")
    for ang, g in df7.groupby("case_id"):
        s = g[(g["x"] >= 1) & (g["x"] <= 3)]
        err = (s["y"] / s["x"] - 1).abs().max() if len(s) else float("nan")
        print(f"  Fig. 7 theta={ang}: max |U+/r+ - 1| for r+ = 1-3 = {err:.1%} (the published DNS curve "
              "itself starts at U+ ~ 0.85 at r+ = 1, below the other DNS in that figure)")


def main(pdf):
    imgs = extract_images(pdf)
    results = {
        "dns_fig6_wall_shear.csv": fig6(imgs["fig6"]),
        "dns_fig7_velocity_wall_units.csv": fig7(imgs["fig7"]),
        "dns_fig9a_normal_stresses.csv": fig9a(imgs["fig9"]),
        "dns_fig11a_wall_heat_flux_isoT.csv": fig11a(imgs["fig11"]),
        "dns_fig12a_temperature_isoT.csv": fig12a(imgs["fig12"]),
    }
    for name, df in results.items():
        df.to_csv(HERE / name, index=False)
        print(f"{name}: {len(df)} points, cases {sorted(df['case_id'].unique(), key=str)}")
    self_checks(results["dns_fig6_wall_shear.csv"], results["dns_fig7_velocity_wall_units.csv"],
                results["dns_fig11a_wall_heat_flux_isoT.csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Mathur et al. (2023) IJHMT 211, 124226 - the PDF")
    args = parser.parse_args()
    if not Path(args.pdf).exists():
        sys.exit(f"PDF not found: {args.pdf}")
    main(args.pdf)
