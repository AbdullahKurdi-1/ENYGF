"""Minimal readers for ASCII OpenFOAM meshes and fields (no OpenFOAM needed).

Used to export the whole CFD field (every cell, not just the sampled lines)
and to compute the CFD's own Nusselt numbers. Works for the 2D extruded
quarter-unit-cell mesh (one cell thick in z); tested on OpenFOAM v13 and
ESI v1912 ASCII output.
"""
import re
from pathlib import Path

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _body(path):
    """File text after the FoamFile header, comments removed."""
    text = _strip_comments(Path(path).read_text())
    m = re.search(r"FoamFile\s*\{[^}]*\}", text)
    return text[m.end():] if m else text


def _read_list_at(text, pos):
    """Parse 'N ( ... )' starting at pos. Returns (array, end_pos)."""
    m = re.compile(r"\s*(\d+)\s*\(").match(text, pos)
    if not m:
        raise ValueError(f"expected 'N (' at: {text[pos:pos + 60]!r}")
    n, start = int(m.group(1)), m.end()
    depth, i = 1, start
    while depth:
        c = text[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    inner = text[start:i - 1]
    if "(" in inner:  # vectors
        vals = np.array([[float(v) for v in t.split()] for t in re.findall(r"\(([^()]*)\)", inner)])
    else:
        vals = np.array([float(v) for v in inner.split()])
    if len(vals) != n:
        raise ValueError(f"list says {n} entries, found {len(vals)}")
    return vals, i


def _parse_value(text, pos, n):
    """Parse 'uniform X' or 'nonuniform List<T> N (...)' at pos, broadcast to n."""
    m = re.compile(r"\s*uniform\s+(\([^)]*\)|" + _NUM + r")").match(text, pos)
    if m:
        v = m.group(1)
        v = np.array([float(x) for x in v.strip("()").split()]) if v.startswith("(") else float(v)
        return np.broadcast_to(v, (n,) + np.shape(v)).copy()
    m = re.compile(r"\s*nonuniform\s+List<\w+>").match(text, pos)
    if not m:
        raise ValueError(f"can't parse field value at: {text[pos:pos + 60]!r}")
    return _read_list_at(text, m.end())[0]


def read_internal_field(path, n_cells):
    body = _body(path)
    m = re.search(r"\binternalField\b", body)
    return _parse_value(body, m.end(), n_cells)


def read_patch_value(path, patch, n_faces, key="value"):
    """An entry of a patch, 'value' by default (e.g. the wall temperature of a
    fixedGradient rod). Returns None if the patch has no such entry."""
    body = _body(path)
    m = re.search(r"\b" + re.escape(patch) + r"\s*\{", body[re.search(r"\bboundaryField\b", body).end():])
    start = re.search(r"\bboundaryField\b", body).end() + m.end()
    depth, i = 1, start
    while depth:
        depth += body[i] == "{"
        depth -= body[i] == "}"
        i += 1
    block = body[start:i - 1]
    v = re.search(r"\b" + re.escape(key) + r"\b", block)
    if v is None:
        return None
    return _parse_value(block, v.end(), n_faces)


class Mesh:
    """Geometry of a 2D (one-cell-thick in z) polyMesh."""

    def __init__(self, case_dir):
        pm = Path(case_dir) / "constant" / "polyMesh"
        self.points = _read_list_at(_body(pm / "points"), 0)[0]
        self.faces = self._read_faces(pm / "faces")
        self.owner = _read_list_at(_body(pm / "owner"), 0)[0].astype(int)
        self.neighbour = _read_list_at(_body(pm / "neighbour"), 0)[0].astype(int)
        self.patches = self._read_boundary(pm / "boundary")
        self.n_cells = int(self.owner.max()) + 1
        self._geometry()

    @staticmethod
    def _read_faces(path):
        text = _body(path)
        if "faceCompactList" in Path(path).read_text()[:600]:
            offsets, end = _read_list_at(text, 0)
            idx, _ = _read_list_at(text, end)
            offsets, idx = offsets.astype(int), idx.astype(int)
            return [idx[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        m = re.compile(r"\s*(\d+)\s*\(").match(text)
        faces = [np.array(f.split(), dtype=int) for f in re.findall(r"\d+\(([\d\s]+)\)", text[m.end():])]
        if len(faces) != int(m.group(1)):
            raise ValueError("face count mismatch in polyMesh/faces")
        return faces

    @staticmethod
    def _read_boundary(path):
        text = _body(path)
        out = {}
        for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", text):
            n = re.search(r"nFaces\s+(\d+)", block)
            s = re.search(r"startFace\s+(\d+)", block)
            if n and s:
                out[name] = (int(s.group(1)), int(n.group(1)))
        return out

    def _geometry(self):
        n_f = len(self.faces)
        self.face_centre = np.zeros((n_f, 3))
        self.face_area = np.zeros((n_f, 3))
        for i, f in enumerate(self.faces):
            p = self.points[f]
            c0 = p.mean(0)
            q = np.roll(p, -1, axis=0)
            tri_area = 0.5 * np.cross(p - c0, q - c0)
            mag = np.linalg.norm(tri_area, axis=1)
            self.face_area[i] = tri_area.sum(0)
            self.face_centre[i] = ((p + q + c0) / 3 * mag[:, None]).sum(0) / mag.sum()
        z = self.points[:, 2]
        self.lz = z.max() - z.min()
        # One z-normal face at the lower z of each cell gives the 2D cell polygon.
        self.cell_xy = np.zeros((self.n_cells, 2))
        self.cell_area = np.zeros(self.n_cells)
        nz = np.abs(self.face_area[:, 2]) > 0.9 * np.linalg.norm(self.face_area, axis=1)
        low = nz & (self.face_centre[:, 2] < z.min() + 0.5 * self.lz)
        for i in np.where(low)[0]:
            xy = self.points[self.faces[i], :2]
            x, y = xy[:, 0], xy[:, 1]
            xn, yn = np.roll(x, -1), np.roll(y, -1)
            cr = x * yn - xn * y
            a = cr.sum() / 2
            self.cell_xy[self.owner[i]] = [((x + xn) * cr).sum() / (6 * a), ((y + yn) * cr).sum() / (6 * a)]
            self.cell_area[self.owner[i]] = abs(a)
        if (self.cell_area == 0).any():
            raise ValueError("some cells have no z-normal face - is this a one-cell-thick 2D mesh?")

    def patch_faces(self, name):
        s, n = self.patches[name]
        return np.arange(s, s + n)
