"""PINN for fully developed flow and heat transfer in a rod-bundle unit cell.

Unknowns live in the cross-section (x, y) of the quarter unit cell:
  w(x, y)    axial (streamwise) velocity
  nu_t(x, y) eddy viscosity (Boussinesq closure; supervised by CFD nu_t)
  k(x, y)    turbulent kinetic energy (data-fit only, no PDE of its own)
  G(Re)      driving axial pressure gradient, fixed by the bulk-velocity constraint
  T(x, y)    temperature, per Prandtl number and wall boundary condition

Two sub-networks, matching the papers' passive-scalar treatment: temperature
never feeds back into the flow.

Wall conditions that can be built in exactly are: w = nu_t = k = 0 at the rod,
and T = 0 at the rod for iso-temperature cases (multiplying by
tanh(d/wall_layer), d = distance from the rod).

Viscosity follows the 2023 DNS definition of the Reynolds number:
nu = U_b * Dh_DNS / Re (Dh_DNS = 0.0712 m), the same as the CFD case.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, n_features, scale, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, n_features, generator=g) * scale)

    def forward(self, xy):
        proj = 2 * math.pi * xy @ self.B
        return torch.cat([xy, torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def out_dim(self):
        return 2 + 2 * self.B.shape[1]


def mlp(in_dim, hidden, out_dim):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _log_normaliser(values):
    lo, hi = math.log(min(values)), math.log(max(values))
    if hi - lo < 1e-12:
        return lambda v: torch.zeros_like(v)
    return lambda v: 2 * (torch.log(v) - lo) / (hi - lo) - 1


class RodBundlePINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g, m, fl, th = cfg["geometry"], cfg["model"], cfg["flow"], cfg["thermal"]
        self.r = g["rod_radius"]
        self.a = g["half_pitch"]
        self.dh = g["dh_cell"]          # geometry: 4 * area / wetted perimeter of this cell
        self.dh_ref = g["dh_dns"]       # Re and Nu definitions, as in the 2023 DNS
        self.u_bulk = fl["u_bulk"]
        self.pr_t = th["pr_turbulent"]
        self.q_wall = th["wall_heat_flux"]
        self.sink_iso_t = th["sink_iso_t"]
        self.sink_iso_flux = self.q_wall * 4.0 / self.dh
        self.q_scale = self.sink_iso_t * self.dh / 4           # mean iso-temperature wall heat flux
        self.wall_layer = m["wall_layer"]
        self.nut_scale = m["nut_scale"]
        self.k_scale = m["k_scale"]
        self.g_scale = m["g_scale"]
        self.tau_scale = self.g_scale * self.dh / 4          # typical wall shear stress / rho
        self.re_norm = _log_normaliser(fl["re_values"])
        self.pr_norm = _log_normaliser(th["pr_values"])

        self.wall_scales = m["wall_feature_scales"]

        self.ff_m = FourierFeatures(m["fourier_features"], m["fourier_scale"], seed=0)
        self.ff_t = FourierFeatures(m["fourier_features"], m["fourier_scale"], seed=1)
        n_wall = len(self.wall_scales) + 1
        self.momentum_net = mlp(self.ff_m.out_dim + n_wall + 1, m["momentum_hidden"], 3)
        # One output per trained (Pr, wall BC) case: much easier to fit than a
        # single output conditioned on Pr. Predictions exist only at trained Pr.
        self.case_pr = sorted(th["pr_values"])
        self.thermal_net = mlp(self.ff_t.out_dim + n_wall + 1, m["thermal_hidden"], 2 * len(self.case_pr))
        self.g_net = mlp(1, [16], 1)

        # Temperature scale per (Pr, wall BC), so every case's network output is
        # O(1). Starts from a physics estimate; train.py replaces it with the
        # range of the CFD data (saved with the model).
        prs = sorted(th["pr_values"])
        self.register_buffer("t_log_pr", torch.log(torch.tensor(prs, dtype=torch.float32)))
        est = [[self._t_estimate(p, s) for s in (self.sink_iso_t, self.sink_iso_flux)] for p in prs]
        self.register_buffer("t_scale_table", torch.tensor(est, dtype=torch.float32))

    # ---- helpers -----------------------------------------------------------
    def nu(self, re):
        return self.u_bulk * self.dh_ref / re

    def wall_distance(self, x, y):
        return torch.sqrt(x**2 + y**2) - self.r

    def wall_factor(self, x, y):
        # No clamp: at points on the rod, rounding can make d slightly negative,
        # and a clamp would then zero the wall-normal gradient (wall shear, heat flux).
        return torch.tanh(self.wall_distance(x, y) / self.wall_layer)

    def sink(self, bc):
        return torch.where(bc > 0.5, torch.full_like(bc, self.sink_iso_flux), torch.full_like(bc, self.sink_iso_t))

    def _t_estimate(self, pr, sink, re=9800.0):
        nu = self.u_bulk * self.dh_ref / re
        return sink * self.dh**2 / (nu / pr + 20.0 * nu / self.pr_t)

    def set_temperature_scales(self, table):
        """table: (n_pr, 2) tensor, columns iso-temperature / iso-flux, rows in sorted-Pr order."""
        self.t_scale_table.copy_(torch.as_tensor(table, dtype=torch.float32))

    def t_ref(self, re, pr, bc):
        """Temperature scale for (Pr, BC); log-log interpolation between trained Pr."""
        lp = torch.log(pr)
        k = self.t_log_pr.numel()
        table = torch.log(self.t_scale_table)
        if k == 1:
            logs = table[0].expand(lp.shape[0], 2)
        else:
            lp = lp.clamp(self.t_log_pr[0], self.t_log_pr[-1])
            hi = torch.searchsorted(self.t_log_pr, lp.squeeze(1).contiguous()).clamp(1, k - 1)
            lo = hi - 1
            wgt = ((lp.squeeze(1) - self.t_log_pr[lo]) / (self.t_log_pr[hi] - self.t_log_pr[lo])).unsqueeze(1)
            logs = (1 - wgt) * table[lo] + wgt * table[hi]
        scale = torch.exp(torch.where(bc > 0.5, logs[:, 1:2], logs[:, 0:1]))
        return scale

    def _xy(self, x, y):
        return torch.cat([x / self.a, y / self.a], dim=-1)

    def _wall_features(self, x, y):
        """Multi-scale functions of wall distance. The near-wall layer is ~100x
        thinner than the cell - too fine for coordinate-based features alone."""
        d = self.wall_distance(x, y)
        feats = [torch.tanh(d / s) for s in self.wall_scales]
        feats.append(torch.log1p(torch.clamp(d, min=0.0) / self.wall_scales[0]) / 10.0)
        return torch.cat(feats, dim=-1)

    # ---- outputs -----------------------------------------------------------
    def momentum(self, x, y, re):
        """x, y, re: (N, 1). Returns w, nu_t, k, each (N, 1)."""
        feats = torch.cat([self.ff_m(self._xy(x, y)), self._wall_features(x, y), self.re_norm(re)], dim=-1)
        out = self.momentum_net(feats)
        f = self.wall_factor(x, y)
        w = f * out[:, 0:1]
        nut = f**2 * F.softplus(out[:, 1:2]) * self.nut_scale
        k = f**2 * F.softplus(out[:, 2:3]) * self.k_scale
        return w, nut, k

    def pressure_gradient(self, re):
        return F.softplus(self.g_net(self.re_norm(re))) * self.g_scale

    def case_index(self, pr, bc):
        """Column of the thermal network for each (Pr, BC); Pr must be a trained value."""
        prs = self.t_log_pr.exp()
        i = torch.argmin(torch.abs(torch.log(pr) - torch.log(prs).unsqueeze(0)), dim=1, keepdim=True)
        if not torch.allclose(prs[i.squeeze(1)], pr.squeeze(1), rtol=1e-3):
            raise ValueError(f"temperature is only available at the trained Pr values {self.case_pr}")
        return 2 * i + (bc > 0.5).long()

    def temperature(self, x, y, re, pr, bc):
        """bc: 0 = iso-temperature (T=0 at rod, built in), 1 = iso-flux."""
        feats = torch.cat([self.ff_t(self._xy(x, y)), self._wall_features(x, y), self.re_norm(re)], dim=-1)
        theta = torch.gather(self.thermal_net(feats), 1, self.case_index(pr, bc))
        f = self.wall_factor(x, y)
        theta = torch.where(bc > 0.5, theta, f * theta)
        return theta * self.t_ref(re, pr, bc)
