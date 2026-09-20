"""Physics-informed model for the rod-bundle unit cell.

Two sub-networks, mirroring the paper's own physics (Section 4.3: momentum
is identical across all four Prandtl numbers and both wall BCs; only the
passive-scalar temperature equation changes):

  MomentumNet(x, y, Re)         -> u, v, p, nu_t
  ThermalNet(x, y, Re, Pr, bc)  -> theta   (uses MomentumNet's u, v as input)

Inputs are normalized before the Fourier feature embedding; Re is log-scaled
since the calibration sweep spans 1531-49000 (Table 2).
"""
import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_features, scale, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_features, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def out_dim(self):
        return self.B.shape[1] * 2


def _mlp(in_dim, hidden, out_dim):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class MomentumNet(nn.Module):
    def __init__(self, cfg, re_min, re_max):
        super().__init__()
        self.re_min, self.re_max = re_min, re_max
        self.fourier = FourierFeatures(2, cfg["fourier_features"], cfg["fourier_scale"], seed=0)
        # outputs: u, v, p, nut_raw, k_raw
        self.net = _mlp(self.fourier.out_dim + 1, cfg["momentum_hidden"], 5)
        self.nut_scale = cfg["nut_scale"]

    def _re_norm(self, re):
        log_re = torch.log(re)
        log_min, log_max = np.log(self.re_min), np.log(self.re_max)
        return 2 * (log_re - log_min) / (log_max - log_min) - 1

    def forward(self, xy, re):
        feats = self.fourier(xy)
        re_n = self._re_norm(re).unsqueeze(-1)
        out = self.net(torch.cat([feats, re_n], dim=-1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        nut_raw, k_raw = out[:, 3:4], out[:, 4:5]
        nut = torch.nn.functional.softplus(nut_raw) * self.nut_scale
        # k is data-fit only (Fig. 13 profiles) - the Boussinesq closure here
        # uses nut directly and never reads k back, so it carries no physics
        # residual of its own; it's along for the ride as an extra observable.
        k = torch.nn.functional.softplus(k_raw)
        return u, v, p, nut, k


class ThermalNet(nn.Module):
    def __init__(self, cfg, re_min, re_max, pr_values):
        super().__init__()
        self.re_min, self.re_max = re_min, re_max
        self.pr_min, self.pr_max = min(pr_values), max(pr_values)
        self.fourier = FourierFeatures(2, cfg["fourier_features"], cfg["fourier_scale"], seed=1)
        # + Re, Pr, bc, u, v (momentum coupling)
        self.net = _mlp(self.fourier.out_dim + 5, cfg["thermal_hidden"], 1)

    def _re_norm(self, re):
        log_re = torch.log(re)
        log_min, log_max = np.log(self.re_min), np.log(self.re_max)
        return 2 * (log_re - log_min) / (log_max - log_min) - 1

    def _pr_norm(self, pr):
        log_pr = torch.log(pr)
        log_min, log_max = np.log(self.pr_min), np.log(self.pr_max)
        return 2 * (log_pr - log_min) / (log_max - log_min) - 1

    def forward(self, xy, re, pr, bc, u, v):
        feats = self.fourier(xy)
        re_n = self._re_norm(re).unsqueeze(-1)
        pr_n = self._pr_norm(pr).unsqueeze(-1)
        bc_n = bc.unsqueeze(-1).float()
        inp = torch.cat([feats, re_n, pr_n, bc_n, u, v], dim=-1)
        return self.net(inp)


class RodBundlePINN(nn.Module):
    def __init__(self, cfg, re_min, re_max, pr_values):
        super().__init__()
        self.momentum = MomentumNet(cfg["model"], re_min, re_max)
        self.thermal = ThermalNet(cfg["model"], re_min, re_max, pr_values)

    def forward(self, x, y, re, pr, bc):
        xy = torch.cat([x, y], dim=-1)
        u, v, p, nut, k = self.momentum(xy, re)
        # detach: temperature is a passive scalar (paper Section 4.3) - it is
        # advected by the frozen momentum solution, exactly like running
        # scalarTransportFoam on a frozen U field after simpleFoam converges
        # (cfd/openfoam/generate_scalar_cases.py). One-way coupling only.
        theta = self.thermal(xy, re, pr, bc, u.detach(), v.detach())
        return {"u": u, "v": v, "p": p, "nut": nut, "k": k, "theta": theta}
