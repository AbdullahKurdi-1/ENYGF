"""Checks the PDE residual operators against hand-derived expressions for
simple analytic fields, including the spatially varying eddy viscosity terms
(d(nu_t)/dx) that are easy to drop by accident.

Run: cd ml && python3 -m pytest tests/test_physics.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import physics as ph  # noqa: E402

NU, C1, PR, PRT, G, S = 1e-3, 0.5, 2.0, 0.9, 0.3, 4.0


class AnalyticModel:
    """w = x^2 y, nu_t = C1 x, T = x y^2, constant nu, G and sink S."""
    r, a, g_scale, u_bulk, pr_t = 0.07, 0.0775, 0.2, 1.0, PRT

    def nu(self, re):
        return torch.full_like(re, NU)

    def momentum(self, x, y, re):
        return x**2 * y, C1 * x, torch.zeros_like(x)

    def pressure_gradient(self, re):
        return torch.full_like(re, G)

    def temperature(self, x, y, re, pr, bc):
        return x * y**2

    def sink(self, bc):
        return torch.full_like(bc, S)


def points():
    torch.manual_seed(0)
    return torch.rand(50, 1), torch.rand(50, 1)


def test_momentum_residual_includes_eddy_viscosity_gradient():
    m = AnalyticModel()
    x, y = points()
    re = torch.ones_like(x)
    got = ph.momentum_residual(m, x.clone(), y.clone(), re)
    # d/dx[(nu + C1 x) 2xy] + d/dy[(nu + C1 x) x^2] = 4 C1 x y + 2 nu y
    expected = (G + 4 * C1 * x * y + 2 * NU * y) / m.g_scale
    assert torch.allclose(got, expected, atol=1e-5)


def test_energy_residual_includes_eddy_diffusivity_gradient():
    m = AnalyticModel()
    x, y = points()
    re, pr, bc = torch.ones_like(x), torch.full_like(x, PR), torch.ones_like(x)
    got = ph.energy_residual(m, x.clone(), y.clone(), re, pr, bc)
    # alpha = nu/Pr + C1 x/Prt; d/dx[alpha y^2] + d/dy[alpha 2xy] = (C1/Prt) y^2 + 2x alpha
    alpha = NU / PR + C1 * x / PRT
    expected = ((C1 / PRT) * y**2 + 2 * x * alpha - S) / S
    assert torch.allclose(got, expected, atol=1e-5)


if __name__ == "__main__":
    test_momentum_residual_includes_eddy_viscosity_gradient()
    test_energy_residual_includes_eddy_diffusivity_gradient()
    print("physics operator tests passed")
