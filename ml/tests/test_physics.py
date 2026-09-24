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


def test_wall_gradients_are_along_the_rod_normal():
    m = AnalyticModel()
    phi = torch.linspace(0.1, 1.4, 7).unsqueeze(1)
    x, y = m.r * torch.cos(phi), m.r * torch.sin(phi)
    re, pr, bc = torch.ones_like(x), torch.full_like(x, PR), torch.ones_like(x)
    # d/drho = cos(phi) d/dx + sin(phi) d/dy
    tau = ph.wall_shear_at(m, x, y, re)
    assert torch.allclose(tau, NU * (2 * x * y * torch.cos(phi) + x**2 * torch.sin(phi)), atol=1e-7)
    _, q = ph.wall_temperature_at(m, x, y, re, pr, bc)
    assert torch.allclose(q, -(NU / PR) * (y**2 * torch.cos(phi) + 2 * x * y * torch.sin(phi)), atol=1e-7)


def test_temperature_scale_and_case_lookup():
    import yaml
    from model import RodBundlePINN
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    model = RodBundlePINN(cfg)
    table = torch.tensor([[1.0, 100.0], [7.0, 900.0], [10.0, 1200.0], [20.0, 2000.0]])
    model.set_temperature_scales(table)
    pr = torch.tensor([[0.025], [1.0], [2.0], [7.0], [7.0]])
    bc = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]])
    assert torch.allclose(model.t_ref(None, pr, bc).squeeze(1), torch.tensor([1.0, 900.0, 10.0, 2000.0, 20.0]))
    assert model.case_index(pr, bc).squeeze(1).tolist() == [0, 3, 4, 7, 6]
    try:
        model.case_index(torch.tensor([[1.5]]), torch.tensor([[0.0]]))
        raise AssertionError("an untrained Pr must be rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    test_momentum_residual_includes_eddy_viscosity_gradient()
    test_energy_residual_includes_eddy_diffusivity_gradient()
    test_wall_gradients_are_along_the_rod_normal()
    test_temperature_scale_and_case_lookup()
    print("physics operator tests passed")
