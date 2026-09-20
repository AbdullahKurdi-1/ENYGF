"""RANS-with-eddy-viscosity residuals, computed by autograd at collocation
points. Boussinesq closure: the network's own `nut` output stands in for the
turbulence model (no k-omega transport equations inside the PINN - nut is
just a smooth field the network is free to fit, regularized only by the
momentum residual and by matching the paper's Re-dependent flow behaviour
through the data loss). This is a deliberate simplification: full closure
transport equations would double the residual terms for a training-time
benefit that a Boussinesq eddy-viscosity term already captures reasonably
well for this attached, non-separating flow.
"""
import torch


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True,
    )[0]


def momentum_residuals(momentum_net, x, y, re, nu):
    """x, y must be leaf tensors with requires_grad=True."""
    xy = torch.cat([x, y], dim=-1)
    u, v, p, nut, k = momentum_net(xy, re)

    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)

    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)

    nu_eff = nu + nut  # cross-gradient term in d/dx[(nu+nut) du/dx] dropped, see module docstring
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu_eff * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu_eff * (v_xx + v_yy)
    return continuity, momentum_x, momentum_y, (u, v, p, nut, k)


def energy_residual(model, x, y, re, pr, bc, turbulent_prandtl=0.9):
    """x, y must be leaf tensors with requires_grad=True."""
    xy = torch.cat([x, y], dim=-1)
    u, v, p, nut, k = model.momentum(xy, re)
    theta = model.thermal(xy, re, pr, bc, u.detach(), v.detach())

    theta_x, theta_y = grad(theta, x), grad(theta, y)
    theta_xx, theta_yy = grad(theta_x, x), grad(theta_y, y)

    nu = 1.5e-05  # air, matches cfd/openfoam/unit_cell/constant/transportProperties
    alpha_eff = nu / pr + nut.detach() / turbulent_prandtl
    residual = u.detach() * theta_x + v.detach() * theta_y - alpha_eff * (theta_xx + theta_yy)
    return residual, theta


def wall_bc_loss(model, x, y, re):
    """No-slip at the rod surface for momentum; wall value/gradient for
    theta is enforced through the data loss (digitized_data / CFD samples
    already sit on the wall at t=0), not duplicated here.
    """
    xy = torch.cat([x, y], dim=-1)
    u, v, p, nut, k = model.momentum(xy, re)
    return (u**2 + v**2).mean()


def symmetry_bc_loss(model, x, y, re, normal_is_x: bool):
    """symmetryPlane: zero velocity normal to the plane, and zero
    normal-gradient of the tangential velocity component."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    xy = torch.cat([x, y], dim=-1)
    u, v, p, nut, k = model.momentum(xy, re)
    if normal_is_x:
        normal_velocity = u
        tangential_normal_grad = grad(v, x)
    else:
        normal_velocity = v
        tangential_normal_grad = grad(u, y)
    return (normal_velocity**2).mean() + (tangential_normal_grad**2).mean()
