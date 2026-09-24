"""Governing equations for fully developed, streamwise-periodic flow in the
quarter unit cell, with the same thermal set-up as the 2023 DNS.

Axial momentum (RANS, Boussinesq eddy viscosity, secondary flow neglected -
consistent with the k-omega SST CFD, and the DNS reports it as weak):
    0 = G + d/dx[(nu + nu_t) dw/dx] + d/dy[(nu + nu_t) dw/dy]
with G the driving pressure gradient, set so the area-averaged w equals U_b.

Energy (passive scalar, rho*cp = 1, periodic so no net axial advection):
    0 = d/dx[alpha dT/dx] + d/dy[alpha dT/dy] - S,   alpha = nu/Pr + nu_t/Pr_t
    iso-temperature: T = 0 at the rod,          S = 1
    iso-flux:        (nu/Pr) dT/dn = q at rod,  S = q * (wetted perimeter / area) = 4q/Dh
Iso-flux temperature is only defined up to a constant; the gauge T = 0 at the
rod in the narrow gap (x = r, y = 0) matches cfd/openfoam/extract_profiles.py.

Symmetry planes x = 0, y = 0, x = a, y = a: zero normal gradient of w and T.
"""
import math

import torch


def grad(out, inp):
    g = torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inp) if g is None else g


# ---- point sampling ---------------------------------------------------------
def sample_interior(model, n, device, wall_biased=False):
    """Uniform samples (area-weighted, usable for averages) or wall-biased ones."""
    r, a = model.r, model.a
    if wall_biased:
        phi = torch.rand(n, 1, device=device) * (math.pi / 2)
        rho_max = a / torch.maximum(torch.cos(phi), torch.sin(phi))
        rho = r + (rho_max - r) * torch.rand(n, 1, device=device) ** 2
        return rho * torch.cos(phi), rho * torch.sin(phi)
    pts = []
    while sum(p.shape[0] for p in pts) < n:
        xy = torch.rand(2 * n, 2, device=device) * a
        pts.append(xy[(xy**2).sum(1) > r**2])
    xy = torch.cat(pts)[:n]
    return xy[:, 0:1], xy[:, 1:2]


def sample_wall(model, n, device, max_angle=math.pi / 2):
    phi = torch.rand(n, 1, device=device) * max_angle
    return model.r * torch.cos(phi), model.r * torch.sin(phi), phi


def sample_symmetry(model, n, device):
    """Returns list of (x, y, normal_is_x) for the four symmetry planes."""
    r, a = model.r, model.a
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, device=device)
    full = lambda v: torch.full((n, 1), v, device=device)
    return [
        (full(0.0), u(r, a), True),   # x = 0
        (u(r, a), full(0.0), False),  # y = 0
        (full(a), u(0.0, a), True),   # x = a
        (u(0.0, a), full(a), False),  # y = a
    ]


def conditions(values, n, device):
    idx = torch.randint(0, len(values), (n, 1), device=device)
    return torch.tensor(values, dtype=torch.float32, device=device)[idx.squeeze(1)].unsqueeze(1)


# ---- residuals --------------------------------------------------------------
def momentum_residual(model, x, y, re):
    x, y = x.requires_grad_(True), y.requires_grad_(True)
    w, nut, _ = model.momentum(x, y, re)
    mu = model.nu(re) + nut
    div = grad(mu * grad(w, x), x) + grad(mu * grad(w, y), y)
    return (model.pressure_gradient(re) + div) / model.g_scale


def energy_residual(model, x, y, re, pr, bc):
    x, y = x.requires_grad_(True), y.requires_grad_(True)
    T = model.temperature(x, y, re, pr, bc)
    # alpha and its spatial gradient come from the flow network but are
    # detached: temperature is passive, so the energy loss must not train the
    # flow network - yet d(alpha)/dx is still needed in the conservative form.
    _, nut, _ = model.momentum(x, y, re)
    alpha = (model.nu(re) / pr + nut / model.pr_t).detach()
    alpha_x = (grad(nut, x) / model.pr_t).detach()
    alpha_y = (grad(nut, y) / model.pr_t).detach()
    T_x, T_y = grad(T, x), grad(T, y)
    div = alpha * (grad(T_x, x) + grad(T_y, y)) + alpha_x * T_x + alpha_y * T_y
    S = model.sink(bc)
    return (div - S) / S


def bulk_velocity_loss(model, x_uniform, y_uniform, re_value):
    re = torch.full_like(x_uniform, re_value)
    w, _, _ = model.momentum(x_uniform, y_uniform, re)
    return ((w.mean() - model.u_bulk) / model.u_bulk) ** 2


def symmetry_loss(model, planes, re, pr, bc, include_thermal):
    total = 0.0
    for x, y, normal_is_x in planes:
        x, y = x.requires_grad_(True), y.requires_grad_(True)
        coord = x if normal_is_x else y
        w, _, _ = model.momentum(x, y, re)
        total = total + ((grad(w, coord) * model.a / model.u_bulk) ** 2).mean()
        if include_thermal:
            T = model.temperature(x, y, re, pr, bc)
            t_ref = model.t_ref(re, pr, bc)
            total = total + ((grad(T, coord) * model.a / t_ref) ** 2).mean()
    return total / len(planes)


def wall_heat_flux_loss(model, n, device, re_values, pr_values):
    """Iso-flux cases: (nu/Pr) * dT/dn_out = q, with n_out pointing into the rod."""
    x, y, phi = sample_wall(model, n, device)
    x, y = x.requires_grad_(True), y.requires_grad_(True)
    re = conditions(re_values, n, device)
    pr = conditions(pr_values, n, device)
    bc = torch.ones_like(re)
    T = model.temperature(x, y, re, pr, bc)
    dT_drho = grad(T, x) * torch.cos(phi) + grad(T, y) * torch.sin(phi)
    flux_into_fluid = -(model.nu(re) / pr) * dT_drho
    return (((flux_into_fluid - model.q_wall) / model.q_wall) ** 2).mean()


def wall_shear_balance_loss(model, n, device, re_values):
    """Integral momentum balance of the whole cell: G * A = integral of the wall
    shear over the rod arc (the symmetry planes carry no shear). Ties the
    learned pressure gradient to the network's own wall shear, which the
    pointwise residual alone does not pin down well."""
    x, y, phi = sample_wall(model, n, device)
    x, y = x.requires_grad_(True), y.requires_grad_(True)
    re = conditions(re_values, n, device)
    w, _, _ = model.momentum(x, y, re)
    tau = model.nu(re) * (grad(w, x) * torch.cos(phi) + grad(w, y) * torch.sin(phi))
    area = model.a**2 - math.pi * model.r**2 / 4
    perimeter = math.pi * model.r / 2
    G = model.pressure_gradient(re)
    return (((G - tau * perimeter / area) / model.g_scale) ** 2).mean()


def gauge_loss(model, device, re_values, pr_values):
    n = len(re_values) * len(pr_values)
    re = torch.tensor([r for r in re_values for _ in pr_values], dtype=torch.float32, device=device).unsqueeze(1)
    pr = torch.tensor([p for _ in re_values for p in pr_values], dtype=torch.float32, device=device).unsqueeze(1)
    bc = torch.ones(n, 1, device=device)
    x = torch.full((n, 1), model.r, device=device)
    y = torch.zeros(n, 1, device=device)
    T = model.temperature(x, y, re, pr, bc)
    return ((T / model.t_ref(re, pr, bc)) ** 2).mean()


# ---- derived quantities ---------------------------------------------------------
def wall_shear_at(model, x, y, re):
    """tau_w/rho = nu * dw/drho at rod points (x, y). Keeps the graph (used in training)."""
    x, y = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    w, _, _ = model.momentum(x, y, re)
    rho = torch.sqrt(x**2 + y**2)
    return model.nu(re) * (grad(w, x) * x + grad(w, y) * y) / rho


def wall_temperature_at(model, x, y, re, pr, bc):
    """(T_wall, heat flux into the fluid) at rod points (x, y). Keeps the graph."""
    x, y = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    T = model.temperature(x, y, re, pr, bc)
    rho = torch.sqrt(x**2 + y**2)
    dT_drho = (grad(T, x) * x + grad(T, y) * y) / rho
    return T, -(model.nu(re) / pr) * dT_drho


def wall_shear(model, re_value, angles_rad, device):
    """tau_w/rho = nu * dw/drho at the rod, at the given angles from the gap."""
    phi = torch.as_tensor(angles_rad, dtype=torch.float32, device=device).reshape(-1, 1)
    x = (model.r * torch.cos(phi)).requires_grad_(True)
    y = (model.r * torch.sin(phi)).requires_grad_(True)
    re = torch.full_like(x, re_value)
    w, _, _ = model.momentum(x, y, re)
    dw_drho = grad(w, x) * torch.cos(phi) + grad(w, y) * torch.sin(phi)
    return (model.nu(re) * dw_drho).detach()


def wall_temperature_data(model, re_value, pr, bc, angles_rad, device):
    """Returns (T_wall, heat flux into the fluid) at the rod."""
    phi = torch.as_tensor(angles_rad, dtype=torch.float32, device=device).reshape(-1, 1)
    x = (model.r * torch.cos(phi)).requires_grad_(True)
    y = (model.r * torch.sin(phi)).requires_grad_(True)
    re = torch.full_like(x, re_value)
    prt = torch.full_like(x, pr)
    bct = torch.full_like(x, bc)
    T = model.temperature(x, y, re, prt, bct)
    dT_drho = grad(T, x) * torch.cos(phi) + grad(T, y) * torch.sin(phi)
    flux = -(model.nu(re) / prt) * dT_drho
    return T.detach(), flux.detach()


@torch.no_grad()
def bulk_temperature(model, re_value, pr, bc, n, device):
    x, y = sample_interior(model, n, device)
    re = torch.full_like(x, re_value)
    w, _, _ = model.momentum(x, y, re)
    T = model.temperature(x, y, re, torch.full_like(x, pr), torch.full_like(x, bc))
    return float((w * T).sum() / w.sum())


def nusselt(model, re_value, pr, bc, device, n_bulk=50000, n_wall=400):
    """Nu = phi_m * Dh / (lambda * (T_w,m - T_b)), as in the 2023 DNS (their Eq. 4),
    with its Dh = 0.0712 m. Wall averages over 0-45 deg, which by symmetry
    equals the DNS's -45..45."""
    angles = torch.linspace(0, math.pi / 4, n_wall)
    Tw, flux = wall_temperature_data(model, re_value, pr, bc, angles, device)
    Tb = bulk_temperature(model, re_value, pr, bc, n_bulk, device)
    lam = model.u_bulk * model.dh_ref / re_value / pr
    return float(flux.mean() * model.dh_ref / (lam * (Tw.mean() - Tb)))
