"""Shape-only Sinkhorn-EMD over (z, eta, phi) point clouds.

This uses the log-domain balanced-OT solver and ground cost in ``sinkhorn.py``.
Coordinates and nonnegative weights are consumed directly, instead of being
derived from a four-momentum with an energy component that this model does
not use.
"""

from __future__ import annotations

import torch

from .sinkhorn import (
    _regularized_ot_cost,
    normalize_nonnegative_weights,
)

__all__ = [
    "jet_axis_from_shape",
    "recenter",
    "shape_sinkhorn_emd_loss",
]


def jet_axis_from_shape(
    z: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direction of the massless vector sum, treating ``z`` as each particle's
    transverse momentum. Same idea as the four-momentum jet axis, but
    starts from ``(z, eta, phi)`` instead of a four-momentum with energy.
    """
    if mask is not None:
        z = z * mask.to(z.dtype)
    px = (z * torch.cos(phi)).sum(dim=1)
    py = (z * torch.sin(phi)).sum(dim=1)
    pz = (z * torch.sinh(eta)).sum(dim=1)
    pt = torch.sqrt(px.square() + py.square() + eps * eps)
    eta_axis = torch.asinh(pz / pt.clamp_min(eps))
    phi_axis = torch.atan2(py, px)
    return eta_axis, phi_axis


def recenter(
    eta: torch.Tensor,
    phi: torch.Tensor,
    eta_axis: torch.Tensor,
    phi_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift ``(eta, phi)`` so ``(eta_axis, phi_axis)`` becomes the origin."""
    shifted_eta = eta - eta_axis.unsqueeze(-1)
    raw_dphi = phi - phi_axis.unsqueeze(-1)
    shifted_phi = torch.atan2(torch.sin(raw_dphi), torch.cos(raw_dphi))
    return shifted_eta, shifted_phi


def shape_sinkhorn_emd_loss(
    true_z: torch.Tensor,
    true_eta: torch.Tensor,
    true_phi: torch.Tensor,
    true_mask: torch.Tensor,
    pred_z: torch.Tensor,
    pred_eta: torch.Tensor,
    pred_phi: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 50,
    p: float = 1.0,
    angular_scale: float = 0.4,
    debias: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence between two pT-fraction-weighted jets.

    ``angular_scale`` is the jet radius ``R`` the ground cost is scaled by,
    ``C_ij = deltaR_ij / R`` -- pass the clustering radius (0.4 for these
    jets), not 1.
    """
    if true_z.shape != pred_z.shape:
        raise ValueError("true and predicted shapes must match")
    if true_mask.shape != true_z.shape:
        raise ValueError("true_mask must have shape [batch, max_particles]")

    working_dtype = (
        torch.float32
        if true_z.dtype in (torch.float16, torch.bfloat16)
        else true_z.dtype
    )
    true_eta = true_eta.to(working_dtype)
    true_phi = true_phi.to(working_dtype)
    pred_eta = pred_eta.to(working_dtype)
    pred_phi = pred_phi.to(working_dtype)

    a = normalize_nonnegative_weights(true_z.to(working_dtype), mask=true_mask, eps=eps)
    # pred_z already sums to 1 over all K slots by construction (softmax in
    # decode()); renormalizing here is a defensive no-op, not a correction.
    b = normalize_nonnegative_weights(pred_z.to(working_dtype), mask=None, eps=eps)

    true_coordinates = torch.stack((true_eta, true_phi), dim=-1)
    pred_coordinates = torch.stack((pred_eta, pred_phi), dim=-1)

    cross = _regularized_ot_cost(
        a, true_coordinates, b, pred_coordinates, epsilon, n_iters, p, angular_scale
    )
    if not debias:
        return cross.mean()

    true_self = _regularized_ot_cost(
        a, true_coordinates, a, true_coordinates, epsilon, n_iters, p, angular_scale
    )
    pred_self = _regularized_ot_cost(
        b, pred_coordinates, b, pred_coordinates, epsilon, n_iters, p, angular_scale
    )
    divergence = cross - 0.5 * true_self - 0.5 * pred_self
    return divergence.clamp_min(0.0).mean()
