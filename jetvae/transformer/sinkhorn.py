"""Differentiable optimal-transport losses for particle jets.

The target measure contains only real constituents (padding has zero weight).
The prediction may either use every decoder slot, which makes the objective
agnostic to constituent multiplicity, or reuse the target mask for conditional
reconstruction experiments.
"""

from __future__ import annotations

import torch


def four_momentum_to_pt_eta_phi(
    four_momenta: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[..., px, py, pz, E]`` to pT weights and ``[..., eta, phi]``.

    Energy is intentionally not used by this conversion.  The model therefore
    pairs this shape loss with a jet-level four-momentum auxiliary loss.
    """
    if four_momenta.shape[-1] != 4:
        raise ValueError("four_momenta must end with [px, py, pz, E]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    px, py, pz, _ = four_momenta.unbind(dim=-1)
    pt = torch.sqrt(px.square() + py.square() + eps * eps)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / pt.clamp_min(eps))
    coordinates = torch.stack((eta, phi), dim=-1)
    return pt, coordinates


def pairwise_angular_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 1.0,
    angular_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the periodic eta-phi ground-cost matrix for two batched sets."""
    if x.ndim != 3 or y.ndim != 3 or x.shape[-1] != 2 or y.shape[-1] != 2:
        raise ValueError("x and y must have shapes [batch, points, 2]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size")
    if p <= 0.0:
        raise ValueError("p must be positive")
    if angular_scale <= 0.0:
        raise ValueError("angular_scale must be positive")

    delta_eta = x[:, :, None, 0] - y[:, None, :, 0]
    raw_delta_phi = x[:, :, None, 1] - y[:, None, :, 1]
    delta_phi = torch.atan2(torch.sin(raw_delta_phi), torch.cos(raw_delta_phi))
    delta_r = torch.sqrt(delta_eta.square() + delta_phi.square() + eps * eps)
    scaled = delta_r / float(angular_scale)
    return scaled if p == 1.0 else scaled.pow(p)


def normalize_nonnegative_weights(
    weights: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply an optional support mask and normalize every measure to unit mass."""
    if weights.ndim != 2:
        raise ValueError("weights must have shape [batch, points]")
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError("mask must have the same shape as weights")
        weights = weights * mask.to(weights.dtype)
    weights = weights.clamp_min(0.0)
    totals = weights.sum(dim=1, keepdim=True)
    if torch.any(totals <= eps):
        raise ValueError("each jet must have positive total transport weight")
    return weights / totals


def _regularized_ot_cost(
    a: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    n_iters: int,
    p: float,
    angular_scale: float,
) -> torch.Tensor:
    """Compute entropy-regularized balanced OT in the log domain.

    The returned tensor contains one cost per batch element.  Zero weights are
    supported, which is how padded target constituents are excluded.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    cost = pairwise_angular_cost(x, y, p=p, angular_scale=angular_scale)
    neg_inf_a = torch.full_like(a, -torch.inf)
    neg_inf_b = torch.full_like(b, -torch.inf)
    log_a = torch.where(a > 0.0, torch.log(a), neg_inf_a)
    log_b = torch.where(b > 0.0, torch.log(b), neg_inf_b)
    log_kernel = -cost / float(epsilon)

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(int(n_iters)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(
            log_kernel.transpose(1, 2) + log_u[:, None, :], dim=2
        )

    log_plan = log_u[:, :, None] + log_kernel + log_v[:, None, :]
    plan = torch.exp(log_plan)

    transport_cost = (plan * cost).sum(dim=(1, 2))

    # KL(plan || a tensor-product b) is the entropy regularizer associated
    # with the scaling kernel above.  Entries outside either support have
    # exactly zero transport and are excluded from the logarithm.
    log_reference = log_a[:, :, None] + log_b[:, None, :]
    finite_plan = plan > 0.0
    log_ratio = torch.where(
        finite_plan,
        log_plan - log_reference,
        torch.zeros_like(log_plan),
    )
    kl = torch.where(finite_plan, plan * log_ratio, torch.zeros_like(plan)).sum(
        dim=(1, 2)
    )
    return transport_cost + float(epsilon) * kl


def jet_axis(
    four_momenta: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(eta_jet, phi_jet)``, the direction of the summed four-momentum."""
    if mask is not None:
        four_momenta = four_momenta * mask.unsqueeze(-1).to(four_momenta.dtype)
    total = four_momenta.sum(dim=1)
    px, py, pz = total[..., 0], total[..., 1], total[..., 2]
    pt = torch.sqrt(px.square() + py.square() + eps * eps)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / pt.clamp_min(eps))
    return eta, phi


def recenter_on_axis(
    coordinates: torch.Tensor,
    eta_axis: torch.Tensor,
    phi_axis: torch.Tensor,
) -> torch.Tensor:
    """Shift ``[..., (eta, phi)]`` so ``(eta_axis, phi_axis)`` becomes the origin."""
    eta, phi = coordinates.unbind(dim=-1)
    shifted_eta = eta - eta_axis.unsqueeze(-1)
    raw_dphi = phi - phi_axis.unsqueeze(-1)
    shifted_phi = torch.atan2(torch.sin(raw_dphi), torch.cos(raw_dphi))
    return torch.stack((shifted_eta, shifted_phi), dim=-1)


def pt_weighted_angular_spread(
    four_momenta: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the pT-weighted RMS opening angle around the jet's own axis.

    This is a per-event scalar measuring how spread out (vs. collimated) the
    constituents are, independent of the jet's absolute lab-frame direction.
    """
    pt, coordinates = four_momentum_to_pt_eta_phi(four_momenta, eps=eps)
    eta_axis, phi_axis = jet_axis(four_momenta, mask=mask, eps=eps)
    centered = recenter_on_axis(coordinates, eta_axis, phi_axis)
    delta_eta, delta_phi = centered.unbind(dim=-1)
    radius_sq = delta_eta.square() + delta_phi.square()

    weights = pt * mask.to(pt.dtype) if mask is not None else pt
    weight_total = weights.sum(dim=1).clamp_min(eps)
    weighted_radius_sq = (weights * radius_sq).sum(dim=1) / weight_total
    return torch.sqrt(weighted_radius_sq.clamp_min(eps * eps))


def spread_relative_loss(
    true_four_momenta: torch.Tensor,
    pred_four_momenta: torch.Tensor,
    true_mask: torch.Tensor,
    prediction_support: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize a mismatch in pT-weighted angular spread (opening angle).

    Unlike the EMD term, this compares a single scalar per event rather than
    constituent-by-constituent positions, so it directly supervises "is the
    predicted jet as spread out as the real one" -- the specific failure mode
    of decoder output collapsing onto a tight cluster near the jet axis.
    """
    pred_mask = true_mask if prediction_support == "target_mask" else None
    true_spread = pt_weighted_angular_spread(true_four_momenta, mask=true_mask, eps=eps)
    pred_spread = pt_weighted_angular_spread(pred_four_momenta, mask=pred_mask, eps=eps)
    relative_error = (pred_spread - true_spread) / true_spread.clamp_min(eps)
    return relative_error.square().mean()


def normalize_against_reference(
    weights: torch.Tensor,
    reference_total: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Divide ``weights`` by someone else's total mass instead of its own.

    Unlike :func:`normalize_nonnegative_weights`, the result is not forced to
    sum to 1 — if ``weights`` carries more or less total mass than
    ``reference_total``, that imbalance survives and is visible to whatever
    consumes the output. This is what makes the resulting measure scale-aware:
    a uniform rescaling of ``weights`` is no longer invisible.
    """
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError("mask must have the same shape as weights")
        weights = weights * mask.to(weights.dtype)
    weights = weights.clamp_min(0.0)
    return weights / reference_total.clamp_min(eps)


def sinkhorn_emd_loss(
    true_four_momenta: torch.Tensor,
    pred_four_momenta: torch.Tensor,
    true_mask: torch.Tensor,
    *,
    prediction_support: str = "all",
    epsilon: float = 0.05,
    n_iters: int = 50,
    p: float = 1.0,
    angular_scale: float = 1.0,
    debias: bool = True,
    scale_aware: bool = False,
    center: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """pT-weighted Sinkhorn approximation to jet EMD in eta-phi space.

    ``prediction_support='all'`` is the count-agnostic mode: all decoder slots
    participate and the target mask is used only to remove target padding.
    ``prediction_support='target_mask'`` conditions both sides on the known
    target multiplicity and is retained only as an ablation option.

    ``scale_aware=False`` (default) normalizes the target and prediction pT
    measures independently, each to its own unit total mass. This makes the
    resulting cost exactly invariant to any uniform rescaling of the
    prediction's momenta -- the loss cannot see or penalize an overall energy
    scale mismatch. ``scale_aware=True`` instead divides *both* measures by
    the target's total pT, so the prediction measure only sums to 1 when its
    total pT matches the target's; a scale mismatch then shows up as a mass
    imbalance the (still-balanced) Sinkhorn iteration has to reconcile. This
    is an approximation, not a principled unbalanced-OT formulation -- treat
    it as experimental and watch for numerical instability at large mismatch.

    ``center=True`` shifts each side's (eta, phi) coordinates by its own jet
    axis (the direction of its own summed four-momentum) before computing
    cost, independently for target and prediction. This removes each jet's
    absolute lab-frame direction -- a nuisance the ground cost otherwise pays
    for -- so the comparison is purely about constituent-level shape relative
    to each jet's own axis.
    """
    if true_four_momenta.shape != pred_four_momenta.shape:
        raise ValueError("true and predicted four-momenta must have equal shapes")
    if true_mask.shape != true_four_momenta.shape[:2]:
        raise ValueError("true_mask must have shape [batch, max_particles]")
    if prediction_support not in ("all", "target_mask"):
        raise ValueError("prediction_support must be 'all' or 'target_mask'")

    # Sinkhorn is substantially more stable in float32 than float16/bfloat16.
    working_dtype = (
        torch.float32
        if true_four_momenta.dtype in (torch.float16, torch.bfloat16)
        else true_four_momenta.dtype
    )
    true_four_momenta = true_four_momenta.to(working_dtype)
    pred_four_momenta = pred_four_momenta.to(working_dtype)

    true_pt, true_coordinates = four_momentum_to_pt_eta_phi(true_four_momenta, eps=eps)
    pred_pt, pred_coordinates = four_momentum_to_pt_eta_phi(pred_four_momenta, eps=eps)
    pred_mask = true_mask if prediction_support == "target_mask" else None

    if center:
        true_eta_axis, true_phi_axis = jet_axis(true_four_momenta, mask=true_mask, eps=eps)
        true_coordinates = recenter_on_axis(true_coordinates, true_eta_axis, true_phi_axis)
        pred_eta_axis, pred_phi_axis = jet_axis(pred_four_momenta, mask=pred_mask, eps=eps)
        pred_coordinates = recenter_on_axis(pred_coordinates, pred_eta_axis, pred_phi_axis)

    a = normalize_nonnegative_weights(true_pt, mask=true_mask, eps=eps)
    if scale_aware:
        true_total = (true_pt * true_mask.to(true_pt.dtype)).clamp_min(0.0).sum(
            dim=1, keepdim=True
        )
        b = normalize_against_reference(pred_pt, true_total, mask=pred_mask, eps=eps)
    else:
        b = normalize_nonnegative_weights(pred_pt, mask=pred_mask, eps=eps)

    cross = _regularized_ot_cost(
        a,
        true_coordinates,
        b,
        pred_coordinates,
        epsilon,
        n_iters,
        p,
        angular_scale,
    )
    if not debias:
        return cross.mean()

    true_self = _regularized_ot_cost(
        a,
        true_coordinates,
        a,
        true_coordinates,
        epsilon,
        n_iters,
        p,
        angular_scale,
    )
    pred_self = _regularized_ot_cost(
        b,
        pred_coordinates,
        b,
        pred_coordinates,
        epsilon,
        n_iters,
        p,
        angular_scale,
    )
    divergence = cross - 0.5 * true_self - 0.5 * pred_self
    return divergence.clamp_min(0.0).mean()
