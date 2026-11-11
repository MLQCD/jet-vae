"""Event -> shape-only ``(z, eta, phi)`` point set, plus pid categories.

Every particle becomes ``(z, eta, phi)``:

- ``z``   = pT / sum(pT over the jet's kept constituents), so z sums to 1.
- ``eta`` = asinh(pz / pT), shifted so the jet's own massless axis is at 0.
- ``phi`` = atan2(py, px), shifted likewise and wrapped to (-pi, pi].

The axis shift assumes massless constituents: pseudorapidity then equals true
rapidity, which shifts by an exact constant under a longitudinal boost, so the
recentring is an exact boost + rotation. ``E`` in the file is never used.
Constituents are sorted by pT (descending, stable) and truncated to
``max_particles`` *before* z and the axis are computed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PID_CATEGORIES",
    "CATEGORY_NAMES",
    "pid_to_category",
    "event_to_z_eta_phi",
]

# 0 photon | 1 neutral hadron | 2 charged hadron | 3 lepton.
# Only used for analysis (the no-pid model never sees it); stored in the cache.
PID_CATEGORIES: dict[int, int] = {
    22: 0,
    2112: 1, -2112: 1, 130: 1,
    211: 2, -211: 2, 321: 2, -321: 2, 2212: 2, -2212: 2,
    11: 3, -11: 3, 13: 3, -13: 3,
}
CATEGORY_NAMES = ["photon", "neutral_hadron", "charged_hadron", "lepton"]


def pid_to_category(pid: np.ndarray) -> np.ndarray:
    """Map PDG ids to category indices; raises on an unknown id."""
    pid = np.asarray(pid)
    category = np.full(pid.shape, -1, dtype=np.int64)
    for raw_pid, cat in PID_CATEGORIES.items():
        category[pid == raw_pid] = cat
    unknown = pid[category < 0]
    if unknown.size:
        raise ValueError(f"Unrecognized pid value(s): {sorted(set(unknown.tolist()))}")
    return category


def event_to_z_eta_phi(
    event: np.ndarray, max_particles: int | None = None, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """One event (rows ``pid px py pz E``) -> ``(shape [n,3], category [n])``."""
    if event.ndim != 2 or event.shape[1] != 5:
        raise ValueError("each event must have shape [num_particles, 5]")

    pid = event[:, 0]
    px, py, pz = event[:, 1], event[:, 2], event[:, 3]
    pt = np.sqrt(px * px + py * py + eps * eps)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(pz / np.maximum(pt, eps))

    order = np.argsort(-pt, kind="stable")
    pid, px, py, pz, pt, phi, eta = (a[order] for a in (pid, px, py, pz, pt, phi, eta))
    if max_particles is not None:
        pid, px, py, pz, pt, phi, eta = (
            a[:max_particles] for a in (pid, px, py, pz, pt, phi, eta)
        )

    px_jet, py_jet, pz_jet = px.sum(), py.sum(), pz.sum()
    pt_jet = np.sqrt(px_jet * px_jet + py_jet * py_jet + eps * eps)
    eta_jet = np.arcsinh(pz_jet / max(pt_jet, eps))
    phi_jet = np.arctan2(py_jet, px_jet)

    eta_shifted = eta - eta_jet
    raw_dphi = phi - phi_jet
    phi_shifted = np.arctan2(np.sin(raw_dphi), np.cos(raw_dphi))

    z = pt / max(pt.sum(), eps)
    shape = np.stack([z, eta_shifted, phi_shifted], axis=-1).astype(np.float32)
    return shape, pid_to_category(pid.astype(np.int64))
