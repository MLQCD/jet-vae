"""Shared numerical definitions for current appendix figures (9--11)."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINT_SHA256 = "e8e59c4f836b2c1df728a3970439ecde1db3f889208556a4b4eaccc7f24a42b2"
XEDGES = np.arange(.5, 81.5, 1)
YEDGES = np.linspace(-1.5, 6, 76)
OBSERVABLE_NAMES = ["multiplicity", "ptD", "entropy", "LHA", "girth", "angularity2",
                    "e2_beta0p5", "e2_beta1", "log_mass2_GeV2"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_encoder(checkpoint, device="auto"):
    """Use the exact frozen no-PID particle model used in the paper."""
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Expected the paper Transformer32 checkpoint (beta=1e-5, epoch 134).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT))
    import torch
    from transformer_vae.encode import load_model
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, _ = load_model(checkpoint, device)
    model.requires_grad_(False)
    return model, device


def encode_p4(model, p4, device, batch=128):
    """RAMBO (E,px,py,pz) -> the existing no-PID encoder preprocessing.

    PID 22 is only an internal adapter for the repository's category validator;
    categories are discarded. RAMBO has no species and the model receives none.
    Match the original DAT-reader float32 conversion before preprocessing.
    """
    sys.path.insert(0, str(ROOT))
    from transformer_vae.data.pointset import event_to_z_eta_phi
    from transformer_vae.encode import encode
    count, n, _ = p4.shape
    if not 2 <= n <= 80:
        raise ValueError("The appendix RAMBO samples must have 2 <= N <= 80; no truncation.")
    raw = np.empty((count, n, 5), dtype=np.float32)
    raw[:, :, 0] = 22
    raw[:, :, 1:] = p4[:, :, [1, 2, 3, 0]]
    particles = np.zeros((count, 80, 3), dtype=np.float32)
    mask = np.zeros((count, 80), dtype=bool)
    for i, event in enumerate(raw):
        particles[i, :n] = event_to_z_eta_phi(event, max_particles=80)[0]
        mask[i, :n] = True
    return encode(model, particles, mask, batch=batch, device=device)


def project(logvar, basis):
    if "checkpoint_sha256" in basis and basis["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("PCA basis belongs to a different checkpoint.")
    a, b = np.asarray(basis["a"]), np.asarray(basis["b"])
    return (np.asarray(logvar, dtype=float)[:, basis["active_indices"]] - a) @ b / (b @ b)


def summarize_distribution(n, u):
    """Integer counts and unsmoothed per-N statistics used by Fig. 9."""
    n, u = np.asarray(n), np.asarray(u)
    if n.shape != u.shape or not len(n) or not np.isfinite(u).all():
        raise ValueError("Expected nonempty, aligned multiplicity and width arrays.")
    counts = np.histogram2d(n, u, bins=(XEDGES, YEDGES))[0].astype(int)
    if counts.sum() != len(n):
        raise ValueError("Some events fall outside the paper histogram; do not silently drop them.")
    profiles = []
    for value in np.unique(n):
        values = u[n == value]
        if len(values) >= 20:
            profiles.append(dict(N=int(value), count=len(values), mean=float(values.mean()),
                                 sem=float(values.std(ddof=1) / np.sqrt(len(values))),
                                 q16=float(np.quantile(values, .16)), q84=float(np.quantile(values, .84))))
    return dict(n_events=len(n), counts=counts.tolist(), profiles=profiles)


def observables(event, radius=.4):
    """Nine Fig. 11 observables, all positive-pT raw constituents, GeV units.

    No top-80 truncation or angular cut. The axis uses the summed four-momentum.
    Keep the paper calculation's 1e-12 entropy/mass regulators, including N=1.
    """
    p = np.asarray(event, dtype=float)
    p = p[np.hypot(p[:, 1], p[:, 2]) > 0]
    if len(p) == 0:
        raise ValueError("No positive-pT constituents")
    px, py, pz, energy = p[:, 1:].T
    pt = np.hypot(px, py)
    z = pt / pt.sum()
    eta, phi = np.arcsinh(pz / pt), np.arctan2(py, px)
    jx, jy, jz, je = (float(v.sum()) for v in (px, py, pz, energy))
    jet_pt = math.hypot(jx, jy)
    jet_eta = math.asinh(jz / max(jet_pt, np.finfo(float).tiny))
    jet_phi = math.atan2(jy, jx)
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    theta = np.sqrt((eta - jet_eta)**2 + wrap(phi - jet_phi)**2) / radius
    i, j = np.triu_indices(len(z), 1)
    pair = np.sqrt((eta[i] - eta[j])**2 + wrap(phi[i] - phi[j])**2) / radius
    weight = z[i] * z[j]
    mass2 = max(je**2 - jx**2 - jy**2 - jz**2, 0.)
    return np.array([len(z), np.sqrt(np.sum(z**2)), -np.sum(z * np.log(z + 1e-12)),
                     np.sum(z * np.sqrt(theta)), np.sum(z * theta), np.sum(z * theta**2),
                     np.sum(weight * np.sqrt(pair)), np.sum(weight * pair), math.log(mass2 + 1e-12)])


def pearson(mu, logvar, values):
    x = np.column_stack([mu, logvar[:, 0]]).astype(float)
    y = np.asarray(values, dtype=float)
    if x.shape[1] != 33 or y.shape != (len(x), 9):
        raise ValueError("Expected 32 mu coordinates, logvar_0, and 9 observables.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x.std(0) == 0) or np.any(y.std(0) == 0):
        raise ValueError("Pearson correlation requires finite, nonconstant columns.")
    return np.corrcoef(x.T, y.T)[:33, 33:]
