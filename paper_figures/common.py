"""Shared helpers for the paper-figure scripts: jet loading, model loading, encoding, image conversion.

Jets can come from the raw ``.dat`` files (``pid px py pz E`` per constituent) or from the released ``.npz``
files (arrays ``particles80``, ``mask80``, and ``p4``/``pid``/``mask`` for the CNN images).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cnn_vae" / "code_training"))

from transformer_vae.data.dat_io import load_events_from_dat          # noqa: E402
from transformer_vae.data.preprocess import _arrays                    # noqa: E402
from transformer_vae.encode import encode as encode_transformer        # noqa: E402
from transformer_vae.encode import load_model as load_transformer      # noqa: E402
from betaVae import BetaVAE                                            # noqa: E402
from jet_image_utils import event_to_jet_image_no_pid, pt_normalize    # noqa: E402

BINS, LIM = 32, 0.5           # image grid used by both models: 32 x 32 over |d_eta|, |d_phi| < 0.5

__all__ = ["BINS", "LIM", "load_jets", "balanced_sample", "load_transformer", "encode_transformer",
           "load_cnn", "encode_cnn", "decode_cnn", "decode_transformer", "particles_to_image",
           "cnn_input_image", "balanced_inputs", "resolve_device", "require_image_events",
           "validate_paper_cnn", "active_dimensions", "sigma_scalar_coefficients", "set_paper_style"]


def load_jets(path, max_particles=80):
    """Return dict(particles [N,K,3], mask [N,K], events list-of-[n,5] arrays or None)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            events = None
            if {"p4", "pid", "mask"} <= set(d.files):
                p4, pid, m = d["p4"], d["pid"], d["mask"].astype(bool)
                events = [np.column_stack([pid[i][m[i]], p4[i][m[i]]]).astype(np.float32)
                          for i in range(len(p4))]
            return dict(particles=d["particles80"], mask=d["mask80"].astype(bool), events=events)
    events = load_events_from_dat(path)
    particles, mask, _ = _arrays(events, max_particles)
    return dict(particles=particles, mask=mask, events=events)


def balanced_inputs(u_path, g_path, n_per_class=5000, seed=42):
    """Select the same seeded, sorted event indices for both representations."""
    if n_per_class < 1:
        raise ValueError("n_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected = []
    for label, path in enumerate((u_path, g_path)):
        jets = load_jets(path)
        if n_per_class > len(jets["particles"]):
            raise ValueError(f"{path}: fewer than {n_per_class} jets")
        idx = np.sort(rng.choice(len(jets["particles"]), n_per_class, replace=False))
        selected.append(dict(particles=jets["particles"][idx], mask=jets["mask"][idx],
                             events=None if jets["events"] is None else [jets["events"][i] for i in idx],
                             labels=np.full(n_per_class, label), file_indices=idx))
    return {key: (None if any(j[key] is None for j in selected) else
                  selected[0][key] + selected[1][key] if key == "events" else
                  np.concatenate([j[key] for j in selected])) for key in selected[0]}


def balanced_sample(u_path, g_path, n_per_class=5000, seed=42):
    """Legacy particle-only interface, with the same event selection as balanced_inputs."""
    jets = balanced_inputs(u_path, g_path, n_per_class, seed)
    return jets["particles"], jets["mask"]


# ----------------------------------------------------------------------------- CNN (image-level) VAE
def load_cnn(checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = {k: ck[k] for k in ("output_mode", "recenter_output", "eta_max", "phi_max", "loss_type", "beta",
                                "recon_loss_type", "recon_loss_weight", "beta_start", "beta_anneal_iter",
                                "beta_anneal_schedule") if k in ck}
    model = BetaVAE(in_channels=ck["n_channels"], latent_dim=ck["latent_dim"], image_size=ck["img_size"],
                    hidden_dims=list(ck.get("hidden_dims", [32, 64, 128, 256, 512])), **extra)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck


def cnn_input_image(event):
    """One event (rows pid px py pz E) -> normalised image [1, phi, eta] as the CNN sees it."""
    return pt_normalize(event_to_jet_image_no_pid(event, BINS, BINS, LIM, LIM))


@torch.no_grad()
def encode_cnn(model, images, batch=500, device="cpu"):
    model = model.to(device)
    mus, lvs = [], []
    for s in range(0, len(images), batch):
        mu, lv = model.encode(torch.from_numpy(images[s:s + batch]).float().to(device))
        mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


@torch.no_grad()
def decode_cnn(model, mu):
    device = next(model.parameters()).device
    return model.decode(torch.from_numpy(np.asarray(mu)).float().to(device)).cpu().numpy()      # [N, 1, phi, eta]


# ----------------------------------------------------------------------------- Transformer (particle-level) VAE
@torch.no_grad()
def decode_transformer(model, mu):
    device = next(model.parameters()).device
    z, eta, phi = model.decode(torch.from_numpy(np.asarray(mu)).float().to(device))
    return torch.stack([z, eta, phi], -1).cpu().numpy()                                # [N, 80, 3]


def particles_to_image(particles, mask=None, bins=BINS, lim=LIM):
    """(z, d_eta, d_phi) lists -> pT-fraction images [N, eta, phi] (d_eta vertical, d_phi horizontal),
    same binning as the CNN. Particles outside the window are dropped."""
    particles = np.asarray(particles)
    out = np.zeros((len(particles), bins, bins), dtype=np.float32)
    for i, p in enumerate(particles):
        m = np.ones(len(p), bool) if mask is None else np.asarray(mask[i], bool)
        z, de, dp = p[m, 0], p[m, 1], p[m, 2]
        ok = (np.abs(de) < lim) & (np.abs(dp) < lim)
        ie = np.clip(((de[ok] + lim) / (2 * lim) * bins).astype(int), 0, bins - 1)
        ip = np.clip(((dp[ok] + lim) / (2 * lim) * bins).astype(int), 0, bins - 1)
        np.add.at(out[i], (ie, ip), z[ok])
    return out


# ----------------------------------------------------------------------------- posterior diagnostics
def active_dimensions(mu, logvar):
    """Active latent dimension: Var_x[mu_j] / E_x[sigma_j^2] > 1."""
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    return mu.var(0) / np.exp(logvar).mean(0) > 1.0


def sigma_scalar_coefficients(logvar_active):
    """b_j of ln sigma_j^2 ~ a_j + b_j u(x): leading principal component of the centred ln sigma^2 (covariance, no
    rescaling of the individual variances), normalised to mean(b) = 1. Returns (b, relative spread)."""
    x = logvar_active - logvar_active.mean(0)
    _, v = np.linalg.eigh(np.cov(x, rowvar=False))
    b = v[:, -1]; b = b / b.mean()
    return b, float(b.std() / b.mean())


def set_paper_style(fontset="stix"):
    """Portable serif fonts; the posterior panels use Computer Modern."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "cmr10" if fontset == "cm" else "STIXGeneral",
        "mathtext.fontset": fontset, "axes.formatter.use_mathtext": True,
        "font.size": 11, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white",
    })


def resolve_device(device):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def require_image_events(jets):
    if jets["events"] is None:
        raise ValueError("CNN inference needs raw constituents: use .dat or .npz with p4, pid and mask arrays.")


def validate_paper_cnn(checkpoint):
    """Reject a stale CNN or a different representation before producing paper figures."""
    effective_beta = checkpoint.get("beta", float("nan")) / checkpoint.get("recon_loss_weight", 1.0)
    correct = (checkpoint.get("latent_dim") == 32 and checkpoint.get("img_size") == 32
               and checkpoint.get("n_channels") == 1 and np.isclose(effective_beta, 2e-6, rtol=1e-6, atol=0)
               and checkpoint.get("normalization") == "event_pt"
               and checkpoint.get("output_mode") == "event_pt")
    if not correct:
        raise ValueError("Current paper figures require the no-PID CNN32 with effective beta=2e-6 "
                         "and event_pt normalization (paper checkpoint: epoch 739).")
    print(f"CNN checkpoint: epoch {checkpoint.get('epoch')}, effective beta={effective_beta:g}")
