"""Load a trained model and encode jets to posterior (mu, logvar).

    python -m jetvae.transformer.encode --checkpoint runs/s1/best_model.pth \
        --cache-dir cache/ --split val --out latents_val.npz

Output npz: ``mu``, ``logvar`` [N, latent_dim], plus ``labels`` if the cache has them.
Accepts both full training checkpoints and slim ones (``config`` sub-dict).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import ParticleTransformerShapeEMDVAE

__all__ = ["load_model", "encode"]


def load_model(checkpoint, device="cpu"):
    """Return ``(model in eval mode, checkpoint dict)``."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"] if isinstance(ck.get("config"), dict) else ck
    model = ParticleTransformerShapeEMDVAE(
        max_particles=c["max_particles"], latent_dim=c["latent_dim"],
        embed_dim=c["embed_dim"], depth=c["depth"], num_heads=c["num_heads"],
        dropout=c["dropout"], beta=c["beta"],
        reconstruction_weight=c["reconstruction_weight"],
        sinkhorn_epsilon=c.get("sinkhorn_epsilon_final") or c["sinkhorn_epsilon"],
        sinkhorn_n_iters=c.get("sinkhorn_n_iters_final") or c["sinkhorn_n_iters"],
        sinkhorn_p=c["sinkhorn_p"], angular_scale=c["angular_scale"],
        sinkhorn_debias=not c["no_sinkhorn_debias"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def encode(model, particles, mask, batch=1000, device="cpu"):
    """Posterior mean and log-variance as numpy arrays [N, latent_dim]."""
    mus, lvs = [], []
    for s in range(0, len(particles), batch):
        p = torch.from_numpy(np.asarray(particles[s:s + batch])).float().to(device)
        m = torch.from_numpy(np.asarray(mask[s:s + batch])).bool().to(device)
        mu, lv = model.encode(p, m)
        mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    cache = Path(a.cache_dir)
    model, _ = load_model(a.checkpoint, a.device)
    par = np.load(cache / f"{a.split}_particles.npy", mmap_mode="r")
    msk = np.load(cache / f"{a.split}_mask.npy", mmap_mode="r")
    mu, lv = encode(model, par, msk, device=a.device)
    out = {"mu": mu, "logvar": lv}
    lab = cache / f"{a.split}_labels.npy"
    if lab.exists():
        out["labels"] = np.load(lab)
    np.savez(a.out, **out)
    snr = mu.var(0) / np.exp(lv).mean(0)
    print(f"wrote {a.out}: {mu.shape}, active dims (SNR>1): {int((snr > 1).sum())}/{mu.shape[1]}")


if __name__ == "__main__":
    main()
