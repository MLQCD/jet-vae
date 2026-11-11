"""Fig. 3: Pearson correlation matrices of the posterior means mu_j and of ln sigma_j^2 among the active dimensions of
the particle-level (Transformer) VAE, on a balanced sample (default 5000 quark + 5000 gluon jets, seed 42).

    python paper_figures/fig3_posterior_structure.py --u-jet u_jet.npz --g-jet g_jet.npz --transformer-checkpoint run_2.pth

Also prints the number of active dimensions and the relative spread of the scaling coefficients b_j (paper: 32 and 0.0723).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (active_dimensions, balanced_sample, encode_transformer, load_transformer,
                    set_paper_style, sigma_scalar_coefficients)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--u-jet", required=True); p.add_argument("--g-jet", required=True)
    p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--n-per-class", type=int, default=5000); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="fig3_posterior_structure.pdf")
    a = p.parse_args(argv)
    set_paper_style()

    par, msk = balanced_sample(a.u_jet, a.g_jet, a.n_per_class, a.seed)
    model, _ = load_transformer(a.transformer_checkpoint)
    mu, lv = encode_transformer(model, par, msk)
    act = active_dimensions(mu, lv); idx = np.flatnonzero(act)
    b, spread = sigma_scalar_coefficients(lv[:, act])
    c_mu = np.corrcoef(mu[:, act], rowvar=False); c_lv = np.corrcoef(lv[:, act], rowvar=False)
    off = ~np.eye(len(idx), dtype=bool)
    print(f"{len(par)} jets, {act.sum()} active dimensions of {mu.shape[1]}; relative spread of b_j = {spread:.4f}")
    print(f"mean |off-diagonal| correlation: mu {np.abs(c_mu[off]).mean():.3f}, ln sigma^2 {np.abs(c_lv[off]).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(wspace=0.3))
    for ax, c, title in ((axes[0], c_mu, r"(a) Posterior means $\mu_j$"), (axes[1], c_lv, r"(b) Log variances $\ln\sigma_j^2$")):
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(idx))); ax.set_yticklabels(idx, fontsize=6.5)
        ax.set_xlabel("Latent index", fontsize=13); ax.set_ylabel("Latent index", fontsize=13)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1]); cb.set_label("Pearson correlation", fontsize=13)
    fig.suptitle("Particle-level VAE", fontsize=14)
    fig.savefig(a.out, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
