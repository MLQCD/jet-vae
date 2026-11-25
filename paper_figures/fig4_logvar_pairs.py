"""Legacy standalone diagnostic (these distributions are now integrated into Fig. 3).

Joint distributions of ln sigma^2 for five pairs of active latent dimensions of the particle-level VAE
(same balanced sample as Fig. 3). Pairs are given by their original latent indices.

    python paper_figures/fig4_logvar_pairs.py --u-jet u_jet.npz --g-jet g_jet.npz --transformer-checkpoint run_2.pth
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import active_dimensions, balanced_sample, encode_transformer, load_transformer, set_paper_style

PAIRS = [(0, 1), (0, 2), (4, 10), (16, 20), (3, 29)]        # as in the paper


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--u-jet", required=True); p.add_argument("--g-jet", required=True)
    p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--n-per-class", type=int, default=5000); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="fig4_logvar_pairs.pdf")
    a = p.parse_args(argv)
    set_paper_style()

    par, msk = balanced_sample(a.u_jet, a.g_jet, a.n_per_class, a.seed)
    model, _ = load_transformer(a.transformer_checkpoint)
    mu, lv = encode_transformer(model, par, msk)
    active = list(np.flatnonzero(active_dimensions(mu, lv)))
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), gridspec_kw=dict(wspace=0.38))
    for ax, (i, j), tag in zip(axes, PAIRS, "abcde"):
        x, y = lv[:, i], lv[:, j]
        slope = np.polyfit(x, y, 1)[0]                                        # ordinary least squares
        w, v = np.linalg.eigh(np.cov(x, y)); tls = v[1, -1] / v[0, -1]           # leading principal axis
        hb = ax.hexbin(x, y, gridsize=45, bins="log", cmap="viridis", mincnt=1, norm=LogNorm(vmin=1))
        ax.set_facecolor("#0b1020")
        ax.set_title(f"({tag}) {active.index(i) + 1}, {active.index(j) + 1}", fontsize=13)
        ax.set_xlabel(rf"$\ln\sigma_{{{i}}}^2$", fontsize=13); ax.set_ylabel(rf"$\ln\sigma_{{{j}}}^2$", fontsize=13)
        print(f"({tag}) dims {i},{j}: slope OLS {slope:.3f}, principal-axis {tls:.3f}, corr {np.corrcoef(x, y)[0, 1]:.3f}")
    cb = fig.colorbar(hb, ax=axes, fraction=0.012, pad=0.01); cb.set_label("Jets / bin", fontsize=12)
    fig.savefig(a.out, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
