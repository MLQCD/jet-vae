"""Fig. 2: input jets and their reconstructions f(mu(x)) from the image-level (CNN) and particle-level (Transformer) VAE.

    python paper_figures/fig2_reconstruction.py --u-jet u_jet_real_sample.dat --g-jet g_jet_real_sample.dat \
        --cnn-checkpoint CNN32/best_model_betavae.pth --transformer-checkpoint run_2.pth --out fig2.pdf

The paper uses event 4 of the quark file and event 77395 of the gluon file. Both models see the same two jets.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (BINS, LIM, cnn_input_image, decode_cnn, decode_transformer, encode_cnn, encode_transformer,
                    load_cnn, load_jets, load_transformer, particles_to_image, set_paper_style)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--u-jet", required=True); p.add_argument("--g-jet", required=True)
    p.add_argument("--u-index", type=int, default=4); p.add_argument("--g-index", type=int, default=77395)
    p.add_argument("--cnn-checkpoint", required=True); p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--vmax", type=float, default=None, help="colour-scale maximum (default: largest input pixel)")
    p.add_argument("--out", default="fig2_reconstruction.pdf")
    a = p.parse_args(argv)
    set_paper_style()

    jets = [load_jets(a.u_jet), load_jets(a.g_jet)]
    idx = [a.u_index, a.g_index]
    events = [j["events"][i] for j, i in zip(jets, idx)]
    par = np.stack([j["particles"][i] for j, i in zip(jets, idx)])
    msk = np.stack([j["mask"][i] for j, i in zip(jets, idx)])

    # image-level VAE: input image, decode(mu)
    cnn, _ = load_cnn(a.cnn_checkpoint)
    img_in = np.stack([cnn_input_image(e) for e in events])                         # [2, 1, phi, eta]
    mu_c, _ = encode_cnn(cnn, img_in)
    cnn_in = img_in[:, 0].transpose(0, 2, 1)                                         # -> [eta, phi]
    cnn_out = decode_cnn(cnn, mu_c)[:, 0].transpose(0, 2, 1)

    # particle-level VAE: input list, decode(mu), both converted to the same image representation
    tr, _ = load_transformer(a.transformer_checkpoint)
    mu_t, _ = encode_transformer(tr, par, msk)
    tr_in = particles_to_image(par, msk)
    tr_out = particles_to_image(decode_transformer(tr, mu_t))

    vmax = a.vmax or float(max(cnn_in.max(), tr_in.max()))
    print(f"colour scale 0 .. {vmax:.4f}  (paper: 0.175)")
    print("input images, CNN vs particle list: max |difference| =", float(np.abs(cnn_in - tr_in).max()))

    from matplotlib.colors import LinearSegmentedColormap
    cmap_in = LinearSegmentedColormap.from_list("white_blue", ["white", "#0b4f9c"])
    cmap_out = LinearSegmentedColormap.from_list("white_orange", ["white", "#cc5500"])
    ticks = [0, vmax / 3, 2 * vmax / 3, vmax]
    fig = plt.figure(figsize=(14, 7.4))
    outer = fig.add_gridspec(1, 2, wspace=0.28, left=0.06, right=0.98, top=0.86, bottom=0.24)
    panels = [("(a) Image-level VAE", cnn_in, cnn_out), ("(b) Particle-level VAE", tr_in, tr_out)]
    ext = [-LIM, LIM, -LIM, LIM]
    maps = {}
    for k, (title, im_in, im_out) in enumerate(panels):
        inner = outer[k].subgridspec(2, 2, wspace=0.45, hspace=0.45)
        fig.text(0.27 + 0.47 * k, 0.955, title, ha="center", fontsize=15)
        fig.text(0.27 + 0.47 * k, 0.915, f"u: event {a.u_index}    g: event {a.g_index}", ha="center", fontsize=9, color="0.4")
        for r, (imgs, cmap, kind) in enumerate(((im_in, cmap_in, "Input"), (im_out, cmap_out, "Reconstruction"))):
            for c, cls in enumerate(("u", "g")):
                ax = fig.add_subplot(inner[r, c])
                maps[kind] = ax.imshow(imgs[c], origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax)
                ax.set_title(f"{cls} jet | {kind}", fontsize=11)
                ax.set_xlabel(r"$\Delta\phi$"); ax.set_ylabel(r"$\Delta\eta$")
                ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    for x0, kind, label in ((0.10, "Input", r"Input: normalized $p_T$ per pixel"),
                            (0.57, "Reconstruction", r"Reconstruction: normalized $p_T$ per pixel")):
        cb = fig.colorbar(maps[kind], cax=fig.add_axes([x0, 0.12, 0.30, 0.022]), orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels([f"{v:.3g}" for v in ticks]); cb.set_label(label, fontsize=10)
    fig.text(0.5, 0.02, "Illustrative examples | linear scale | deterministic decode(mu)", ha="center", fontsize=9, color="0.4")
    fig.savefig(a.out, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
