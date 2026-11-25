"""Fig. 2: input, particle reconstruction and image reconstruction (three rows, u/g columns).

Defaults reproduce the current illustrative events, file-local indices 45050 and 77395,
with the epoch-739 CNN (effective beta=2e-6) and epoch-134 Transformer (beta=1e-5).
Run from any directory, passing both raw sample files and checkpoint paths.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from common import (cnn_input_image, decode_cnn, decode_transformer, encode_cnn, encode_transformer,
                    load_cnn, load_jets, load_transformer, particles_to_image, set_paper_style,
                    resolve_device, require_image_events, validate_paper_cnn)


def plot_reconstructions(panels, out, vmax=0.3):
    """panels[class, input/particle/image, eta, phi]; fixed paper color scale."""
    if np.shape(panels) != (2, 3, 32, 32) or vmax <= 0:
        raise ValueError("Expected six 32x32 maps and a positive color maximum")
    set_paper_style()
    norm=Normalize(0,vmax)
    cmaps=[LinearSegmentedColormap.from_list('original_input',['white',np.array([81/255,167/255,192/255])*.9],N=4097,gamma=.62),
     LinearSegmentedColormap.from_list('original_reco',['white',np.array([1.,.5,0])*.9],N=4097,gamma=.62)]
    W,H=4.95,5.65
    side=1.50
    xs=[.87,2.62]
    ys=[3.82,2.12,.42]
    fig=plt.figure(figsize=(W,H))
    axes=[]
    row_labels=['Input','Particle-level VAE\nreconstruction','Image-level VAE\nreconstruction']
    for r in range(3):
        fig.text(.15/W,(ys[r]+side/2)/H,row_labels[r],rotation=90,fontsize=10.5,ha='center',va='center',linespacing=1.05)
        for c in range(2):
            ax=fig.add_axes([xs[c]/W,ys[r]/H,side/W,side/H])
            ax.imshow(panels[c,r],origin='lower',extent=(-.5,.5,-.5,.5),norm=norm,
                      cmap=cmaps[0 if r==0 else 1],interpolation='nearest',aspect='equal')
            ticks=[-.5,-.25,0,.25,.5]
            labels=['-0.5','-0.25','0','0.25','0.5']
            ax.set_xticks(ticks,labels);ax.set_yticks(ticks,labels)
            ax.set_xlim(-.5,.5);ax.set_ylim(-.5,.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which='major',top=True,right=True,length=3.2,width=.7,labelsize=9.5,pad=5,
                           labelbottom=(r==2),labelleft=(c==0))
            ax.tick_params(which='minor',top=True,right=True,length=1.6,width=.5)
            if r==2:ax.set_xlabel(r'$\Delta\phi$',fontsize=11.5,labelpad=1)
            if c==0:ax.set_ylabel(r'$\Delta\eta$',fontsize=11.5,labelpad=1)
            axes.append(ax)
    for x,jet in zip(xs,['u','g']):
        fig.text((x+side/2)/W,5.47/H,rf'${jet}$ jet',fontsize=11.5,ha='center',va='center')
    for r0,r1,cmap in [(0,0,cmaps[0]),(1,2,cmaps[1])]:
        cax=fig.add_axes([4.25/W,ys[r1]/H,.065/W,(ys[r0]+side-ys[r1])/H])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='vertical')
        cb.set_ticks(np.linspace(0,vmax,4),labels=[f'{v:g}' for v in np.linspace(0,vmax,4)])
        cb.set_label(r'Normalized $p_T$ per pixel',fontsize=10,labelpad=4)
        cb.ax.tick_params(direction='in',length=3,width=.6,labelsize=9.5,pad=3)
        cb.outline.set_linewidth(.65)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print("saved", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-jet", required=True); p.add_argument("--g-jet", required=True)
    p.add_argument("--u-index", type=int, default=45050); p.add_argument("--g-index", type=int, default=77395)
    p.add_argument("--cnn-checkpoint", required=True); p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--device", default="cpu", help="CPU matches the paper reconstruction; GPU kernels may differ slightly")
    p.add_argument("--vmax", type=float, default=0.3)
    p.add_argument("--out", default="fig2_reconstruction.pdf")
    a = p.parse_args(argv)
    device = resolve_device(a.device)
    events, particles, masks = [], [], []
    for path, idx in ((a.u_jet, a.u_index), (a.g_jet, a.g_index)):
        jets = load_jets(path); require_image_events(jets)
        if not 0 <= idx < len(jets["particles"]):
            p.error(f"Event index {idx} is outside {path}")
        events.append(jets["events"][idx]); particles.append(jets["particles"][idx]); masks.append(jets["mask"][idx])
    par, mask = np.stack(particles), np.stack(masks)
    cnn, ck = load_cnn(a.cnn_checkpoint); validate_paper_cnn(ck)
    tr, _ = load_transformer(a.transformer_checkpoint, device)
    images = np.stack([cnn_input_image(e) for e in events])
    mu_c, _ = encode_cnn(cnn, images, device=device)
    mu_t, _ = encode_transformer(tr, par, mask, device=device)
    panels = np.stack([images[:, 0].transpose(0, 2, 1),
                       particles_to_image(decode_transformer(tr, mu_t)),
                       decode_cnn(cnn, mu_c)[:, 0].transpose(0, 2, 1)], axis=1)
    print(f"File-local event indices: u={a.u_index}, g={a.g_index}; deterministic decode(mu)")
    plot_reconstructions(panels, a.out, a.vmax)


if __name__ == "__main__":
    main()
