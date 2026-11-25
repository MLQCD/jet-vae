"""Fig. 3: posterior correlations and three common log-variance pairs for both VAEs.

Uses the same seeded 5000 u + 5000 g jets for both encoders. The correlation panels
include all active dimensions. The image collective-width statistic excludes
indices 0,11,17,30, as in the paper; the particle statistic uses all active indices.
Density pairs (original, zero-based indices) are (1,2), (16,20), (3,29).

Example (from the repository root):
    python paper_figures/fig3_posterior_structure.py --u-jet u_jet.dat --g-jet g_jet.dat \
        --transformer-checkpoint transformer.pth --cnn-checkpoint cnn_beta2e-6.pth
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, NullFormatter, ScalarFormatter
import numpy as np

from common import (active_dimensions, balanced_inputs, cnn_input_image, encode_cnn,
                    encode_transformer, load_cnn, load_transformer, require_image_events,
                    resolve_device, set_paper_style, sigma_scalar_coefficients, validate_paper_cnn)

PAIRS = [(1, 2), (16, 20), (3, 29)]
CNN_COLLECTIVE_EXCLUDE = [0, 11, 17, 30]


class ShortMinusTenFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        label = super().__call__(x, pos)
        return label.replace("10.0", "10") if abs(x + 10) < 1e-10 else label


def posterior_summary(mu, logvar, excluded=()):
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    active = np.flatnonzero(active_dimensions(mu, logvar))
    if len(active) < 2:
        raise ValueError("At least two active dimensions are needed for the correlation panels")
    collective = np.array([j for j in active if j not in excluded], dtype=int)
    if len(collective) < 2:
        raise ValueError("At least two retained dimensions are needed for the collective-width statistic")
    return dict(mu=mu, logvar=logvar, active=active, collective=collective,
                corr_mu=np.corrcoef(mu[:, active], rowvar=False),
                corr_logvar=np.corrcoef(logvar[:, active], rowvar=False))


def plot_posterior_structure(data, out, pairs=PAIRS):
    """Draw both models with identical pair bin scales and a shared count colorbar."""
    for name, d in data.items():
        if any(j not in d["active"] for pair in pairs for j in pair):
            raise ValueError(f"{name}: a selected density-pair dimension is inactive")
    set_paper_style("cm")
    W,H=12.1,7.95
    fig=plt.figure(figsize=(W,H))
    def axes(x,y,w,h):return fig.add_axes([x/W,(y+.30)/H,w/W,h/H])
    blue=np.array([81,167,192])/255
    corr_cmap=LinearSegmentedColormap.from_list('blue_white_orange',[blue,'white','#FF8000'],N=4097)
    display=np.array([0.1875,0.54687,0.347656])
    cmap=LinearSegmentedColormap.from_list('jet_density',[1-.18*(1-display),display],N=4097,gamma=.70)
    model_positions={'particle':(.62,3.88),'image':(6.58,9.84)}
    letters={'particle':('a','b','c'),'image':('d','e','f')}
    images=[];hexes=[];counts=[]
    for m,(xc,xd) in model_positions.items():
        d=data[m];active=d['active'];la,lb,ld=letters[m]
        fig.text((xc+xd+1.65)/2/W,7.73/H,('Particle-level' if m=='particle' else 'Image-level')+' VAE',ha='center',va='center',fontsize=16)
        for quantity,y,letter,title in [('mu',4.43,la,r'Means $\mu_j$'),('logvar',.88,lb,r'Log variances $\ell_j$')]:
            ax=axes(xc,y,2.55,2.55)
            images.append(ax.imshow(d['corr_'+quantity],vmin=-1,vmax=1,cmap=corr_cmap,origin='upper',interpolation='nearest',rasterized=True))
            ticks=[i for i,j in enumerate(active) if j%4==0]
            if not ticks: ticks=[0]
            if len(active)-1-ticks[-1]>=2:ticks.append(len(active)-1)
            ax.set_xticks(ticks,active[ticks]);ax.set_yticks(ticks,active[ticks])
            ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=10,length=3,pad=3)
            ax.set_xlabel('Latent index',fontsize=12,labelpad=3);ax.set_ylabel('Latent index',fontsize=12,labelpad=3)
            ax.set_title(f'({letter})  '+title,fontsize=13,pad=7)
        fig.text((xd+.825)/W,7.36/H,f'({ld})',ha='center',va='bottom',fontsize=13)
        for idx,((j,k),y) in enumerate(zip(pairs,[5.33,3.105,.88])):
            ax=axes(xd,y,1.65,1.65)
            x,yv=d['logvar'][:,j],d['logvar'][:,k]
            # Equal square bin scale for a given pair in both models; centers follow each model.
            span=max(float(np.ptp(data[n]['logvar'][:,q])) for n in data for q in [j,k])*1.08
            xm=(x.min()+x.max())/2;ym=(yv.min()+yv.max())/2
            ext=(xm-span/2,xm+span/2,ym-span/2,ym+span/2)
            hb=ax.hexbin(x,yv,gridsize=42,extent=ext,mincnt=1,cmap=cmap,linewidths=0,rasterized=True)
            assert int(hb.get_array().sum()) == len(x)
            hexes.append(hb);counts.append({'model':m,'pair':[j,k],'count':int(hb.get_array().sum()),'max_bin':int(hb.get_array().max()),'extent':list(map(float,ext)),'span':span})
            ax.set_xlim(ext[:2]);ax.set_ylim(ext[2:]);ax.set_aspect('equal',adjustable='box')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=11,length=3,pad=2)
            ax.tick_params(which='minor',length=1.5)
            ax.set_xlabel(rf'$\ell_{{{j}}}=\ln\sigma_{{{j}}}^2$',fontsize=12,labelpad=1)
            ax.set_ylabel(rf'$\ell_{{{k}}}=\ln\sigma_{{{k}}}^2$',fontsize=12,labelpad=2)
    vmax=max(h.get_array().max() for h in hexes)
    for h in hexes:h.set_norm(LogNorm(vmin=1,vmax=vmax))
    cb=fig.colorbar(images[0],cax=axes(.62,.23,4.91,.095),orientation='horizontal',ticks=[-1,-.5,0,.5,1])
    cb.ax.tick_params(labelsize=10,length=2,pad=2);cb.set_label('Pearson correlation',fontsize=12,labelpad=1);cb.outline.set_linewidth(.5)
    cb2=fig.colorbar(hexes[0],cax=axes(6.58,.23,4.91,.095),orientation='horizontal',ticks=[v for v in [1,10,100,1000] if v<=vmax])
    cb2.ax.set_xlim(1,vmax);cb2.ax.xaxis.set_minor_formatter(NullFormatter());cb2.ax.tick_params(labelsize=10,length=2,pad=2)
    cb2.set_label('Jets per bin',fontsize=12,labelpad=1);cb2.outline.set_linewidth(.5)
    target = hexes[4].axes
    for axis in (target.xaxis, target.yaxis):
        axis.set_major_formatter(ShortMinusTenFormatter(useMathText=True))
    fig.canvas.draw()
    right_axes = [h.axes for h in hexes[3:]]
    label_x = min(ax.yaxis.label.get_transform().transform(ax.yaxis.label.get_position())[0]
                  for ax in right_axes)
    for ax in right_axes:
        xx = ax.transAxes.inverted().transform((label_x, ax.bbox.y0))[0]
        ax.yaxis.set_label_coords(xx, 0.5)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print("saved", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-jet", required=True); p.add_argument("--g-jet", required=True)
    p.add_argument("--transformer-checkpoint", required=True); p.add_argument("--cnn-checkpoint", required=True)
    p.add_argument("--n-per-class", type=int, default=5000); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256); p.add_argument("--device", default="auto")
    p.add_argument("--out", default="fig3_posterior_structure.pdf")
    a = p.parse_args(argv)
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    jets = balanced_inputs(a.u_jet, a.g_jet, a.n_per_class, a.seed); require_image_events(jets)
    device = resolve_device(a.device)
    cnn, ck = load_cnn(a.cnn_checkpoint); validate_paper_cnn(ck)
    tr, _ = load_transformer(a.transformer_checkpoint, device)
    images = np.stack([cnn_input_image(e) for e in jets["events"]])
    mc, lc = encode_cnn(cnn, images, batch=a.batch_size, device=device)
    mt, lt = encode_transformer(tr, jets["particles"], jets["mask"], batch=a.batch_size, device=device)
    data = dict(particle=posterior_summary(mt, lt),
                image=posterior_summary(mc, lc, CNN_COLLECTIVE_EXCLUDE))
    for name, d in data.items():
        _, spread = sigma_scalar_coefficients(d["logvar"][:, d["collective"]])
        off = ~np.eye(len(d["active"]), dtype=bool)
        print(f"{name}: {len(d['active'])} active, {len(d['collective'])} collective; "
              f"relative spread of b_j = {spread:.6f}; "
              f"median |corr(mu)| = {np.median(np.abs(d['corr_mu'][off])):.6f}; "
              f"median corr(logvar) = {np.median(d['corr_logvar'][off]):.6f}")
    plot_posterior_structure(data, a.out)


if __name__ == "__main__":
    main()
