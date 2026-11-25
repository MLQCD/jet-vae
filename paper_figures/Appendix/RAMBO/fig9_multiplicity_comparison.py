"""Current Fig. 9: QCD/RAMBO common-width distributions and conditional profiles."""
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import read_json, XEDGES, YEDGES
from plot_style import plt, set_style, style_axes, save
from matplotlib.colors import LogNorm, LinearSegmentedColormap
HERE = Path(__file__).resolve().parent
COLORS = {"u": .9*np.array([81/255, 167/255, 192/255]),
          "g": .9*np.array([1, .5, 0]), "rambo": np.array([.1875, .54687, .347656])}


def smooth_profile(rows):
    n = np.array([r["N"] for r in rows])
    mean = np.array([r["mean"] for r in rows])
    sem = np.array([r["sem"] for r in rows])
    if len(n) < 4 or np.any(sem <= 0):
        raise ValueError("Need at least four populated N bins with positive SEM.")
    spline = UnivariateSpline(n, mean, w=1/sem, s=.6*len(n), k=3, ext=2)
    grid = np.unique(np.r_[n, np.linspace(n[0], n[-1], int((n[-1]-n[0])*30)+1)])
    return grid, spline(grid)


def plot(data, out):
    set_style()
    samples = data["samples"]
    curves = {key: smooth_profile(samples[key]["profiles"]) for key in COLORS}
    fig = plt.figure(figsize=(7.2, 5.0))
    rects = [(.075,.59,.345,.335), (.59,.59,.345,.335), (.075,.125,.345,.335), (.59,.125,.345,.335)]
    axes = [fig.add_axes(rect) for rect in rects]
    titles = [r"(a) $u$ jets", r"(b) $g$ jets", "(c) RAMBO", "(d) Summary of distributions"]
    for ax, title in zip(axes, titles):
        ax.set(xlim=(.5,80.5), ylim=(-1.5,4), xticks=[20,40,60,80], yticks=[-1,0,1,2,3,4])
        ax.set_xlabel(r"Particle multiplicity $N$", labelpad=3)
        ax.set_ylabel(r"$u(x)$", labelpad=3)
        ax.set_title(title, loc="left", pad=6)
        style_axes(ax)
    for i, key in enumerate(("u", "g", "rambo")):
        color, sample = COLORS[key], samples[key]
        counts = np.asarray(sample["counts"])
        if counts.shape != (80,75) or counts.sum() != sample["n_events"]:
            raise ValueError("Histogram shape or event normalization mismatch")
        cmap = LinearSegmentedColormap.from_list(key, [1-.18*(1-color), color], N=4097, gamma=.70)
        cmap.set_bad("white")
        mesh = axes[i].pcolormesh(XEDGES, YEDGES, np.ma.masked_equal((counts/sample["n_events"]).T, 0),
                                  norm=LogNorm(vmin=1e-5,vmax=1e-2), cmap=cmap, rasterized=True)
        axes[i].plot(*curves[key], color="black", lw=.85, label=r"Mean at fixed $N$", zorder=6)
        axes[i].legend(loc="upper right", frameon=False, fontsize=8, handlelength=2.3)
        rect = rects[i]
        cax = fig.add_axes([rect[0]+rect[2]+.012, rect[1], .012, rect[3]])
        cb = fig.colorbar(mesh,cax=cax,ticks=[1e-5,1e-4,1e-3,1e-2])
        cb.ax.tick_params(which="major",direction="in",labelsize=8,length=2.5,width=.6)
        cb.ax.tick_params(which="minor",direction="in",length=1.2,width=.4)
        cb.outline.set_linewidth(.6)
    labels = {"u": r"$u$ jets", "g": r"$g$ jets", "rambo": "RAMBO"}
    for key in ("rambo", "g", "u"):
        rows = samples[key]["profiles"]
        axes[3].fill_between([r["N"] for r in rows], [r["q16"] for r in rows], [r["q84"] for r in rows],
                             color=COLORS[key], alpha=.16, linewidth=0)
        axes[3].plot(*curves[key], color=COLORS[key], lw=1.25, label=labels[key], zorder=3)
    handles, names = axes[3].get_legend_handles_labels()
    axes[3].legend(handles[::-1], names[::-1], loc="upper right", frameon=False, fontsize=9)
    save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE/"data/fig9_distributions.json")
    p.add_argument("--out", type=Path, default=Path("figs/fig9_rambo_multiplicity.pdf"))
    a = p.parse_args()
    plot(read_json(a.data), a.out)


if __name__ == "__main__":
    main()
