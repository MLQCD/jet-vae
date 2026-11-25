"""Portable appendix plotting style and PDF/PNG output."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def set_style():
    plt.rcParams.update({"font.family": "cmr10", "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .8,
        "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white"})


def style_axes(ax):
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.5, width=.7)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.8, width=.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)


def save(fig, out, tight=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    options = dict(bbox_inches="tight", pad_inches=.08) if tight else {}
    fig.savefig(out, **options)
    fig.savefig(out.with_suffix(".png"), dpi=200, **options)
    plt.close(fig)
