"""Current Fig. 10: all 32 width responses of multiplicity-matched RAMBO samples."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import read_json
from plot_style import plt, set_style, style_axes, save
from matplotlib.ticker import NullLocator
HERE = Path(__file__).resolve().parent


def plot(data, out):
    x, delta = np.asarray(data["mean_N"]), np.asarray(data["delta_mean_logvar"])
    if x.shape != (5,) or delta.shape != (5,32) or not np.all(delta[2] == 0):
        raise ValueError("Expected five samples, 32 dimensions, and nominal index 2.")
    np.testing.assert_allclose(delta, np.asarray(data["mean_logvar"]) - np.asarray(data["mean_logvar"])[2], atol=1e-14)
    set_style()
    fig, ax = plt.subplots(figsize=(4.7,2.8))
    fig.subplots_adjust(left=.16,right=.97,bottom=.20,top=.85)
    ax.axhline(0,color=".65",lw=.55,zorder=0)
    for j in range(32):
        ax.plot(x,delta[:,j],color=(33/255,132/255,161/255),lw=.65,alpha=.38,marker="o",ms=2.4,mew=0,zorder=2)
    ax.plot(x[2],0,"o",color="black",ms=2.8,zorder=5)
    ax.set(xlim=(x[0]-.24,x[-1]+.24),ylim=data["plot_ylim"],yticks=np.arange(-2,3)/10)
    ax.set_xticks(x,[f"{v:.2f}" for v in x])
    ax.set_xlabel(r"Mean RAMBO multiplicity $\langle N\rangle$",labelpad=4)
    ax.set_ylabel(r"$\Delta\langle\ln\sigma_j^2\rangle$",labelpad=5)
    ax.set_title("RAMBO response",loc="left",pad=8,fontsize=11)
    style_axes(ax)
    ax.xaxis.set_minor_locator(NullLocator())
    save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE/"data/fig10_response.json")
    p.add_argument("--out", type=Path, default=Path("figs/fig10_rambo_response.pdf"))
    a = p.parse_args()
    plot(read_json(a.data), a.out)


if __name__ == "__main__":
    main()
