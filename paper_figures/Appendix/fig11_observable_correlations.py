"""Current Fig. 11: signed Pearson correlations of 32 means and logvar_0 with 9 observables."""
import argparse
from pathlib import Path
import numpy as np
from analysis import HERE, read_json
from plot_style import plt, save
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


def plot(data, out):
    corr = np.asarray(data["correlations"])
    if corr.shape != (33,9) or not np.isfinite(corr).all() or np.max(np.abs(corr)) > 1 + 1e-12:
        raise ValueError("Expected a finite signed 33 x 9 correlation matrix.")
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "STIXGeneral"
    plt.rcParams.update({"font.family": font, "mathtext.fontset": "stix", "font.size": 12,
        "axes.labelsize":13,"axes.titlesize":15,"xtick.labelsize":10,"ytick.labelsize":10,
        "axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in","pdf.fonttype":42,
        "ps.fonttype":42,"figure.facecolor":"white","savefig.facecolor":"white"})
    cmap = LinearSegmentedColormap.from_list("blue_white_orange",["#51A7C0","white","#FF8000"],N=4097)
    pretty = [r"Multiplicity $N$",r"$p_TD$",r"Entropy $H_{p_T}$",r"LHA $\lambda_{1/2}$",r"Girth $\lambda_1$",
              r"$\lambda_2$",r"$e_2^{(1/2)}$",r"$e_2^{(1)}$",r"$\ln(m_J^2/\mathrm{GeV}^2)$"]
    latent = [rf"$\mu_{{{j}}}$" for j in range(32)] + [r"$\ln\sigma_0^2$"]
    fig, ax = plt.subplots(figsize=(10.1,12.3))
    fig.subplots_adjust(left=.12,right=.865,bottom=.16,top=.9)
    mesh = ax.pcolormesh(np.arange(10)-.5,np.arange(34)-.5,corr,cmap=cmap,vmin=-1,vmax=1,shading="flat")
    ax.set(xlim=(-.5,8.5),ylim=(32.5,-.5))
    ax.set_xticks(range(9),pretty,rotation=48,ha="right",rotation_mode="anchor",fontsize=12)
    ax.set_yticks(range(33),latent,fontsize=11)
    ax.tick_params(length=0,pad=7)
    ax.set_ylabel("Latent feature",labelpad=8)
    ax.axhline(31.5,color="#333333",lw=1.2)
    for row in range(33):
        for col in range(9):
            value = corr[row,col]
            label = f"{value:.2f}" if abs(value)>=.005 else "0.00"
            ax.text(col,row,label,ha="center",va="center",fontsize=9.4,color="#161616",fontweight="bold" if row==32 else "normal")
    fig.suptitle(r"Particle-level VAE: $\{\mu_0,\ldots,\mu_{31},\ln\sigma_0^2\}$ vs. physical observables",y=.94,fontsize=14)
    cax = fig.add_axes([.892,.25,.021,.57])
    cb = fig.colorbar(mesh,cax=cax,ticks=[-1,-.5,0,.5,1])
    cb.set_label(r"Pearson correlation $r$",labelpad=10)
    cb.solids.set_rasterized(False)
    save(fig, out, tight=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE/"data/fig11_correlations.json")
    p.add_argument("--out", type=Path, default=Path("figs/fig11_observable_correlations.pdf"))
    a = p.parse_args()
    plot(read_json(a.data), a.out)


if __name__ == "__main__":
    main()
