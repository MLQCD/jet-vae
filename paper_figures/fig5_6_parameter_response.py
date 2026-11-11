"""Figs. 5 and 6: response of the posterior widths (sigma-space) to Pythia parameters.

The trained encoders are kept fixed. For each of the 33 settings (8 parameters x 4 relative changes + nominal) the
10 000 quark and 10 000 gluon jets are encoded and the mean log variance <ln sigma_j^2> is computed on the
active dimensions (defined once, on the nominal mixed sample). Then

    Delta<l_j>(p) = <l_j>_p - <l_j>_p0                         (Eq. 'widthshift')
    Delta<l_j>(p) ~ c_jp + s_jp (p - p0)/p0                    free intercept, the four non-zero points

    python paper_figures/fig5_6_parameter_response.py --scan-dir <200GeV_process> \
        --cnn-checkpoint CNN32/best_model_betavae.pth --transformer-checkpoint run_2.pth --out-dir figs/

Expected layout of --scan-dir:  {u_jet,g_jet}/<parameter>/{minus_20pct,minus_10pct,plus_10pct,plus_20pct}.npz
and {u_jet,g_jet}/nominal.npz, each file holding 10 000 jets (arrays particles80, mask80, p4, pid, mask, metadata).
The per-setting means are cached in <out-dir>/scan_means.npz, so replotting is instant.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (active_dimensions, cnn_input_image, encode_cnn, encode_transformer, load_cnn, load_jets,
                    load_transformer, set_paper_style)

PARAMS = [  # (folder, label, colour)
    ("alphaS", r"$\alpha_s$", "#2563eb"), ("aLund", r"$a_{\rm Lund}$", "#e07b00"),
    ("bLund", r"$b_{\rm Lund}$", "#1a8f5a"), ("sigma", r"$\sigma_{\rm string}$", "#8b5cf6"),
    ("pTmin", r"$p_{T,\rm min}$", "#c0392b"), ("mesonUDvector", r"$r_{V/P}$", "#0f8b8d"),
    ("probStoUD", r"$r_{s/(u,d)}$", "#a64d79"), ("probQQtoQ", r"$r_{qq/q}$", "#7a5c3e")]
CHANGES = [-0.2, -0.1, 0.0, 0.1, 0.2]
FILES = {-0.2: "minus_20pct", -0.1: "minus_10pct", 0.1: "plus_10pct", 0.2: "plus_20pct"}


def setting_paths(root, param, change):
    name = "nominal.npz" if change == 0.0 else f"{param}/{FILES[change]}.npz"
    return [Path(root) / cls / name for cls in ("u_jet", "g_jet")]


def load_setting(root, param, change):
    """Quark and gluon files of one setting, checked against their own metadata, concatenated."""
    jets = []
    for path in setting_paths(root, param, change):
        meta = json.loads(str(np.load(path)["metadata"]))
        assert meta["jets"] == 10000, path
        if change != 0.0:
            assert abs(meta["relative_change"] - change) < 1e-9, (path, meta["relative_change"])
        jets.append(load_jets(path))
    return dict(particles=np.concatenate([j["particles"] for j in jets]), mask=np.concatenate([j["mask"] for j in jets]),
                events=jets[0]["events"] + jets[1]["events"])


def compute_means(a):
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    cnn, _ = load_cnn(a.cnn_checkpoint); cnn = cnn.to(dev)
    tr, _ = load_transformer(a.transformer_checkpoint, dev)

    def encode(jets):
        img = np.stack([cnn_input_image(e) for e in jets["events"]])
        c = encode_cnn(cnn, img, device=dev)
        t = encode_transformer(tr, jets["particles"], jets["mask"], device=dev)
        return {"cnn": c, "transformer": t}

    nominal = encode(load_setting(a.scan_dir, PARAMS[0][0], 0.0))
    active = {m: active_dimensions(*nominal[m]) for m in nominal}
    print("active dimensions at nominal:", {m: int(v.sum()) for m, v in active.items()}, "(paper: cnn 30, transformer 32)")
    nominal_mean = {m: nominal[m][1].mean(0) for m in nominal}
    out = {m: np.zeros((len(PARAMS), len(CHANGES), nominal_mean[m].size)) for m in nominal}
    for ip, (param, _, _) in enumerate(PARAMS):
        for ic, ch in enumerate(CHANGES):
            if ch == 0.0:
                for m in out: out[m][ip, ic] = nominal_mean[m]
                continue
            enc = encode(load_setting(a.scan_dir, param, ch))
            for m in out: out[m][ip, ic] = enc[m][1].mean(0)
            print(f"  {param:14s} {ch:+.1f} done", flush=True)
    return {"mean_lnvar_cnn": out["cnn"], "mean_lnvar_transformer": out["transformer"],
            "active_cnn": active["cnn"], "active_transformer": active["transformer"]}


def responses(means, active):
    """delta[p, change, active j] and fitted slopes s[p, j] (free intercept, non-zero changes only)."""
    delta = (means - means[:, [CHANGES.index(0.0)]])[:, :, active]
    x = np.array([c for c in CHANGES if c != 0.0]); keep = [i for i, c in enumerate(CHANGES) if c != 0.0]
    design = np.stack([np.ones_like(x), x], 1)
    slopes = np.stack([[np.linalg.lstsq(design, delta[p, keep, j], rcond=None)[0][1] for j in range(delta.shape[2])]
                       for p in range(delta.shape[0])])
    return delta, slopes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scan-dir", required=True); p.add_argument("--out-dir", default=".")
    p.add_argument("--cnn-checkpoint", required=True); p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--device", default="auto"); p.add_argument("--recompute", action="store_true")
    p.add_argument("--weak-threshold", type=float, default=0.1,
                   help="dimensions whose largest |Delta<l_j>| over the alpha_s scan is below this are left out of the "
                        "alpha_s mean in Fig. 6 (the paper does this for the image-level VAE)")
    a = p.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "scan_means.npz"
    if cache.exists() and not a.recompute:
        data = dict(np.load(cache))
    else:
        data = compute_means(a); np.savez(cache, **data)
    set_paper_style()

    models = [("cnn", "Image-level VAE"), ("transformer", "Particle-level VAE")]
    res = {}
    for m, _ in models:
        act = data[f"active_{m}"]; idx = np.flatnonzero(act)
        delta, slopes = responses(data[f"mean_lnvar_{m}"], act)
        weak = [int(idx[j]) for j in range(len(idx)) if np.abs(delta[0][:, j]).max() < a.weak_threshold]
        res[m] = dict(idx=idx, delta=delta, slopes=slopes, weak=weak)
        strongest = [PARAMS[int(np.abs(slopes[:, j]).argmax())][0] for j in range(len(idx))]
        print(f"{m}: {len(idx)} active; alpha_s has the largest |slope| in {strongest.count('alphaS')}/{len(idx)} dims; "
              f"weak alpha_s dims (max|d|<{a.weak_threshold}): {weak}")

    xs = np.array(CHANGES) * 100
    # ---------------------------------------------------------------- Fig. 5: first active dimension
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (m, title) in zip(axes, models):
        d = res[m]["delta"]
        for ip, (_, label, col) in enumerate(PARAMS):
            ax.plot(xs, d[ip, :, 0], "-o", color=col, ms=2.5, lw=1.6 if ip == 0 else 1.1, label=label)
        ax.plot([0], [0], "o", color="k", ms=4, zorder=5)
        ax.set_title(f"({'ab'[models.index((m, title))]}) {title}", loc="left", fontsize=13)
        ax.set_xticks([-20, -10, 0, 10, 20]); ax.grid(alpha=0.15); ax.tick_params(direction="in", top=True, right=True)
    j0 = int(res["cnn"]["idx"][0]); axes[0].set_ylabel(rf"$\Delta\langle\ln\sigma_{{{j0}}}^2\rangle$", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.08))
    fig.supxlabel("Parameter relative change (%)", fontsize=13, y=-0.04)
    fig.savefig(out / "fig5_alpha_response.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------------- Fig. 6: all active dimensions + mean slopes
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), gridspec_kw=dict(width_ratios=[1.25, 1], hspace=0.32, wspace=0.28))
    for row, (m, title) in enumerate(models):
        r = res[m]; ax, bx = axes[row]
        for ip, (_, label, col) in enumerate(PARAMS):
            for j in range(r["delta"].shape[2]):
                ax.plot(xs, r["delta"][ip, :, j], "-o", color=col, ms=2, lw=0.8, alpha=0.3)
        ax.plot([0], [0], "o", color="k", ms=4, zorder=5)
        ax.set_title(f"({'ac'[row]}) {title}", loc="left", fontsize=12); ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_ylabel(r"$\Delta\langle\ln\sigma_j^2\rangle$", fontsize=12); ax.set_ylim(-0.3, 0.3)
        ax.grid(alpha=0.15); ax.tick_params(direction="in", top=True, right=True)
        for ip, (_, label, col) in enumerate(PARAMS):
            s = r["slopes"][ip]
            if ip == 0 and r["weak"]:
                s = np.delete(s, [int(np.flatnonzero(r["idx"] == w)[0]) for w in r["weak"]])
            bx.errorbar(s.mean(), ip, xerr=s.std(), fmt="o", ms=4, color=col, capsize=3, lw=1.3)
            print(f"  {m:12s} {label:16s} mean slope {s.mean():+.3f}  std {s.std():.3f}  (n={len(s)})")
        bx.set_yticks(range(len(PARAMS))); bx.set_yticklabels([p[1] for p in PARAMS], fontsize=11); bx.invert_yaxis()
        bx.axvline(0, color="0.6", lw=0.8); bx.grid(axis="x", alpha=0.15)
        bx.set_title(f"({'bd'[row]}) Mean response", loc="left", fontsize=12); bx.tick_params(direction="in")
    axes[1][0].set_xlabel("Parameter relative change (%)", fontsize=12)
    axes[1][1].set_xlabel(r"Mean fitted coefficient $\bar{s}_p$", fontsize=12)
    h = [plt.Line2D([], [], color=c, marker="o", ms=3, label=lab) for _, lab, c in PARAMS]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out / "fig6_parameter_responses.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved", out / "fig5_alpha_response.pdf", "and", out / "fig6_parameter_responses.pdf")


if __name__ == "__main__":
    main()
