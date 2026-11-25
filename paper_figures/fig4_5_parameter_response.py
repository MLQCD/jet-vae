"""Current Figs. 4 and 5: dijet posterior-width responses to the eight Pythia inputs.

Fig. 4 uses original latent index j=1 in both models. Fig. 5 shows all active
coordinates; only the CNN alpha_s mean/SD excludes weak-response coordinates
(maximum absolute shift < 0.1). The default checkpoints give 29/32 active CNN/
Transformer dimensions and 25 CNN coordinates in the alpha_s summary.

Each setting has 10000 u and 10000 gluon jets, with equal class weights. Fits use
free-intercept GLS on the four nonzero variations, including shared-nominal
covariance. Displayed error bars are population SD across coordinate slopes,
not the statistical uncertainty of their mean. pTmin labels the generator input,
not the effective shower cutoff.

    python paper_figures/fig4_5_parameter_response.py --scan-dir data/pythia_scan/200GeV_process \
        --cnn-checkpoint cnn_beta2e-6.pth --transformer-checkpoint transformer.pth --out-dir figs/

The scan layout remains {u_jet,g_jet}/nominal.npz and
{u_jet,g_jet}/<parameter>/{minus_20pct,minus_10pct,plus_10pct,plus_20pct}.npz.
Cached means and variances are tied to the checkpoint content and input files.
Use --recompute after changing inputs or when replacing a legacy cache.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from common import (active_dimensions, cnn_input_image, encode_cnn, encode_transformer, load_cnn,
                    load_jets, load_transformer, require_image_events, resolve_device,
                    set_paper_style, validate_paper_cnn)

PARAMS = [
    ("alphaS", r"$\alpha_s$", "#2184a1"), ("aLund", r"$a_{\rm Lund}$", "#eb7e6e"),
    ("bLund", r"$b_{\rm Lund}$", (0.1875, 0.54687, 0.347656)),
    ("sigma", r"$\sigma_{\rm string}$", "#b47e8b"), ("pTmin", r"$p_{T,\min}$", "#ffb838"),
    ("mesonUDvector", r"$r_{V/P}$", "#7b3294"), ("probStoUD", r"$r_{s/(u,d)}$", "#ff0000"),
    ("probQQtoQ", r"$r_{qq/q}$", (0.36, 0.39, 0.42)),
]
MODELS = [("cnn", "Image-level VAE"), ("transformer", "Particle-level VAE")]
CHANGES = [-0.2, -0.1, 0.0, 0.1, 0.2]
FILES = {-0.2: "minus_20pct", -0.1: "minus_10pct", 0.1: "plus_10pct", 0.2: "plus_20pct"}
CACHE_VERSION = 2


def setting_paths(root, param, change):
    name = "nominal.npz" if change == 0 else f"{param}/{FILES[change]}.npz"
    return [Path(root) / cls / name for cls in ("u_jet", "g_jet")]


def cache_identity(a):
    def digest(path):
        with open(path, "rb") as handle:
            h = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
            return h.hexdigest()
    inputs = {}
    for param, _, _ in PARAMS:
        for change in CHANGES:
            for path in setting_paths(a.scan_dir, param, change):
                stat = path.stat()
                inputs[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns]
    return json.dumps(dict(version=CACHE_VERSION, inputs=inputs,
                           cnn_sha256=digest(a.cnn_checkpoint),
                           transformer_sha256=digest(a.transformer_checkpoint)), sort_keys=True)


def load_setting(root, param, change):
    """Return the two classes separately, preserving fixed class weights in the MC variance."""
    jets = []
    for path in setting_paths(root, param, change):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
        sample = load_jets(path); require_image_events(sample)
        if meta["jets"] != 10000 or len(sample["particles"]) != 10000:
            raise ValueError(f"{path}: expected 10000 jets per class")
        if change and not np.isclose(meta["relative_change"], change, atol=1e-9, rtol=0):
            raise ValueError(f"{path}: metadata does not match the requested variation")
        jets.append(sample)
    return jets


def pooled_moments(class_logvars):
    """Mean and MC variance of an equally weighted, fixed-composition u/g mixture."""
    if len(class_logvars) != 2 or any(len(v) < 2 for v in class_logvars):
        raise ValueError("Two classes with at least two jets each are required")
    arrays = [np.asarray(v, dtype=float) for v in class_logvars]
    mean = sum(v.mean(0) for v in arrays) / 2
    variance = sum(v.var(0, ddof=1) / len(v) for v in arrays) / 4
    return mean, variance


def compute_means(a):
    device = resolve_device(a.device)
    cnn, ck = load_cnn(a.cnn_checkpoint); validate_paper_cnn(ck)
    tr, _ = load_transformer(a.transformer_checkpoint, device)
    print("device:", device, flush=True)

    def encode_setting(param, change):
        result = {name: [] for name, _ in MODELS}
        for jets in load_setting(a.scan_dir, param, change):
            images = np.stack([cnn_input_image(e) for e in jets["events"]])
            result["cnn"].append(encode_cnn(cnn, images, batch=a.batch_size, device=device))
            result["transformer"].append(encode_transformer(tr, jets["particles"], jets["mask"],
                                                            batch=a.batch_size, device=device))
        return result

    nominal = encode_setting("alphaS", 0)
    data = {}
    for name, _ in MODELS:
        mu, lv = [np.concatenate([v[i] for v in nominal[name]]) for i in (0, 1)]
        data[f"active_{name}"] = active_dimensions(mu, lv)
        shape = (len(PARAMS), len(CHANGES), lv.shape[1])
        data[f"mean_lnvar_{name}"] = np.zeros(shape)
        data[f"variance_mean_{name}"] = np.zeros(shape)
    for ip, (param, _, _) in enumerate(PARAMS):
        for ic, change in enumerate(CHANGES):
            encoded = nominal if change == 0 else encode_setting(param, change)
            for name, _ in MODELS:
                mean, var = pooled_moments([v[1] for v in encoded[name]])
                data[f"mean_lnvar_{name}"][ip, ic] = mean
                data[f"variance_mean_{name}"][ip, ic] = var
            print(f"{param:14s} {change:+.1f} done", flush=True)
    return data


def responses(means, active, variance_mean):
    """Five shifts and free-intercept GLS slopes, using the shared nominal covariance."""
    means, variance_mean = np.asarray(means, dtype=float), np.asarray(variance_mean, dtype=float)
    if means.shape != variance_mean.shape or means.ndim != 3 or means.shape[:2] != (8, 5):
        raise ValueError("Expected means and MC variances with shape (8, 5, latent_dim)")
    if not np.isfinite(means).all() or not np.isfinite(variance_mean).all() or (variance_mean < 0).any():
        raise ValueError("Means and MC variances must be finite, with nonnegative variances")
    delta = (means - means[:, [2]])[:, :, active]
    var = variance_mean[:, :, active]
    nonzero = [0, 1, 3, 4]
    x = np.array(CHANGES)[nonzero]
    design = np.column_stack([np.ones(4), x])
    slopes = np.empty((len(PARAMS), delta.shape[2]))
    for ip in range(len(PARAMS)):
        for j in range(delta.shape[2]):
            covariance = np.diag(var[ip, nonzero, j]) + var[ip, 2, j]
            try:
                chol = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Singular MC covariance; check the per-setting posterior arrays") from exc
            slopes[ip, j] = np.linalg.lstsq(np.linalg.solve(chol, design),
                                           np.linalg.solve(chol, delta[ip, nonzero, j]), rcond=None)[0][1]
    return delta, slopes


def summarize(data, weak_threshold=0.1):
    result = {}
    for name, _ in MODELS:
        idx = np.flatnonzero(data[f"active_{name}"])
        if not len(idx):
            raise ValueError(f"{name}: no active dimensions")
        delta, slopes = responses(data[f"mean_lnvar_{name}"], data[f"active_{name}"],
                                  data[f"variance_mean_{name}"])
        weak = np.abs(delta[0]).max(axis=0) < weak_threshold
        selected = np.ones(slopes.shape, dtype=bool)
        if name == "cnn":
            selected[0] = ~weak
        if not selected.any(axis=1).all():
            raise ValueError("No dimensions remain in a response summary")
        summaries = [(s[keep].mean(), s[keep].std(ddof=0), int(keep.sum()))
                     for s, keep in zip(slopes, selected)]
        result[name] = dict(idx=idx, delta=delta, slopes=slopes, summary=summaries)
        print(f"{name}: {len(idx)} active; CNN alpha_s exclusions: "
              f"{idx[weak].tolist() if name == 'cnn' else []}")
        for (param, _, _), (mean, sd, count) in zip(PARAMS, summaries):
            print(f"  {param:14s} mean slope {mean:+.6f}; coordinate SD {sd:.6f}; n={count}")
    return result


def add_header(fig, axes, handles):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first = axes.flat[0]._left_title
    prefix = renderer.get_text_width_height_descent("(a) ", first.get_fontproperties(), False)[0]
    x = first.get_window_extent(renderer).x0 + prefix
    top = max(ax._left_title.get_window_extent(renderer).y1 for ax in (axes if axes.ndim == 1 else axes[0]))
    y = top + 0.10 * fig.dpi
    label = fig.text(x / fig.bbox.width, y / fig.bbox.height, "Dijet samples", va="bottom", fontsize=11)
    fig.canvas.draw()
    box = label.get_window_extent(fig.canvas.get_renderer())
    fig.legend(handles=handles, loc="center left", ncol=4, frameon=False, borderaxespad=0,
               bbox_to_anchor=((box.x1 + 0.22 * fig.dpi) / fig.bbox.width,
                               (box.y0 + box.height / 2) / fig.bbox.height),
               columnspacing=1.3, handlelength=2.0, fontsize=9)


def plot_responses(result, out, latent_index=1):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9.3,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    xs = np.array(CHANGES) * 100
    handles = [Line2D([], [], color=c, lw=1.35 if i == 0 else 1.05, marker="o", ms=2.6,
                      mew=0, label=label) for i, (_, label, c) in enumerate(PARAMS)]

    def format_curve(ax, limit):
        ax.axhline(0, color=".65", lw=.55, zorder=0)
        ax.plot([0], [0], "o", color="black", ms=2.8, mew=0, zorder=5)
        ax.set_xlim(-21, 21); ax.set_xticks(xs); ax.set_ylim(-limit, limit)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.21, top=.73, wspace=.17)
    curves = {}
    for name, _ in MODELS:
        matches = np.flatnonzero(result[name]["idx"] == latent_index)
        if len(matches) != 1:
            raise ValueError(f"{name}: requested original index {latent_index} is not active")
        curves[name] = result[name]["delta"][:, :, matches[0]]
    limit = max(float(np.abs(d).max()) for d in curves.values()) * 1.15 or .01
    for k, (ax, (name, title)) in enumerate(zip(axes, MODELS)):
        for ip, (_, label, color) in enumerate(PARAMS):
            ax.plot(xs, curves[name][ip], color=color, lw=1.35 if ip == 0 else 1.05,
                    marker="o", ms=2.6, mew=0, markevery=[0, 1, 3, 4])
        format_curve(ax, limit); ax.set_title(f"({'ab'[k]}) {title}", loc="left", pad=7)
    axes[0].set_ylabel(rf"$\Delta\langle\ln\sigma_{{{latent_index}}}^2\rangle$", labelpad=4)
    fig.text(.55, .035, "Parameter relative change (%)", ha="center", fontsize=10)
    add_header(fig, axes, handles)
    fig.savefig(out / "fig4_alpha_response.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), gridspec_kw={"width_ratios": [1.12, 1.]})
    fig.subplots_adjust(left=.095, right=.985, bottom=.115, top=.81, wspace=.37, hspace=.39)
    for row, (name, title) in enumerate(MODELS):
        r = result[name]; left, right = axes[row]
        for ip, (_, label, color) in enumerate(PARAMS):
            for j in range(len(r["idx"])):
                left.plot(xs, r["delta"][ip, :, j], color=color, lw=.55, alpha=.27,
                          marker="o", ms=2.6, mew=0, markevery=[0, 1, 3, 4])
            mean, sd, _ = r["summary"][ip]
            right.errorbar(mean, ip, xerr=sd, fmt="none", color=color, elinewidth=.8, capsize=2, capthick=.8)
            right.plot(mean, ip, "o", ms=2.6, mew=0, color=color)
        format_curve(left, float(np.abs(r["delta"]).max()) * 1.12 or .01)
        left.set_ylabel(r"$\Delta\langle\ln\sigma_j^2\rangle$", labelpad=4)
        left.set_title(f"({'ac'[row]}) {title}", loc="left", pad=7)
        right.set_title(f"({'bd'[row]}) Average response", loc="left", pad=7)
        right.set_yticks(range(8), [label for _, label, _ in PARAMS]); right.set_ylim(7.7, -.7)
        low = min([-1.6 if name == "cnn" else -1.4] + [v[0] - v[1] - .05 for v in r["summary"]])
        high = max([.2] + [v[0] + v[1] + .05 for v in r["summary"]])
        right.set_xlim(low, high); right.axvline(0, color=".65", lw=.55, zorder=0)
        right.tick_params(which="both", direction="in", top=True, right=True)
        right.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1, 0].set_xlabel("Parameter relative change (%)")
    axes[1, 1].set_xlabel(r"Average fitted slope $\bar{s}_p$")
    add_header(fig, axes, handles)
    fig.savefig(out / "fig5_parameter_responses.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved", out / "fig4_alpha_response.pdf", "and", out / "fig5_parameter_responses.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-dir", required=True); p.add_argument("--out-dir", default=".")
    p.add_argument("--cnn-checkpoint", required=True); p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--device", default="auto"); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--latent-index", type=int, default=1, help="original latent index in Fig. 4")
    p.add_argument("--weak-threshold", type=float, default=.1, help="CNN alpha_s summary exclusion threshold")
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.weak_threshold < 0:
        p.error("batch size must be positive and weak threshold nonnegative")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "scan_means.npz"
    identity = cache_identity(a)
    if cache.exists() and not a.recompute:
        with np.load(cache, allow_pickle=False) as z:
            if "identity" not in z or str(z["identity"]) != identity:
                p.error("This scan cache belongs to older code, inputs or checkpoints. Use --recompute.")
            data = {k: z[k] for k in z.files if k != "identity"}
    else:
        data = compute_means(a)
        np.savez(cache, identity=np.array(identity), **data)
    plot_responses(summarize(data, a.weak_threshold), out, a.latent_index)


if __name__ == "__main__":
    main()
