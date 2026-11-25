"""Rebuild the shared QCD reference for Figs. 9 and 11 from raw reference jets."""
import argparse
import sys
from pathlib import Path
import numpy as np
from analysis import (HERE, CHECKPOINT_SHA256, OBSERVABLE_NAMES, digest, load_encoder,
                      observables, pearson, project, summarize_distribution, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u-jet", required=True)
    p.add_argument("--g-jet", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(HERE.parent))
    from common import balanced_inputs, encode_transformer, active_dimensions, sigma_scalar_coefficients
    model, device = load_encoder(a.checkpoint, a.device)
    jets = balanced_inputs(a.u_jet, a.g_jet, n_per_class=5000, seed=42)
    if jets["events"] is None:
        p.error("Raw constituents are required: use .dat or .npz with full p4/pid/mask.")
    mu, lv = encode_transformer(model, jets["particles"], jets["mask"], batch=256, device=device)
    active = np.flatnonzero(active_dimensions(mu, lv))
    if active.tolist() != list(range(32)):
        raise ValueError("The paper reference must have 32 active particle-level dimensions.")
    lv64 = lv.astype(float)
    b, _ = sigma_scalar_coefficients(lv64)
    basis = dict(a=lv64.mean(0).tolist(), b=b.tolist(), active_indices=active.tolist(),
                 checkpoint_sha256=CHECKPOINT_SHA256, n_per_class=5000, seed=42,
                 source_sha256={"u": digest(a.u_jet), "g": digest(a.g_jet)})
    u = project(lv, basis)
    values = np.array([observables(event) for event in jets["events"]])
    counts = values[:, 0].astype(int)
    samples = {name: summarize_distribution(counts[jets["labels"] == k], u[jets["labels"] == k])
               for k, name in enumerate(("u", "g"))}
    corr = pearson(mu, lv, values)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(a.out_dir / "reference_basis.json", basis)
    write_json(a.out_dir / "qcd_distributions.json", dict(samples=samples))
    write_json(a.out_dir / "fig11_correlations.json", dict(correlations=corr.tolist(),
               observable_names=OBSERVABLE_NAMES, latent_names=[f"mu_{j}" for j in range(32)] + ["logvar_0"],
               n_per_class=5000, seed=42, R_ref=.4, checkpoint_sha256=CHECKPOINT_SHA256))
    np.savez_compressed(a.out_dir / "reference.npz", mu=mu, logvar=lv, observables=values,
                        u=u, n_particles=counts, labels=jets["labels"], file_indices=jets["file_indices"])
    print(f"Saved QCD reference and all 297 Fig. 11 correlations to {a.out_dir}")


if __name__ == "__main__":
    main()
