"""Generate fresh RAMBO ensembles matching full QCD multiplicity histograms (Fig. 10)."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import load_encoder, encode_p4, read_json, write_json
from rambo import generate_massless, total_from_mass_pt, validate_events
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--targets", type=Path, default=HERE / "data/multiplicity_targets.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    targets = read_json(a.targets)
    model, device = load_encoder(a.checkpoint, a.device)
    target = total_from_mass_pt(20, 200)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    means, mean_n, sizes = [], [], []
    for group, row in enumerate(targets["groups"]):
        counts = np.asarray(row["counts_after_N1_cut"])
        if counts.shape != (81,) or np.any(counts[:2]) or np.any(counts < 0) or not np.equal(counts, counts.astype(int)).all() or not counts.sum():
            raise ValueError("Expected nonnegative integer counts for N=2..80.")
        all_lv, all_n = [], []
        for n in np.flatnonzero(counts):
            seed = np.random.SeedSequence([20261116, 3906, group, int(n)])
            p4 = generate_massless(int(counts[n]), int(n), target, rng=np.random.default_rng(seed))
            if validate_events(p4, target)["status"] != "PASS":
                raise RuntimeError("RAMBO conservation/mass-shell check failed")
            _, lv = encode_p4(model, p4, device)
            all_lv.append(lv); all_n.append(np.full(int(counts[n]), n, dtype=np.int16))
        lv, n = np.concatenate(all_lv), np.concatenate(all_n)
        np.testing.assert_array_equal(np.bincount(n, minlength=81), counts)
        means.append(lv.mean(0, dtype=float)); mean_n.append(float(n.mean())); sizes.append(len(n))
        np.savez_compressed(a.out_dir / f"posterior_group{group}.npz", logvar=lv, n_particles=n)
        print(f"Group {group+1}/5: {len(n)} events, <N>={n.mean():.8f}", flush=True)
    means = np.asarray(means)
    write_json(a.out_dir / "fig10_response.json", dict(mean_N=mean_n, mean_logvar=means.tolist(),
               delta_mean_logvar=(means-means[2]).tolist(), active_indices=list(range(32)), baseline_index=2,
               n_events=sizes, plot_ylim=[-.2983057705650325, .2983057705650325]))
    write_json(a.out_dir / "settings.json", dict(seed=20261116, stream_tag=3906,
               seed_sequence="[seed,3906,group_index,N]", mass_GeV=20, pt_GeV=200, targets=targets, device=device))


if __name__ == "__main__":
    main()
