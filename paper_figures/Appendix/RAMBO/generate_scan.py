"""Generate and encode the fixed-N RAMBO scan for current Fig. 9."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import load_encoder, encode_p4, project, read_json, write_json, summarize_distribution
from rambo import generate_massless, total_from_mass_pt, validate_events
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--basis", type=Path, default=HERE / "data/reference_basis.json")
    p.add_argument("--qcd-data", type=Path, default=HERE / "data/fig9_distributions.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    basis, qcd = read_json(a.basis), read_json(a.qcd_data)
    model, device = load_encoder(a.checkpoint, a.device)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    target = total_from_mass_pt(20, 200)
    all_mu, all_lv, all_n = [], [], []
    for n in range(2, 81):
        p4 = generate_massless(1000, n, target, rng=np.random.default_rng(np.random.SeedSequence([20261115, n])))
        if validate_events(p4, target)["status"] != "PASS":
            raise RuntimeError(f"RAMBO kinematics failed at N={n}")
        mu, lv = encode_p4(model, p4, device)
        all_mu.append(mu); all_lv.append(lv); all_n.append(np.full(1000, n, dtype=np.int16))
        print(f"Encoded N={n}: 1000 events", flush=True)
    mu, lv, n = map(np.concatenate, (all_mu, all_lv, all_n))
    u = project(lv, basis)
    np.savez_compressed(a.out_dir / "scan_posteriors.npz", mu=mu, logvar=lv, n_particles=n, u=u)
    samples = {name: qcd["samples"][name] for name in ("u", "g")}
    samples["rambo"] = summarize_distribution(n, u)
    write_json(a.out_dir / "fig9_distributions.json", dict(samples=samples))
    write_json(a.out_dir / "settings.json", dict(seed=20261115, seed_sequence="[seed,N]", events_per_N=1000,
               mass_GeV=20, pt_GeV=200, rapidity=0, phi=0, basis=basis, device=device))


if __name__ == "__main__":
    main()
