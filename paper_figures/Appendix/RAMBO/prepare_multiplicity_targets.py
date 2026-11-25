"""Rebuild Fig. 10 target counts from the existing dijet alpha_s scan files."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    names = ["alphaS/minus_20pct.npz", "alphaS/minus_10pct.npz", "nominal.npz",
             "alphaS/plus_10pct.npz", "alphaS/plus_20pct.npz"]
    groups, sources = [], {}
    for percent, name in zip([-20,-10,0,10,20], names):
        pieces = []
        for flavor in ("u_jet", "g_jet"):
            path = a.scan_dir/flavor/name
            with np.load(path, allow_pickle=False) as z:
                if "mask" not in z:
                    raise ValueError("Need full constituent mask, not the truncated mask80.")
                n = np.asarray(z["mask"], dtype=bool).sum(1)
            if len(n) != 10000 or n.min() < 1 or n.max() > 80:
                raise ValueError("Expected the paper sample: 10000 jets per class, 1 <= N <= 80.")
            pieces.append(n)
            sources[f"{flavor}/{name}"] = digest(path)
        n = np.concatenate(pieces)
        original = np.bincount(n, minlength=81)
        selected = original.copy(); selected[1] = 0
        groups.append(dict(qcd_alphaS_percent=percent, original_counts=original.tolist(),
                           counts_after_N1_cut=selected.tolist()))
    write_json(a.out, dict(groups=groups, source_sha256=sources,
               policy="Remove N=1; match all remaining integer-N counts exactly. RAMBO has no alpha_s parameter."))


if __name__ == "__main__":
    main()
