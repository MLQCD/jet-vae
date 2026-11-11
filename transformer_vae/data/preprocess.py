"""Build the training cache from raw ``.dat`` files.

    python -m transformer_vae.data.preprocess --u-jet u.dat --g-jet g.dat --cache-dir cache/

Writes ``{train,val}_{particles,mask,category,labels}.npy`` and ``meta.json``:

    particles  float32 [N, K, 3]  (z, eta, phi), pT-descending, zero padded
    mask       bool    [N, K]     True for real constituents
    category   int8    [N, K]     pid category 0-3 (valid where mask is True; 0 in padding)
    labels     int8    [N]        0 = quark (u) jet, 1 = gluon jet

The split is stratified by label with ``--random-state`` (default 42), and is
fixed by the cache: training ``--seed`` does not change it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dat_io import load_events_from_dat
from .pointset import CATEGORY_NAMES, event_to_z_eta_phi

__all__ = ["build_cache", "load_labeled_events"]


def load_labeled_events(u_jet_path=None, g_jet_path=None):
    """Quark jets first (label 0), then gluon jets (label 1)."""
    events, labels = [], []
    for path, label in ((u_jet_path, 0), (g_jet_path, 1)):
        if path:
            ev = load_events_from_dat(path)
            events.extend(ev)
            labels.extend([label] * len(ev))
    if not events:
        raise ValueError("Provide at least one of u_jet_path / g_jet_path")
    return events, np.asarray(labels, dtype=np.int64)


def _arrays(events, max_particles):
    n = len(events)
    particles = np.zeros((n, max_particles, 3), dtype=np.float32)
    mask = np.zeros((n, max_particles), dtype=bool)
    category = np.zeros((n, max_particles), dtype=np.int8)
    for i, event in enumerate(events):
        shape, cat = event_to_z_eta_phi(event, max_particles=max_particles)
        k = len(shape)
        particles[i, :k] = shape
        mask[i, :k] = True
        category[i, :k] = cat
    return particles, mask, category


def build_cache(events, labels, cache_dir, max_particles=80, val_fraction=0.15,
                random_state=42, sources=None, angular_scale=0.4):
    """Split, convert and save. Returns the metadata dict."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, val_idx = train_test_split(
        np.arange(len(events)), test_size=val_fraction,
        random_state=random_state, stratify=stratify)

    for split, idx in (("train", train_idx), ("val", val_idx)):
        particles, mask, category = _arrays([events[i] for i in idx], max_particles)
        np.save(cache_dir / f"{split}_particles.npy", particles)
        np.save(cache_dir / f"{split}_mask.npy", mask)
        np.save(cache_dir / f"{split}_category.npy", category)
        np.save(cache_dir / f"{split}_labels.npy", labels[idx].astype(np.int8))

    meta = {
        "format": "z_eta_phi_shape_v2",
        "columns": ["z", "eta", "phi"],
        "sort_key": "pt_desc",
        "z_definition": "pt_i / sum(pt over kept constituents), per jet",
        "axis_definition": ("massless vector-sum axis of (px, py, pz) over kept constituents; "
                            "eta/phi shifted so this axis sits at (0, 0). Source E is unused."),
        "category_names": CATEGORY_NAMES,
        "labels": {"0": "quark (u) jet", "1": "gluon jet"},
        "recommended_angular_scale": float(angular_scale),
        "max_particles": int(max_particles),
        "train_events": int(len(train_idx)),
        "val_events": int(len(val_idx)),
        "val_fraction": float(val_fraction),
        "random_state": int(random_state),
        "sources": sources or {},
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--u-jet", default=None, help=".dat file of quark jets (label 0)")
    p.add_argument("--g-jet", default=None, help=".dat file of gluon jets (label 1)")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-particles", type=int, default=80)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--angular-scale", type=float, default=0.4)
    a = p.parse_args(argv)
    events, labels = load_labeled_events(a.u_jet, a.g_jet)
    meta = build_cache(events, labels, a.cache_dir, a.max_particles, a.val_fraction,
                       a.random_state, {"u_jet": a.u_jet or "", "g_jet": a.g_jet or ""},
                       a.angular_scale)
    print(f"Saved cache to {a.cache_dir}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
