#!/usr/bin/env python
# coding: utf-8
"""
Preprocess jet .dat files into single-channel eta/phi jet images.

This cache format is compatible with train_betavae.py --cache-dir, but it
does not encode PID/channel information.  Every valid particle contributes
only to one shared pT image:

  image[0, phi_bin, eta_bin] += particle pT

Images are saved after event-level pT normalization, so each image sums to 1.
"""

import argparse
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_utils import add_config_argument, parse_args_with_config


def load_events_from_dat(path):
    events = []
    current_particles = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("<event_begin>"):
                current_particles = []
                continue
            if line.startswith("<event_end>"):
                if current_particles:
                    events.append(np.array(current_particles, dtype=np.float32))
                current_particles = []
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected line format in {path}: {line}")
            pid = int(parts[0])
            px, py, pz, energy = map(float, parts[1:])
            current_particles.append([pid, px, py, pz, energy])
    return events


def load_dataset(g_jet_path=None, u_jet_path=None):
    X_events, y_labels = [], []
    if u_jet_path is not None:
        for ev in load_events_from_dat(u_jet_path):
            X_events.append(ev)
            y_labels.append(0)
    if g_jet_path is not None:
        for ev in load_events_from_dat(g_jet_path):
            X_events.append(ev)
            y_labels.append(1)
    if not X_events:
        raise ValueError("Provide at least one of --g-jet or --u-jet")
    return X_events, np.array(y_labels, dtype=np.int64)


def event_to_jet_image(ev, eta_bins, phi_bins, eta_max, phi_max):
    """Convert one event array (N_particles, 5) to a single-channel image."""
    px = ev[:, 1]
    py = ev[:, 2]
    pz = ev[:, 3]

    p = np.sqrt(px**2 + py**2 + pz**2) + 1e-8
    pT = np.sqrt(px**2 + py**2)
    eta = 0.5 * np.log((p + pz + 1e-8) / (p - pz + 1e-8))
    phi = np.arctan2(py, px)

    px_jet = px.sum()
    py_jet = py.sum()
    pz_jet = pz.sum()
    p_jet = np.sqrt(px_jet**2 + py_jet**2 + pz_jet**2) + 1e-8
    eta_jet = 0.5 * np.log((p_jet + pz_jet + 1e-8) / (p_jet - pz_jet + 1e-8))
    phi_jet = np.arctan2(py_jet, px_jet)

    deta = eta - eta_jet
    dphi = (phi - phi_jet + np.pi) % (2.0 * np.pi) - np.pi

    valid = (
        (deta > -eta_max) & (deta < eta_max) &
        (dphi > -phi_max) & (dphi < phi_max)
    )
    deta_v = deta[valid]
    dphi_v = dphi[valid]
    pT_v = pT[valid]

    w_bins = np.clip(
        ((deta_v + eta_max) / (2.0 * eta_max) * eta_bins).astype(np.int32),
        0,
        eta_bins - 1,
    )
    h_bins = np.clip(
        ((dphi_v + phi_max) / (2.0 * phi_max) * phi_bins).astype(np.int32),
        0,
        phi_bins - 1,
    )

    flat_idx = h_bins * eta_bins + w_bins
    image_2d = np.bincount(
        flat_idx,
        weights=pT_v,
        minlength=phi_bins * eta_bins,
    ).reshape(phi_bins, eta_bins).astype(np.float32)
    return image_2d[None, :, :]


def pt_normalize(image):
    total = image.sum()
    if total > 0.0:
        image = image / total
    return image


def events_to_images(X_events, eta_bins, phi_bins, eta_max, phi_max,
                     log_prefix="", out_path=None):
    N = len(X_events)
    shape = (N, 1, phi_bins, eta_bins)
    if out_path is not None:
        images = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
    else:
        images = np.zeros(shape, dtype=np.float16)

    for i, ev in enumerate(X_events):
        if (i + 1) % 5000 == 0 or (i + 1) == N:
            print(f"  {log_prefix}{i + 1}/{N}", flush=True)
        img = event_to_jet_image(ev, eta_bins, phi_bins, eta_max, phi_max)
        images[i] = pt_normalize(img)
    return images


def split_indices(y_labels, test_frac, val_frac):
    indices = np.arange(len(y_labels))
    stratify = y_labels if len(np.unique(y_labels)) > 1 else None
    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_frac,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"  Stratified test split unavailable ({exc}); using unstratified split.")
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_frac,
            random_state=42,
            stratify=None,
        )
    train_val_stratify = y_labels[train_val_idx] if stratify is not None else None
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac,
            random_state=42,
            stratify=train_val_stratify,
        )
    except ValueError as exc:
        print(f"  Stratified val split unavailable ({exc}); using unstratified split.")
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac,
            random_state=42,
            stratify=None,
        )
    return train_idx, val_idx, test_idx


def main():
    t0 = datetime.now()
    print(f"Single-channel preprocessing started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(
        description="Preprocess jet data into single-channel eta/phi cached images"
    )
    parser.add_argument("--g-jet", type=str, default=None)
    parser.add_argument("--u-jet", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache_jet_images_no_pid")
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Square image size (eta_bins = phi_bins = image_size)",
    )
    parser.add_argument("--eta-max", type=float, default=2.5)
    parser.add_argument("--phi-max", type=float, default=np.pi)
    parser.add_argument(
        "--sample-events",
        type=int,
        default=0,
        help="Max events per available class (0 = all)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
        help="Fraction of train+val used for validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.20,
        help="Fraction of all data used for test",
    )
    add_config_argument(parser)
    args = parse_args_with_config(parser)

    if args.g_jet is None and args.u_jet is None:
        parser.error("Provide at least one of --g-jet or --u-jet")

    os.makedirs(args.cache_dir, exist_ok=True)
    eta_bins = phi_bins = args.image_size

    print(f"\nLoading events from:\n  g-jet: {args.g_jet}\n  u-jet: {args.u_jet}")
    X_events, y_labels = load_dataset(args.g_jet, args.u_jet)
    print(f"Total events loaded: {len(X_events)}")

    if args.sample_events > 0:
        np.random.seed(42)
        selected = []
        for label in np.unique(y_labels):
            label_idx = np.where(y_labels == label)[0]
            n = min(args.sample_events, len(label_idx))
            selected.append(np.random.choice(label_idx, n, replace=False))
        sel = np.concatenate(selected)
        X_events = [X_events[i] for i in sel]
        y_labels = y_labels[sel]
        print(f"Sub-sampled to {len(X_events)} events across {len(selected)} class(es)")

    n_channels = 1
    print("\nPID information is ignored for image channels.")
    print(f"Image shape: ({n_channels}, {phi_bins}, {eta_bins})")

    train_idx, val_idx, test_idx = split_indices(
        y_labels,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    X_train = [X_events[i] for i in train_idx]
    X_val = [X_events[i] for i in val_idx]
    X_test = [X_events[i] for i in test_idx]
    y_train = y_labels[train_idx]
    y_val = y_labels[val_idx]
    y_test = y_labels[test_idx]
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    meta_path = os.path.join(args.cache_dir, "meta.npz")
    np.savez_compressed(
        meta_path,
        pid_lookup=np.array([0], dtype=np.int32),
        pid_min=np.int64(0),
        n_channels=np.int64(n_channels),
        eta_bins=np.int64(eta_bins),
        phi_bins=np.int64(phi_bins),
        eta_max=np.float32(args.eta_max),
        phi_max=np.float32(args.phi_max),
        split_sizes=np.array([len(X_train), len(X_val), len(X_test)], dtype=np.int64),
        normalization=np.array("event_pt"),
        channel_mode=np.array("eta_phi_total_pt_no_pid"),
        source_g_jet_path=np.array(args.g_jet if args.g_jet is not None else ""),
        source_u_jet_path=np.array(args.u_jet if args.u_jet is not None else ""),
    )
    print(f"Saved metadata -> {meta_path}")

    img_kwargs = dict(
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        eta_max=args.eta_max,
        phi_max=args.phi_max,
    )

    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        print(f"\nBuilding {name} images ({len(X)} events) ...")
        img_path = os.path.join(args.cache_dir, f"{name}_images.npy")
        lbl_path = os.path.join(args.cache_dir, f"{name}_labels.npy")
        imgs = events_to_images(X, log_prefix=f"  {name}: ",
                                out_path=img_path, **img_kwargs)
        np.save(lbl_path, y)
        size_mb = os.path.getsize(img_path) / 1e6
        print(f"  Saved {name}_images.npy  shape={imgs.shape}  ({size_mb:.1f} MB)")

    elapsed = datetime.now() - t0
    print(f"\nDone in {elapsed}. Cache directory: {args.cache_dir}/")
    print("Files:")
    for filename in sorted(os.listdir(args.cache_dir)):
        path = os.path.join(args.cache_dir, filename)
        if os.path.isfile(path):
            print(f"  {filename:30s}  {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
