import numpy as np
from collections import Counter


def load_events_from_dat(path):
    events = []
    current_particles = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('<event_begin>'):
                current_particles = []
                continue
            if line.startswith('<event_end>'):
                if current_particles:
                    events.append(np.array(current_particles, dtype=np.float32))
                current_particles = []
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected line format in {path}: {line}")
            pid = int(parts[0])
            px, py, pz, E = map(float, parts[1:])
            current_particles.append([pid, px, py, pz, E])
    return events


def build_pid_channel_map(X_events):
    pid_counter = Counter()
    for ev in X_events:
        pid_counter.update(ev[:, 0].astype(int).tolist())
    return {pid: idx for idx, pid in enumerate(sorted(pid_counter.keys()))}


def build_pid_lookup(pid_channel_map):
    pids = np.array(sorted(pid_channel_map.keys()), dtype=np.int64)
    pid_min = int(pids.min())
    pid_max = int(pids.max())
    lookup = np.full(pid_max - pid_min + 1, -1, dtype=np.int32)
    for pid, ch in pid_channel_map.items():
        lookup[pid - pid_min] = ch
    return lookup, pid_min


def event_to_jet_image(ev, pid_lookup, pid_min, n_channels,
                       eta_bins=64, phi_bins=64,
                       eta_max=0.5, phi_max=0.5):
    pid = ev[:, 0].astype(np.int64)
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

    pid_shifted = pid - pid_min
    in_range = (pid_shifted >= 0) & (pid_shifted < len(pid_lookup))
    ch_arr = np.where(
        in_range,
        pid_lookup[np.clip(pid_shifted, 0, len(pid_lookup) - 1)],
        -1,
    )

    valid = (
        (ch_arr >= 0) &
        (deta > -eta_max) & (deta < eta_max) &
        (dphi > -phi_max) & (dphi < phi_max)
    )

    ch_v = ch_arr[valid]
    deta_v = deta[valid]
    dphi_v = dphi[valid]
    pT_v = pT[valid]

    w_bins = np.clip(
        ((deta_v + eta_max) / (2.0 * eta_max) * eta_bins).astype(np.int32),
        0, eta_bins - 1,
    )
    h_bins = np.clip(
        ((dphi_v + phi_max) / (2.0 * phi_max) * phi_bins).astype(np.int32),
        0, phi_bins - 1,
    )

    flat_idx = (ch_v * phi_bins + h_bins) * eta_bins + w_bins
    image = np.bincount(
        flat_idx,
        weights=pT_v,
        minlength=n_channels * phi_bins * eta_bins,
    ).reshape(n_channels, phi_bins, eta_bins).astype(np.float32)
    return image


def event_to_jet_image_no_pid(ev, eta_bins=64, phi_bins=64,
                              eta_max=0.5, phi_max=0.5):
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
        0, eta_bins - 1,
    )
    h_bins = np.clip(
        ((dphi_v + phi_max) / (2.0 * phi_max) * phi_bins).astype(np.int32),
        0, phi_bins - 1,
    )

    flat_idx = h_bins * eta_bins + w_bins
    image = np.bincount(
        flat_idx,
        weights=pT_v,
        minlength=phi_bins * eta_bins,
    ).reshape(1, phi_bins, eta_bins).astype(np.float32)
    return image


def pt_normalize(image):
    total = image.sum()
    if total > 0.0:
        image = image / total
    return image
