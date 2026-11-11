#!/usr/bin/env python
# coding: utf-8
"""
Beta-VAE Jet Image Generation for Apple MPS or CPU

Trains a Beta-VAE on jet images constructed from .dat particle data files.
The jet image pipeline mirrors train_cnn_image_v2.py:
  - True 2D spatial images: Δη (W) × Δφ (H) binned grid
  - One channel per unique particle ID
  - Pixel value = sum of pT deposited in each (Δη, Δφ) cell
  - Event-level pT normalization so each image sums to 1

After training, samples are drawn from the learned latent prior and saved
as numpy arrays and matplotlib visualizations.
"""

import argparse
import os
import sys
import numpy as np
from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import the BetaVAE from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from betaVae import BetaVAE

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


def parse_hidden_dims(value):
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("--hidden-dims must contain at least one channel size")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("--hidden-dims values must be positive integers")
    return dims


def select_mps_or_cpu_device(preference="auto"):
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "mps":
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading  (identical to train_cnn_image_v2.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Jet-image construction  (identical to train_cnn_image_v2.py)
# ---------------------------------------------------------------------------

def build_pid_channel_map(X_events):
    pid_counter = Counter()
    for ev in X_events:
        pid_counter.update(ev[:, 0].astype(int).tolist())
    sorted_pids = sorted(pid_counter.keys())
    pid_map = {pid: idx for idx, pid in enumerate(sorted_pids)}
    total_particles = sum(pid_counter.values())

    print(f"\n{'='*55}")
    print(f"  PID -> Channel Mapping  ({len(sorted_pids)} channels)")
    print(f"{'='*55}")
    print(f"  {'Channel':>7}  {'PID':>6}  {'Count':>10}  {'Fraction':>9}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*10}  {'-'*9}")
    for pid, idx in pid_map.items():
        count = pid_counter[pid]
        print(f"  {idx:>7}  {pid:>6}  {count:>10,}  {count/total_particles:>9.4f}")
    print(f"  Total particles: {total_particles:,}")
    print(f"{'='*55}")
    return pid_map


def build_pid_lookup(pid_channel_map):
    pids    = np.array(sorted(pid_channel_map.keys()), dtype=np.int64)
    pid_min = int(pids.min())
    pid_max = int(pids.max())
    lookup  = np.full(pid_max - pid_min + 1, -1, dtype=np.int32)
    for pid, ch in pid_channel_map.items():
        lookup[pid - pid_min] = ch
    return lookup, pid_min


def event_to_jet_image(ev, pid_lookup, pid_min, n_channels,
                       eta_bins=64, phi_bins=64,
                       eta_max=2.5, phi_max=np.pi):
    """
    Convert one jet event to a (C, H, W) image.
      H = Δφ bins, W = Δη bins, channel = particle species
    Pixel value = sum of pT for particles in that cell.
    """
    pid = ev[:, 0].astype(np.int64)
    px  = ev[:, 1]
    py  = ev[:, 2]
    pz  = ev[:, 3]

    p   = np.sqrt(px**2 + py**2 + pz**2) + 1e-8
    pT  = np.sqrt(px**2 + py**2)
    eta = 0.5 * np.log((p + pz + 1e-8) / (p - pz + 1e-8))
    phi = np.arctan2(py, px)

    px_jet  = px.sum(); py_jet = py.sum(); pz_jet = pz.sum()
    p_jet   = np.sqrt(px_jet**2 + py_jet**2 + pz_jet**2) + 1e-8
    eta_jet = 0.5 * np.log((p_jet + pz_jet + 1e-8) / (p_jet - pz_jet + 1e-8))
    phi_jet = np.arctan2(py_jet, px_jet)

    deta = eta - eta_jet
    dphi = (phi - phi_jet + np.pi) % (2.0 * np.pi) - np.pi

    pid_shifted = pid - pid_min
    in_range    = (pid_shifted >= 0) & (pid_shifted < len(pid_lookup))
    ch_arr      = np.where(
        in_range,
        pid_lookup[np.clip(pid_shifted, 0, len(pid_lookup) - 1)],
        -1
    )

    valid = (
        (ch_arr >= 0) &
        (deta > -eta_max) & (deta < eta_max) &
        (dphi > -phi_max) & (dphi < phi_max)
    )

    ch_v   = ch_arr[valid]
    deta_v = deta[valid]
    dphi_v = dphi[valid]
    pT_v   = pT[valid]

    w_bins = np.clip(
        ((deta_v + eta_max) / (2.0 * eta_max) * eta_bins).astype(np.int32),
        0, eta_bins - 1
    )
    h_bins = np.clip(
        ((dphi_v + phi_max) / (2.0 * phi_max) * phi_bins).astype(np.int32),
        0, phi_bins - 1
    )

    flat_idx = (ch_v * phi_bins + h_bins) * eta_bins + w_bins
    image = np.bincount(
        flat_idx, weights=pT_v,
        minlength=n_channels * phi_bins * eta_bins
    ).reshape(n_channels, phi_bins, eta_bins).astype(np.float32)
    return image


def pt_normalize(image):
    total = image.sum()
    if total > 0.0:
        image = image / total
    return image


# ---------------------------------------------------------------------------
# Dataset - cached .npy images or on-the-fly event pixelization
# ---------------------------------------------------------------------------

class CachedJetImageDataset(Dataset):
    """Load pre-built jet images from .npy files created by preprocess_jet_images.py.
    Images are memory-mapped so the full array is never loaded into RAM at once."""

    def __init__(self, cache_dir, split, normalization="event_pt",
                 mean_img=None, std_img=None, r=1e-5):
        img_path = os.path.join(cache_dir, f"{split}_images.npy")
        lbl_path = os.path.join(cache_dir, f"{split}_labels.npy")
        self.images = np.load(img_path, mmap_mode='r')   # (N, C, H, W) float16
        self.labels = np.load(lbl_path)                  # (N,) int64
        self.normalization = normalization
        self.mean_img = mean_img
        self.std_img = std_img
        self.r = r

        if self.normalization == "standardized_event_pt":
            if self.mean_img is None or self.std_img is None:
                raise ValueError("Legacy standardized cache needs mean_img and std_img in meta.npz")
        elif self.normalization != "event_pt":
            raise ValueError(f"Unsupported cache normalization: {self.normalization}")

        print(f"  Loaded {split}: images={self.images.shape}, normalization={self.normalization}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy().astype(np.float32)
        if self.normalization == "standardized_event_pt":
            img = img * (self.std_img + self.r) + self.mean_img
            img = np.clip(img, 0.0, None)
            img = pt_normalize(img)
        return torch.from_numpy(img).float()


class JetImageDataset(Dataset):
    def __init__(self, X_events, pid_lookup, pid_min, n_channels,
                 eta_bins=32, phi_bins=32,
                 eta_max=0.5, phi_max=0.5):
        self.X_events   = X_events
        self.pid_lookup = pid_lookup
        self.pid_min    = pid_min
        self.n_channels = n_channels
        self.eta_bins   = eta_bins
        self.phi_bins   = phi_bins
        self.eta_max    = eta_max
        self.phi_max    = phi_max

    def __len__(self):
        return len(self.X_events)

    def __getitem__(self, idx):
        ev  = self.X_events[idx]
        img = event_to_jet_image(ev, self.pid_lookup, self.pid_min, self.n_channels,
                                 eta_bins=self.eta_bins, phi_bins=self.phi_bins,
                                 eta_max=self.eta_max,  phi_max=self.phi_max)
        img = pt_normalize(img)
        return torch.from_numpy(img)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, dataset_size, log_every=50, grad_clip=5.0):
    model.train()
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0
    n_batches = len(loader)
    configured_batch_size = loader.batch_size or dataset_size

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        bs = x.size(0)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        M_N    = bs / configured_batch_size
        losses = model.loss_function(*outputs, M_N=M_N)

        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_idx + 1}/{n_batches}: "
                f"loss={losses['loss'].item()} "
                f"recon={losses['Reconstruction_Loss'].item()} "
                f"kld={losses['KLD'].item()}"
            )

        losses['loss'].backward()
        if grad_clip and grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss  += losses['loss'].item()            * bs
        total_recon += losses['Reconstruction_Loss'].item() * bs
        total_kld   += losses['KLD'].item()             * bs

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            n_seen = (batch_idx + 1) * bs
            print(f"  [train {batch_idx+1}/{n_batches}] "
                  f"loss={total_loss/n_seen:.5f}  "
                  f"recon={total_recon/n_seen:.5f}  "
                  f"kld={total_kld/n_seen:.5f}", flush=True)

    N = dataset_size
    return total_loss / N, total_recon / N, total_kld / N


@torch.no_grad()
def eval_one_epoch(model, loader, device, dataset_size, split='val', log_every=50):
    model.eval()
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0
    n_batches = len(loader)
    configured_batch_size = loader.batch_size or dataset_size

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x = batch[0]
        else:
            x = batch
        x  = x.to(device)
        bs = x.size(0)
        M_N    = bs / configured_batch_size
        outputs = model(x)
        losses  = model.loss_function(*outputs, M_N=M_N)

        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(
                f"Non-finite {split} loss at batch {batch_idx + 1}/{n_batches}: "
                f"loss={losses['loss'].item()} "
                f"recon={losses['Reconstruction_Loss'].item()} "
                f"kld={losses['KLD'].item()}"
            )

        total_loss  += losses['loss'].item()            * bs
        total_recon += losses['Reconstruction_Loss'].item() * bs
        total_kld   += losses['KLD'].item()             * bs

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            n_seen = (batch_idx + 1) * bs
            print(f"  [{split}  {batch_idx+1}/{n_batches}] "
                  f"loss={total_loss/n_seen:.5f}  "
                  f"recon={total_recon/n_seen:.5f}  "
                  f"kld={total_kld/n_seen:.5f}", flush=True)

    N = dataset_size
    return total_loss / N, total_recon / N, total_kld / N


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_training_plots(history, output_prefix):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ['loss', 'recon', 'kld'],
        ['Total Loss', 'Reconstruction Loss', 'KLD']
    ):
        ax.plot(epochs, history[f'train_{key}'], label='Train')
        ax.plot(epochs, history[f'val_{key}'],   label='Val')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    path = f"{output_prefix}_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def save_generated_jets(model, n_samples, device,
                        pid_channel_map, output_prefix,
                        eta_bins, phi_bins, eta_max, phi_max):
    """
    Sample from the learned prior and save generated jet images.
    Pixel values are event-level pT fractions and each generated image sums to 1.
    """
    model.eval()
    with torch.no_grad():
        z        = torch.randn(n_samples, model.latent_dim).to(device)
        gen_imgs = model.decode(z).cpu().numpy()  # (N, C, H, W), non-negative and sum-normalized

    gen_imgs = np.clip(gen_imgs, 0.0, None)
    gen_sums = gen_imgs.reshape(gen_imgs.shape[0], -1).sum(axis=1)
    valid = gen_sums > 0.0
    gen_imgs[valid] = gen_imgs[valid] / gen_sums[valid, None, None, None]

    # Save raw arrays
    np_path = f"{output_prefix}_generated.npy"
    np.save(np_path, gen_imgs)
    print(f"Saved generated arrays to {np_path}  shape={gen_imgs.shape}")

    # Visualise: sum over channels to get total pT image
    pid_list = sorted(pid_channel_map.keys())
    n_show   = min(n_samples, 8)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))

    for j in range(n_show):
        total_pT = gen_imgs[j].sum(axis=0)          # (H, W)
        axes[0, j].imshow(total_pT, origin='lower', aspect='auto',
                          extent=[-eta_max, eta_max, -phi_max, phi_max])
        axes[0, j].set_title(f"Sample {j+1}")
        if j == 0:
            axes[0, j].set_xlabel(r'$\Delta\eta$')
            axes[0, j].set_ylabel(r'$\Delta\phi$')

        # Dominant channel
        dom_ch    = gen_imgs[j].sum(axis=(1, 2)).argmax()
        dom_pid   = pid_list[dom_ch] if dom_ch < len(pid_list) else '?'
        axes[1, j].imshow(gen_imgs[j, dom_ch], origin='lower', aspect='auto',
                          extent=[-eta_max, eta_max, -phi_max, phi_max])
        axes[1, j].set_title(f"PID={dom_pid}")

    axes[0, 0].set_ylabel(r'$\Delta\phi$ [rad]')
    axes[1, 0].set_ylabel(r'$\Delta\phi$ [rad]')
    fig.suptitle(r"Generated jet images (top: $\Sigma_c\,p_T$, bottom: dominant PID channel)")
    fig.tight_layout()
    img_path = f"{output_prefix}_generated_samples.png"
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualisation to {img_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = datetime.now()
    print(f"{'='*60}")
    print(f"  Beta-VAE Jet Generation")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    parser = argparse.ArgumentParser(description="Beta-VAE Jet Image Generation")
    # --- data source: either a cache dir OR raw .dat files ---
    parser.add_argument("--cache-dir",     type=str,   default=None,
                        help="Cache directory from preprocess_jet_images.py. "
                             "When provided, --g-jet / --u-jet are not needed.")
    parser.add_argument("--g-jet",         type=str,   default=None)
    parser.add_argument("--u-jet",         type=str,   default=None)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--device",        type=str,   default="auto",
                        choices=["auto", "mps", "cpu"],
                        help="Training device. auto prefers MPS when available.")
    parser.add_argument("--grad-clip",     type=float, default=5.0,
                        help="Max gradient norm; use 0 to disable clipping.")
    parser.add_argument("--sample-events", type=int,   default=0,
                        help="Max events per available class to use (0 = all)")
    parser.add_argument("--patience",      type=int,   default=20)
    parser.add_argument("--img-size",      type=int,   default=64,
                        help="Spatial resolution of jet images (img_size x img_size). "
                             "Must be divisible by 2^len(hidden_dims).")
    parser.add_argument("--eta-max",       type=float, default=0.5,
                        help="Half-width of the Delta eta image window for raw .dat mode")
    parser.add_argument("--phi-max",       type=float, default=0.5,
                        help="Half-width of the Delta phi image window for raw .dat mode")
    parser.add_argument("--hidden-dims",   type=parse_hidden_dims,
                        default=[32, 64, 128, 256, 512],
                        help="Comma-separated CNN channel sizes, e.g. 32,64,128,256,512. "
                        "The number of values is the number of stride-2 conv layers.")
    parser.add_argument("--latent-dim",    type=int,   default=128)
    parser.add_argument("--beta",          type=float,   default=1.0)
    parser.add_argument("--beta-start",    type=float, default=0.0,
                        help="Initial beta for --loss-type A beta annealing")
    parser.add_argument("--beta-anneal-iter", type=int, default=int(1e5),
                        help="Training iterations used to anneal beta_start -> beta for --loss-type A")
    parser.add_argument("--beta-anneal-schedule", type=str, default='linear',
                        choices=['linear', 'log_linear'],
                        help="Schedule for --loss-type A beta annealing")
    parser.add_argument("--gamma",         type=float, default=1000.)
    parser.add_argument("--loss-type",     type=str,   default='B',
                        choices=['H', 'B', 'A'],
                        help="H=Higgins beta-VAE, B=Burgess capacity annealing, A=linear beta annealing")
    parser.add_argument("--recon-loss-type", type=str, default='mse',
                        choices=['mse', 'l1', 'sqrt_mse', 'log_mse'],
                        help="Reconstruction loss: mse=current pixel-space MSE, "
                             "l1=current pixel-space L1/absolute error, "
                             "sqrt_mse=MSE after sqrt transform, "
                             "log_mse=MSE after log1p(x / recon_log_x0)")
    parser.add_argument("--recon-loss-weight", type=float, default=1.0,
                        help="Multiplier on the reconstruction loss before adding the KL term")
    parser.add_argument("--recon-log-x0", type=float, default=1e-3,
                        help="x0 scale for --recon-loss-type log_mse")
    parser.add_argument("--no-recenter-output", action="store_true",
                        help="Disable differentiable decoder-output recentering before reconstruction loss")
    parser.add_argument("--max-capacity",  type=float, default=25.0)
    parser.add_argument("--cap-max-iter",  type=int,   default=int(1e5))
    parser.add_argument("--n-generate",    type=int,   default=64,
                        help="Number of jets to generate after training")
    parser.add_argument("--output",        type=str,   default="best_model_betavae.pth")
    args = parser.parse_args()

    downsample_factor = 2 ** len(args.hidden_dims)
    if args.img_size % downsample_factor != 0:
        parser.error(
            f"--img-size must be divisible by 2^len(hidden_dims)={downsample_factor} "
            f"(got img_size={args.img_size}, hidden_dims={args.hidden_dims})"
        )
    if args.eta_max <= 0:
        parser.error(f"--eta-max must be positive (got {args.eta_max})")
    if args.phi_max <= 0:
        parser.error(f"--phi-max must be positive (got {args.phi_max})")

    if args.cache_dir is None and args.g_jet is None and args.u_jet is None:
        parser.error("Provide either --cache-dir OR at least one of --g-jet or --u-jet")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Branch A: load from pre-built cache                                 #
    # ------------------------------------------------------------------ #
    if args.cache_dir is not None:
        print(f"\nLoading from cache: {args.cache_dir}")
        meta = np.load(os.path.join(args.cache_dir, "meta.npz"))
        n_channels      = int(meta["n_channels"])
        eta_bins        = int(meta["eta_bins"])
        phi_bins        = int(meta["phi_bins"])
        eta_max         = float(meta["eta_max"].item()) if "eta_max" in meta.files else args.eta_max
        phi_max         = float(meta["phi_max"].item()) if "phi_max" in meta.files else args.phi_max
        pid_lookup      = meta["pid_lookup"]
        pid_min         = int(meta["pid_min"])
        if "normalization" in meta.files:
            cache_normalization = str(meta["normalization"].item())
        elif "mean_img" in meta.files and "std_img" in meta.files:
            cache_normalization = "standardized_event_pt"
        else:
            cache_normalization = "event_pt"
        cache_mean_img = meta["mean_img"] if "mean_img" in meta.files else None
        cache_std_img  = meta["std_img"] if "std_img" in meta.files else None

        # Reconstruct pid_channel_map from lookup for visualisation later
        pid_channel_map = {}
        for offset, ch in enumerate(pid_lookup):
            if ch >= 0:
                pid_channel_map[pid_min + offset] = int(ch)

        train_dataset = CachedJetImageDataset(
            args.cache_dir, "train",
            normalization=cache_normalization,
            mean_img=cache_mean_img,
            std_img=cache_std_img,
        )
        val_dataset   = CachedJetImageDataset(
            args.cache_dir, "val",
            normalization=cache_normalization,
            mean_img=cache_mean_img,
            std_img=cache_std_img,
        )
        print(f"  n_channels={n_channels}, image=({phi_bins}x{eta_bins}), "
              f"eta_max={eta_max:g}, phi_max={phi_max:g}")

        # Override img_size from cache so the model is built correctly
        args.img_size = eta_bins

    # ------------------------------------------------------------------ #
    # Branch B: process .dat files on the fly                             #
    # ------------------------------------------------------------------ #
    else:
        print(f"\nLoading data ...\n  g-jet: {args.g_jet}\n  u-jet: {args.u_jet}")
        X_events, y_labels = load_dataset(args.g_jet, args.u_jet)
        print(f"Loaded {len(X_events)} events total")

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

        pid_channel_map     = build_pid_channel_map(X_events)
        pid_lookup, pid_min = build_pid_lookup(pid_channel_map)
        n_channels          = len(pid_channel_map)
        eta_bins = phi_bins = args.img_size
        eta_max = args.eta_max
        phi_max = args.phi_max
        print(f"Image shape: ({n_channels}, {phi_bins}, {eta_bins})")

        indices = np.arange(len(y_labels))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_labels)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, random_state=42,
            stratify=y_labels[train_val_idx])

        X_train = [X_events[i] for i in train_idx]
        X_val   = [X_events[i] for i in val_idx]
        print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(test_idx)}")

        print("\nUsing event-level pT normalized images; no per-pixel standardization.")

        img_kwargs = dict(
            pid_lookup=pid_lookup, pid_min=pid_min, n_channels=n_channels,
            eta_bins=eta_bins, phi_bins=phi_bins,
            eta_max=eta_max, phi_max=phi_max,
        )
        train_dataset = JetImageDataset(X_train, **img_kwargs)
        val_dataset   = JetImageDataset(X_val,   **img_kwargs)

    device = select_mps_or_cpu_device(args.device)
    print(f"\nUsing device: {device}")
    if device.type == "cpu":
        print("  MPS is not available; training will run on CPU.")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    # --- Model ---
    model = BetaVAE(
        in_channels=n_channels,
        latent_dim=args.latent_dim,
        image_size=args.img_size,
        hidden_dims=args.hidden_dims.copy(),
        beta=args.beta,
        beta_start=args.beta_start,
        beta_anneal_iter=args.beta_anneal_iter,
        beta_anneal_schedule=args.beta_anneal_schedule,
        gamma=args.gamma,
        loss_type=args.loss_type,
        recon_loss_type=args.recon_loss_type,
        recon_loss_weight=args.recon_loss_weight,
        recon_log_x0=args.recon_log_x0,
        max_capacity=args.max_capacity,
        Capacity_max_iter=args.cap_max_iter,
        recenter_output=not args.no_recenter_output,
        eta_max=eta_max,
        phi_max=phi_max,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BetaVAE parameters: {n_params:,}")
    print(f"  latent_dim={args.latent_dim}, img_size={args.img_size}, "
          f"hidden_dims={args.hidden_dims}, spatial_size={model.spatial_size}, "
          f"loss_type={args.loss_type}, recon_loss_type={args.recon_loss_type}, "
          f"recon_loss_weight={args.recon_loss_weight:g}, "
          f"recon_log_x0={args.recon_log_x0:g}, "
          f"recenter_output={model.recenter_output}")
    if args.loss_type == 'A':
        print(f"  beta annealing: schedule={args.beta_anneal_schedule}, "
              f"beta_start={args.beta_start:g} -> beta={args.beta:g} "
              f"over {args.beta_anneal_iter} train iterations")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    # --- Training ---
    best_val_loss = float('inf')
    patience_counter = 0
    history = {k: [] for k in
               ['train_loss', 'val_loss', 'train_recon', 'val_recon', 'train_kld', 'val_kld']}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kld = train_one_epoch(
            model, train_loader, optimizer, device, len(train_dataset),
            grad_clip=args.grad_clip)
        val_loss, val_recon, val_kld = eval_one_epoch(
            model, val_loader, device, len(val_dataset), split='val')

        for k, v in zip(
            ['train_loss', 'val_loss', 'train_recon', 'val_recon', 'train_kld', 'val_kld'],
            [train_loss, val_loss, train_recon, val_recon, train_kld, val_kld]
        ):
            history[k].append(v)

        if not np.isfinite(val_loss):
            print("\nValidation loss is non-finite; stopping before checkpoint save.", flush=True)
            break
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch:03d} | "
              f"train_loss={train_loss:.5f} (recon={train_recon:.5f} kld={train_kld:.5f}) | "
              f"val_loss={val_loss:.5f} (recon={val_recon:.5f} kld={val_kld:.5f})", flush=True)
        if args.loss_type == 'A':
            print(f"  beta={model.current_beta():.6g}", flush=True)

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_channels': n_channels,
                'latent_dim': args.latent_dim,
                'img_size': args.img_size,
                'hidden_dims': args.hidden_dims,
                'loss_type': args.loss_type,
                'beta': args.beta,
                'beta_start': args.beta_start,
                'beta_anneal_iter': args.beta_anneal_iter,
                'beta_anneal_schedule': args.beta_anneal_schedule,
                'gamma': args.gamma,
                'max_capacity': args.max_capacity,
                'cap_max_iter': args.cap_max_iter,
                'pid_channel_map': pid_channel_map,
                'normalization': 'event_pt',
                'output_mode': 'event_pt',
                'recenter_output': not args.no_recenter_output,
                'eta_max': eta_max,
                'phi_max': phi_max,
                'recon_loss_type': args.recon_loss_type,
                'recon_loss_weight': args.recon_loss_weight,
                'recon_log_x0': args.recon_log_x0,
            }, args.output)
            print(f"  -> saved {args.output}  (val_loss={val_loss:.5f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.5f}")

    # --- Plot training curves ---
    plot_prefix = args.output.replace(".pth", "")
    save_training_plots(history, output_prefix=plot_prefix)

    # --- Load best model and generate jets ---
    if not os.path.exists(args.output):
        raise RuntimeError(
            f"No checkpoint was saved at {args.output}. "
            "Training did not produce a finite validation loss; try a smaller --lr "
            "such as 1e-4 or run with --device cpu to rule out MPS instability."
        )

    print(f"\nLoading best model from {args.output} ...")
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Generating {args.n_generate} jet images from latent prior ...")
    save_generated_jets(
        model, args.n_generate, device,
        pid_channel_map=pid_channel_map,
        output_prefix=plot_prefix,
        eta_bins=eta_bins, phi_bins=phi_bins,
        eta_max=eta_max, phi_max=phi_max,
    )

    end_time = datetime.now()
    print(f"\nDone. Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
