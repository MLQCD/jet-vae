"""
core.py - reusable backend for CNN Beta-VAE latent space visualization.

No UI dependency. Import from Streamlit, Jupyter, or plain scripts.
"""

import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# -- PID channel display config ----------------------------------------------
# pid_channel_map from checkpoint: {-211:0, -13:1, -11:2, 11:3, 13:4, 22:5, 211:6, 2112:7}
CHANNEL_NAMES  = ['pi-', 'mu-', 'e-', 'e+', 'mu+', 'gamma', 'pi+', 'n']
CHANNEL_COLORS = ['#e05c5c', '#6fb7e8', '#f5a623', '#7ed321', '#50e3c2', '#bd10e0', '#f8e71c', '#9b9b9b']
CHANNEL_PIDS   = [-211, -13, -11, 11, 13, 22, 211, 2112]
NO_PID_CHANNEL_NAMES = ['total pT']
NO_PID_CHANNEL_COLORS = ['#4e9af1']


# -- Model definition ---------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class BetaCNNVAE(nn.Module):
    def __init__(self, n_channels=8, latent_dim=32, img_size=32, hidden_dims=None,
                 use_residual=True, encoder_bias=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        if encoder_bias is None:
            encoder_bias = not use_residual

        enc = []
        in_ch = n_channels
        for h in hidden_dims:
            layers = [
                nn.Conv2d(in_ch, h, 3, stride=2, padding=1, bias=encoder_bias),
                nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            ]
            if use_residual:
                layers.append(_ResidualBlock(h))
            enc.append(nn.Sequential(*layers))
            in_ch = h
        self.encoder = nn.ModuleList(enc)

        sp   = img_size // (2 ** len(hidden_dims))
        flat = hidden_dims[-1] * sp * sp
        self.fc_mu          = nn.Linear(flat, latent_dim)
        self.fc_var         = nn.Linear(flat, latent_dim)
        self.decoder_input  = nn.Linear(latent_dim, flat)

        dec_dims = hidden_dims[::-1]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Sequential(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i+1], 3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(dec_dims[i+1]), nn.ReLU(inplace=True),
            ))
        self.decoder = nn.ModuleList(dec)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[-1], dec_dims[-1], 3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dec_dims[-1]), nn.ReLU(inplace=True),
            nn.Conv2d(dec_dims[-1], n_channels, 3, padding=1),
        )
        self._sp = sp
        self._hd = hidden_dims

    def encode(self, x):
        h = x
        for s in self.encoder:
            h = s(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z):
        h = self.decoder_input(z).view(-1, self._hd[-1], self._sp, self._sp)
        for s in self.decoder:
            h = s(h)
        return self.final_layer(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv


# -- Checkpoint loading --------------------------------------------------------

def load_checkpoint(checkpoint_path, device=None):
    """
    Load a BetaCNNVAE checkpoint.

    Returns
    -------
    model : BetaCNNVAE  (eval mode, on device)
    meta  : dict - latent_dim, n_channels, img_size, mean_img, std_img,
                   pid_channel_map, val_loss, epoch, device, checkpoint_path,
                   log_path (if found)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    hidden_dims, use_residual, encoder_bias = _infer_model_architecture(state_dict)

    model = BetaCNNVAE(
        n_channels  = int(ckpt['n_channels']),
        latent_dim  = int(ckpt['latent_dim']),
        img_size    = int(ckpt['img_size']),
        hidden_dims = hidden_dims,
        use_residual = use_residual,
        encoder_bias = encoder_bias,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    n_channels = int(ckpt['n_channels'])
    img_size = int(ckpt['img_size'])
    normalization = ckpt.get('normalization', 'mean_std')
    channel_mode = _checkpoint_channel_mode(ckpt)
    mean_img = np.asarray(
        ckpt.get('mean_img', np.zeros((n_channels, img_size, img_size), dtype=np.float32)),
        dtype=np.float32,
    )
    std_img = np.asarray(
        ckpt.get('std_img', np.ones((n_channels, img_size, img_size), dtype=np.float32)),
        dtype=np.float32,
    )

    # Try to locate the training log: same dir, *.log, or one level up
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    log_path = _find_log(ckpt_dir)

    meta = dict(
        latent_dim      = int(ckpt['latent_dim']),
        n_channels      = int(ckpt['n_channels']),
        img_size        = int(ckpt['img_size']),
        mean_img        = mean_img,
        std_img         = std_img,
        normalization   = normalization,
        channel_mode    = channel_mode,
        channel_names   = channel_names_for_mode(channel_mode, n_channels),
        channel_colors  = channel_colors_for_mode(channel_mode, n_channels),
        hidden_dims     = hidden_dims,
        use_residual    = use_residual,
        pid_channel_map = ckpt.get('pid_channel_map', {}),
        val_loss        = float(ckpt.get('val_loss', float('nan'))),
        epoch           = int(ckpt.get('epoch', -1)),
        device          = device,
        checkpoint_path = checkpoint_path,
        log_path        = log_path,
    )
    return model, meta


def _checkpoint_channel_mode(ckpt):
    if 'channel_mode' in ckpt:
        return str(ckpt['channel_mode'])
    pid_map = ckpt.get('pid_channel_map', {})
    pid_map = {int(pid): int(ch) for pid, ch in pid_map.items()} if hasattr(pid_map, 'items') else {}
    if int(ckpt['n_channels']) == 1 and (not pid_map or pid_map == {0: 0}):
        return 'eta_phi_total_pt_no_pid'
    return 'pid_channels'


def uses_no_pid_channels(meta):
    return meta.get('channel_mode') == 'eta_phi_total_pt_no_pid'


def channel_names_for_mode(channel_mode, n_channels):
    if channel_mode == 'eta_phi_total_pt_no_pid':
        return NO_PID_CHANNEL_NAMES[:n_channels]
    labels = list(CHANNEL_NAMES[:n_channels])
    labels.extend(f'ch {i}' for i in range(len(labels), n_channels))
    return labels


def channel_colors_for_mode(channel_mode, n_channels):
    if channel_mode == 'eta_phi_total_pt_no_pid':
        return NO_PID_CHANNEL_COLORS[:n_channels]
    colors = list(CHANNEL_COLORS[:n_channels])
    fallback = cm.get_cmap('tab20', max(n_channels, 1))
    colors.extend(matplotlib.colors.to_hex(fallback(i)) for i in range(len(colors), n_channels))
    return colors


def _infer_model_architecture(state_dict):
    """Infer architecture options needed to load older and newer checkpoints."""
    hidden_dims = []
    i = 0
    while f'encoder.{i}.0.weight' in state_dict:
        hidden_dims.append(int(state_dict[f'encoder.{i}.0.weight'].shape[0]))
        i += 1

    if not hidden_dims:
        hidden_dims = [32, 64, 128, 256]

    use_residual = any('.block.' in key for key in state_dict)
    encoder_bias = 'encoder.0.0.bias' in state_dict
    return hidden_dims, use_residual, encoder_bias


def _find_log(ckpt_dir):
    candidates = [
        ckpt_dir,
        os.path.join(ckpt_dir, '..'),           # one level up (e.g. vae_cnn/)
        os.path.join(ckpt_dir, '..', '..'),
        os.path.join(ckpt_dir, '..', '..', 'vae_logs'),
    ]
    for d in candidates:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.log'):
                return os.path.join(d, f)
    return None


# -- Decoding -----------------------------------------------------------------

def decode_z(model, z, meta):
    """
    Decode latent vector z into a denormalized 8-channel jet image.

    Parameters
    ----------
    z : array-like, shape (latent_dim,)

    Returns
    -------
    img : ndarray, shape (C, H, W), float32 - physical units (pT density)
    """
    device = meta['device']
    z_t = torch.tensor(np.asarray(z, dtype=np.float32), device=device).unsqueeze(0)
    with torch.no_grad():
        raw = model.decode(z_t).cpu().numpy()[0]   # (C, H, W) normalized

    if meta.get('normalization') == 'event_pt':
        raw = raw - np.max(raw)
        img = np.exp(raw)
        return (img / (img.sum() + 1e-12)).astype(np.float32)

    img = raw * meta['std_img'] + meta['mean_img']
    img = np.clip(img, 0.0, None)
    return img


# -- Image observables ---------------------------------------------------------

def compute_image_observables(img, pt_min=1e-4):
    """
    Compute physics observables from an (C, H, W) jet image.

    Returns dict with: total_pt, total_pt_per_channel (C,),
                       mean_eta, mean_phi, std_eta, std_phi, std_R,
                       pT_entropy, multiplicity
    """
    C, H, W = img.shape
    # eta-axis = rows (dim 1), phi-axis = cols (dim 2)
    eta_vals = np.linspace(-0.5, 0.5, H)
    phi_vals = np.linspace(-0.5, 0.5, W)
    ETA, PHI = np.meshgrid(eta_vals, phi_vals, indexing='ij')   # (H, W)

    total_img = img.sum(axis=0)   # (H, W)

    total_pt = float(total_img.sum())
    total_pt_per_channel = img.sum(axis=(1, 2))   # (C,)

    eps = 1e-8
    if total_pt > eps:
        mean_eta  = float((total_img * ETA).sum() / total_pt)
        mean_phi  = float((total_img * PHI).sum() / total_pt)
        std_eta = float(np.sqrt(((total_img * (ETA - mean_eta)**2).sum() / total_pt)))
        std_phi = float(np.sqrt(((total_img * (PHI - mean_phi)**2).sum() / total_pt)))
        std_R = float(np.sqrt(std_eta**2 + std_phi**2))
        p = total_img[total_img > eps] / total_pt
        pT_entropy = -float((p * np.log(p)).sum())
    else:
        mean_eta = mean_phi = std_eta = std_phi = std_R = pT_entropy = 0.0

    multiplicity = int((total_img > pt_min).sum())

    return dict(
        total_pt             = total_pt,
        total_pt_per_channel = total_pt_per_channel,
        mean_eta             = mean_eta,
        mean_phi             = mean_phi,
        std_eta              = std_eta,
        std_phi              = std_phi,
        std_R                = std_R,
        pT_entropy           = pT_entropy,
        width_eta            = std_eta,
        width_phi            = std_phi,
        multiplicity         = multiplicity,
    )


# -- Batch encoding -----------------------------------------------------------

def batch_encode(model, images_raw, meta, batch_size=256, already_normalized=True):
    """
    Encode a batch of raw (unnormalized) images.

    Parameters
    ----------
    images_raw : ndarray (N, C, H, W)
        If already_normalized is True, images are fed directly to the encoder.
        If False, physical pT images are normalized with the checkpoint metadata.

    Returns
    -------
    mus    : ndarray (N, latent_dim)
    log_vars : ndarray (N, latent_dim)
    """
    device = meta['device']
    N      = len(images_raw)

    all_mu, all_lv = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = images_raw[start:start + batch_size]
            if already_normalized:
                norm = batch
            elif meta.get('normalization') == 'event_pt':
                totals = batch.sum(axis=(1, 2, 3), keepdims=True)
                norm = np.divide(batch, totals, out=np.zeros_like(batch), where=totals > 0)
            else:
                std = meta['std_img'][None]
                mean = meta['mean_img'][None]
                valid = std > 1e-6
                std_safe = np.where(valid, std, 1.0)
                norm = np.where(valid, (batch - mean) / std_safe, 0.0)
            x_t   = torch.tensor(norm, dtype=torch.float32, device=device)
            mu, lv = model.encode(x_t)
            all_mu.append(mu.cpu().numpy())
            all_lv.append(lv.cpu().numpy())

    return np.concatenate(all_mu, axis=0), np.concatenate(all_lv, axis=0)


def compute_active_units(mus, log_vars, threshold=0.01):
    """
    Compute per-dimension activity statistics.

    Active unit criterion (Burda et al. 2015):
        A_d = Var_x[ mu_d(x) ]  >  threshold

    Parameters
    ----------
    mus      : (N, D)
    log_vars : (N, D)
    threshold: float - default 0.01

    Returns
    -------
    dict with:
        au_mask          : bool (D,)   - True = active
        n_active         : int
        var_mu           : (D,) - Var_x[mu_d]
        mean_kld_per_dim : (D,) - mean KL per dimension: 0.5*(mu^2+sigma^2- logsigma^2-1)
        mean_mu          : (D,) - mean mu_d across data
        std_mu           : (D,) - std of mu_d across data  (= sqrt(var_mu))
    """
    var_mu           = mus.var(axis=0)                          # (D,)
    std_mu           = mus.std(axis=0)
    mean_mu          = mus.mean(axis=0)
    sigma2           = np.exp(log_vars)                         # sigma^2
    mean_kld_per_dim = 0.5 * (mus**2 + sigma2 - log_vars - 1).mean(axis=0)
    au_mask          = var_mu > threshold

    return dict(
        au_mask          = au_mask,
        n_active         = int(au_mask.sum()),
        var_mu           = var_mu,
        std_mu           = std_mu,
        mean_mu          = mean_mu,
        mean_kld_per_dim = mean_kld_per_dim,
    )


# -- Sweep --------------------------------------------------------------------

def sweep_dimension(model, z_base, dim, values, meta):
    """
    Sweep one latent dimension, hold rest fixed at z_base.

    Returns
    -------
    list of (img_ndarray, obs_dict) - one per value.
    """
    results = []
    for v in values:
        z = np.array(z_base, dtype=np.float32)
        z[dim] = float(v)
        img = decode_z(model, z, meta)
        obs = compute_image_observables(img)
        results.append((img, obs))
    return results


def compute_impact_heatmap(model, z_base, meta, probe_sigma=2.0):
    """
    |Delta observable| when each dimension is nudged +/-probe_sigma.

    Returns
    -------
    impact   : ndarray (latent_dim, n_obs)
    obs_keys : list[str]
    """
    obs_keys   = ['std_eta', 'std_phi', 'std_R', 'pT_entropy']
    latent_dim = meta['latent_dim']
    impact     = np.zeros((latent_dim, len(obs_keys)), dtype=np.float32)

    for d in range(latent_dim):
        neg = sweep_dimension(model, z_base, d, [-probe_sigma], meta)[0][1]
        pos = sweep_dimension(model, z_base, d, [+probe_sigma], meta)[0][1]
        for j, k in enumerate(obs_keys):
            impact[d, j] = abs(float(pos[k]) - float(neg[k]))

    return impact, obs_keys


# -- Training log parser -------------------------------------------------------

def parse_training_log(log_path):
    """
    Parse lines like:
        Epoch 001 | train_loss=X (recon=X kld=X) | val_loss=X (recon=X kld=X)

    Returns dict of lists: epoch, train_loss, train_recon, train_kld,
                                   val_loss,   val_recon,   val_kld
    """
    pat = re.compile(
        r'Epoch\s+(\d+)\s*\|'
        r'\s*train_loss=([\d.]+)\s*\(recon=([\d.]+)\s+kld=([\d.]+)\)'
        r'\s*\|\s*val_loss=([\d.]+)\s*\(recon=([\d.]+)\s+kld=([\d.]+)\)'
    )
    data = {k: [] for k in ('epoch', 'train_loss', 'train_recon', 'train_kld',
                              'val_loss', 'val_recon', 'val_kld')}
    try:
        with open(log_path, encoding='utf-8') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    data['epoch'].append(int(m.group(1)))
                    data['train_loss'].append(float(m.group(2)))
                    data['train_recon'].append(float(m.group(3)))
                    data['train_kld'].append(float(m.group(4)))
                    data['val_loss'].append(float(m.group(5)))
                    data['val_recon'].append(float(m.group(6)))
                    data['val_kld'].append(float(m.group(7)))
    except Exception:
        pass
    return data


# -- Plotting helpers ----------------------------------------------------------

def make_channel_grid(img, title='', vmax=None, cmap='Greys', channel_names=None, channel_colors=None):
    """
    Draw all 8 channels as a 2x4 grid. Returns a matplotlib Figure.
    """
    C = img.shape[0]
    if channel_names is None:
        mode = 'eta_phi_total_pt_no_pid' if C == 1 else 'pid_channels'
        channel_names = channel_names_for_mode(mode, C)
    if channel_colors is None:
        mode = 'eta_phi_total_pt_no_pid' if C == 1 else 'pid_channels'
        channel_colors = channel_colors_for_mode(mode, C)
    ncols = 4
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.4))
    if vmax is None:
        vmax = img.max() if img.max() > 0 else 1.0
    vmax = max(vmax, 1e-8)

    for idx in range(ncols * nrows):
        ax = axes.flat[idx]
        if idx < C:
            im = ax.imshow(img[idx], origin='lower', cmap=cmap,
                           vmin=0, vmax=vmax, aspect='auto',
                           extent=[-0.5, 0.5, -0.5, 0.5])
            ax.set_title(f'{channel_names[idx]}  (ch {idx})',
                         fontsize=8, color=channel_colors[idx], pad=3)
            ax.set_xlabel('phi', fontsize=7, labelpad=1)
            ax.set_ylabel('eta', fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def make_total_image(img, title='', cmap='Greys'):
    """
    Draw pT-summed image (all channels collapsed). Returns a Figure.
    """
    total = img.sum(axis=0)
    vmax = float(total.max()) if total.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(total, origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='auto',
                   extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_xlabel('phi', fontsize=9)
    ax.set_ylabel('eta', fontsize=9)
    ax.set_title(title or 'Total pT image', fontsize=9)
    plt.colorbar(im, ax=ax, label='pT density', fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_sweep_figure(results, values, dim):
    """
    Top row: total-pT images for each sweep step.
    Bottom row: observable curves vs z[dim].
    """
    n     = len(values)
    obs_k = ['std_eta', 'std_phi', 'std_R', 'pT_entropy']
    fig   = plt.figure(figsize=(max(n * 2.5, 12), 6))
    gs    = fig.add_gridspec(2, max(n, len(obs_k)), hspace=0.5, wspace=0.35)

    vmax = max(r[0].sum(axis=0).max() for r in results) or 1.0

    for i, (v, (img, obs)) in enumerate(zip(values, results)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img.sum(axis=0), origin='lower', cmap='Greys',
                  vmin=0, vmax=vmax, aspect='auto',
                  extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title(f'z[{dim}]={v:+.1f}', fontsize=8)
        ax.set_xlabel('phi', fontsize=7, labelpad=1)
        ax.set_ylabel('eta', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)

    obs_vals = {k: [r[1][k] for r in results] for k in obs_k}
    for j, key in enumerate(obs_k):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(values, obs_vals[key], 'o-', lw=1.5, ms=4)
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel(f'z[{dim}]', fontsize=8)
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, ls='--', alpha=0.35)

    fig.suptitle(f'Latent sweep: dimension {dim}', fontsize=10, y=1.01)
    return fig


# -- Dataset loading -----------------------------------------------------------

# Label mapping: 0 = quark, 1 = gluon  (inferred from training convention)
LABEL_NAMES = {0: 'quark', 1: 'gluon'}

_NPY_HEADER_BYTES = 128   # conservative upper bound for .npy header size


def _count_complete_images(path, n_declared, C, H, W):
    """Return how many complete (C,H,W) float32 images fit in the file."""
    data_bytes = os.path.getsize(path) - _NPY_HEADER_BYTES
    per_img    = C * H * W * 4   # float32
    return min(n_declared, max(0, data_bytes // per_img))


def load_dataset_split(cache_dir, split='val'):
    """
    Load one split from the cached jet-image dataset.

    Parameters
    ----------
    cache_dir : str  - directory containing *_images.npy etc.
    split     : str  - 'train', 'val', or 'test'

    Returns
    -------
    images : ndarray (N, C, H, W) float32  - raw (unnormalized) pT images
    labels : ndarray (N,)         int64    - 0=quark, 1=gluon
    pt_sums: ndarray (N,)         float32  - scalar jet pT
    n_total: int   - declared size (may differ from N for a truncated train file)
    """
    img_path = os.path.join(cache_dir, f'{split}_images.npy')
    lab_path = os.path.join(cache_dir, f'{split}_labels.npy')
    pt_path  = os.path.join(cache_dir, f'{split}_pt_sums.npy')

    labels  = np.load(lab_path)
    if os.path.exists(pt_path):
        pt_sums = np.load(pt_path)
    else:
        # Older event-pt-normalized caches do not store per-event pT sums.
        pt_sums = np.ones(len(labels), dtype=np.float32)
    n_total = len(labels)

    # detect truncation
    n_complete = _count_complete_images(img_path, n_total, 8, 32, 32)

    if n_complete == n_total:
        images = np.load(img_path)
    else:
        # partial read: load only what is safely on disk
        header_size = os.path.getsize(img_path) - n_complete * 8 * 32 * 32 * 4
        with open(img_path, 'rb') as f:
            f.seek(header_size)
            images = np.frombuffer(f.read(), dtype=np.float32).copy()
        images = images.reshape(n_complete, 8, 32, 32)
        labels  = labels[:n_complete]
        pt_sums = pt_sums[:n_complete]

    return images, labels, pt_sums, n_total


def encode_image(model, raw_img, meta, already_normalized=False):
    """
    Encode a (C,H,W) image -> (mu, log_var).

    Parameters
    ----------
    already_normalized : bool
        True  -> image is already (raw - mean) / std (e.g. loaded from cache .npy).
                 Feed directly to the encoder.
        False -> image is in physical pT units (e.g. built from raw .dat).
                 Normalise first; pixels where training std ~ 0 are zeroed out.
    """
    device = meta['device']
    if already_normalized:
        norm = raw_img
    elif meta.get('normalization') == 'event_pt':
        norm = normalise_to_unit_sum(raw_img)
    else:
        std      = meta['std_img']
        mean     = meta['mean_img']
        valid    = std > 1e-6
        std_safe = np.where(valid, std, 1.0)
        norm     = np.where(valid, (raw_img - mean) / std_safe, 0.0)
    x_t = torch.tensor(norm[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, lv = model.encode(x_t)
    return mu.cpu().numpy()[0], lv.cpu().numpy()[0]


def reconstruct_image(model, raw_img, meta, use_sample=False, already_normalized=False):
    """
    Encode then decode an image. Returns denormalized reconstruction (C,H,W).

    Parameters
    ----------
    already_normalized : bool - see encode_image.
    """
    mu, lv = encode_image(model, raw_img, meta, already_normalized=already_normalized)
    if use_sample:
        std = np.exp(0.5 * lv)
        z   = (mu + std * np.random.randn(*std.shape)).astype(np.float32)
    else:
        z = mu
    return decode_z(model, z, meta), mu


def make_compare_figure(orig, recon, cmap='Greys', title_orig='Original', title_recon='Reconstruction'):
    """
    Side-by-side total-pT images: original vs reconstruction.
    Returns a Figure.
    """
    vmax = max(orig.sum(0).max(), recon.sum(0).max(), 1e-8)
    ext  = [-0.5, 0.5, -0.5, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, img, ttl in zip(axes, [orig, recon], [title_orig, title_recon]):
        im = ax.imshow(img.sum(0), origin='lower', cmap=cmap,
                       vmin=0, vmax=vmax, aspect='auto', extent=ext)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('phi', fontsize=9)
        ax.set_ylabel('eta', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='pT density')
    fig.tight_layout()
    return fig


def make_channel_compare_figure(orig, recon, cmap='Greys', channel_names=None, channel_colors=None):
    """
    2x8 grid: top row = original channels, bottom row = reconstruction channels.
    """
    C    = orig.shape[0]
    if channel_names is None:
        mode = 'eta_phi_total_pt_no_pid' if C == 1 else 'pid_channels'
        channel_names = channel_names_for_mode(mode, C)
    if channel_colors is None:
        mode = 'eta_phi_total_pt_no_pid' if C == 1 else 'pid_channels'
        channel_colors = channel_colors_for_mode(mode, C)
    vmax = max(orig.max(), recon.max(), 1e-8)
    ext  = [-0.5, 0.5, -0.5, 0.5]

    fig, axes = plt.subplots(2, C, figsize=(max(C, 1) * 2.2, 4.8), squeeze=False)
    for col in range(C):
        for row, (img, label) in enumerate([(orig, 'Orig'), (recon, 'Recon')]):
            ax = axes[row, col]
            im = ax.imshow(img[col], origin='lower', cmap=cmap,
                           vmin=0, vmax=vmax, aspect='auto', extent=ext)
            if row == 0:
                ax.set_title(f'{channel_names[col]}\n(ch {col})',
                             fontsize=7.5, color=channel_colors[col])
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=5)
    fig.suptitle('Per-channel comparison  (top: original - bottom: reconstruction)', fontsize=9)
    fig.tight_layout()
    return fig


def make_impact_figure(impact, obs_labels):
    """Normalized heatmap: latent dim x observable."""
    norm = impact / (impact.max(axis=0, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(7, max(6, impact.shape[0] * 0.28)))
    im = ax.imshow(norm.T, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_yticks(range(len(obs_labels)))
    ax.set_yticklabels(obs_labels, fontsize=9)
    ax.set_xlabel('Latent dimension', fontsize=10)
    ax.set_title('Impact heatmap (normalized per observable)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Normalized |Delta|')
    fig.tight_layout()
    return fig


# -- Raw .dat file parsing -----------------------------------------------------

# PID -> channel index (same as checkpoint pid_channel_map)
_PID_TO_CH = {pid: ch for ch, pid in enumerate(CHANNEL_PIDS)}


def parse_dat_file(path, max_events=None):
    """
    Parse a HepMC-style .dat file.

    Format per line:  pid  px  py  pz  E

    Returns
    -------
    list of ndarray, each (N_const, 5) float32: [pid, px, py, pz, E]
    """
    events  = []
    current = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('<event_begin>'):
                current = []
            elif line.startswith('<event_end>'):
                if current:
                    events.append(np.asarray(current, dtype=np.float32))
                if max_events and len(events) >= max_events:
                    break
            else:
                parts = line.split()
                if len(parts) == 5:
                    current.append([float(p) for p in parts])
    return events


def constituents_to_kinematics(constituents):
    """
    Convert (N,5) [pid,px,py,pz,E] array to a dict of kinematic arrays.

    Returns: pid(N), pt(N), eta(N), phi(N), E(N)
    """
    pid = constituents[:, 0].astype(int)
    px, py, pz, E = constituents[:, 1], constituents[:, 2], constituents[:, 3], constituents[:, 4]

    pt  = np.sqrt(px**2 + py**2)
    p   = np.sqrt(px**2 + py**2 + pz**2)
    eps = 1e-8
    eta = 0.5 * np.log((p + pz + eps) / (p - pz + eps))
    phi = np.arctan2(py, px)

    return dict(pid=pid, pt=pt, eta=eta, phi=phi, E=E, px=px, py=py, pz=pz)


def jet_axis(kin):
    """Compute jet axis (eta_jet, phi_jet, pT_jet, m_jet) from constituent kinematics."""
    px_j = kin['px'].sum()
    py_j = kin['py'].sum()
    pz_j = kin['pz'].sum()
    E_j  = kin['E'].sum()
    pt_j = float(np.sqrt(px_j**2 + py_j**2))
    p_j  = float(np.sqrt(px_j**2 + py_j**2 + pz_j**2))
    eps  = 1e-8
    eta_j = float(0.5 * np.log((p_j + pz_j + eps) / (p_j - pz_j + eps)))
    phi_j = float(np.arctan2(py_j, px_j))
    m2_j  = max(float(E_j**2 - p_j**2), 0.0)
    return dict(eta=eta_j, phi=phi_j, pt=pt_j, mass=float(np.sqrt(m2_j)))


def constituents_to_image(constituents, img_size=32, eta_max=0.5, phi_max=0.5,
                          pid_to_ch=None, n_channels=None, no_pid=False):
    """
    Build an (8, img_size, img_size) jet image from raw constituents.

    Steps:
      1. Compute kinematics
      2. Find jet axis, centre eta-phi on it
      3. Bin pT into a grid, one channel per particle type
    """
    if pid_to_ch is None:
        pid_to_ch = _PID_TO_CH

    kin   = constituents_to_kinematics(constituents)
    ax    = jet_axis(kin)
    n_ch  = 1 if no_pid else (n_channels or len(CHANNEL_PIDS))

    deta = kin['eta'] - ax['eta']
    dphi = ((kin['phi'] - ax['phi'] + np.pi) % (2 * np.pi)) - np.pi

    img   = np.zeros((n_ch, img_size, img_size), dtype=np.float32)
    eta_bins = np.linspace(-eta_max, eta_max, img_size + 1)
    phi_bins = np.linspace(-phi_max, phi_max, img_size + 1)

    for i in range(len(kin['pid'])):
        if no_pid:
            ch = 0
        else:
            ch = pid_to_ch.get(int(kin['pid'][i]), None)
            if ch is None or ch >= n_ch:
                continue
        ei = np.searchsorted(eta_bins, deta[i], side='right') - 1
        pi = np.searchsorted(phi_bins, dphi[i], side='right') - 1
        if 0 <= ei < img_size and 0 <= pi < img_size:
            img[ch, ei, pi] += kin['pt'][i]

    return img, ax, kin


def make_scatter_figure(kin, ax_jet, title='', cone_r=0.5, pt_scale=30.0,
                        pid_to_ch=None):
    """
    eta-phi scatter plot of jet constituents in the lab frame.
    Dot size proportional to pT, color = particle type.  Cone circle drawn around jet axis.
    """
    if pid_to_ch is None:
        pid_to_ch = _PID_TO_CH

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for ch, (pid_val, name, color) in enumerate(zip(CHANNEL_PIDS, CHANNEL_NAMES, CHANNEL_COLORS)):
        mask = kin['pid'] == pid_val
        if mask.any():
            ax.scatter(kin['phi'][mask], kin['eta'][mask],
                       s=np.clip(np.log1p(kin['pt'][mask]) * pt_scale, 5, 400),
                       c=color, label=f'{name} ({mask.sum()})',
                       alpha=0.80, edgecolors='k', linewidths=0.3, zorder=3)

    # jet axis marker
    ax.plot(ax_jet['phi'], ax_jet['eta'], '+', color='white',
            ms=14, mew=2, zorder=5, label=f'jet axis (pT={ax_jet["pt"]:.1f} GeV)')

    # cone circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ax_jet['phi'] + cone_r * np.cos(theta),
            ax_jet['eta'] + cone_r * np.sin(theta),
            '--', color='white', lw=0.8, alpha=0.5, zorder=4)

    ax.set_xlabel('phi (lab)', fontsize=9)
    ax.set_ylabel('eta (lab)', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.6,
              markerscale=0.7, handletextpad=0.3)
    ax.grid(True, ls='--', alpha=0.25)
    fig.tight_layout()
    return fig


def make_centered_scatter_figure(kin, ax_jet, title='', eta_max=0.5, phi_max=0.5):
    """
    eta-phi scatter in jet-centred frame (Delta eta, Delta phi), clipped to +/-eta_max/phi_max.
    """
    deta = kin['eta'] - ax_jet['eta']
    dphi = ((kin['phi'] - ax_jet['phi'] + np.pi) % (2 * np.pi)) - np.pi

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for ch, (pid_val, name, color) in enumerate(zip(CHANNEL_PIDS, CHANNEL_NAMES, CHANNEL_COLORS)):
        mask = kin['pid'] == pid_val
        if mask.any():
            ax.scatter(dphi[mask], deta[mask],
                       s=np.clip(np.log1p(kin['pt'][mask]) * 30, 5, 400),
                       c=color, label=name, alpha=0.80,
                       edgecolors='k', linewidths=0.3, zorder=3)

    ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlim(-phi_max, phi_max)
    ax.set_ylim(-eta_max, eta_max)
    ax.set_xlabel('Delta phi', fontsize=9)
    ax.set_ylabel('Delta eta', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.6,
              markerscale=0.7, handletextpad=0.3)
    ax.grid(True, ls='--', alpha=0.25)
    fig.tight_layout()
    return fig


def make_scatter_channel_grid(kin, ax_jet, title='', eta_max=0.5, phi_max=0.5):
    """
    2x4 grid of Delta eta-Delta phi scatter plots, one subplot per PID channel.
    Dot size proportional to log(1+pT).
    """
    deta = kin['eta'] - ax_jet['eta']
    dphi = ((kin['phi'] - ax_jet['phi'] + np.pi) % (2 * np.pi)) - np.pi

    fig, axes = plt.subplots(2, 4, figsize=(4 * 2.8, 2 * 2.8))

    for ch, (pid_val, name, color) in enumerate(zip(CHANNEL_PIDS, CHANNEL_NAMES, CHANNEL_COLORS)):
        ax = axes.flat[ch]
        mask = kin['pid'] == pid_val
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-phi_max, phi_max)
        ax.set_ylim(-eta_max, eta_max)
        ax.axhline(0, color='white', lw=0.4, ls='--', alpha=0.3)
        ax.axvline(0, color='white', lw=0.4, ls='--', alpha=0.3)
        ax.set_xlabel('Delta phi', fontsize=7, labelpad=1)
        ax.set_ylabel('Delta eta', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        if mask.any():
            ax.scatter(dphi[mask], deta[mask],
                       s=np.clip(np.log1p(kin['pt'][mask]) * 30, 5, 300),
                       c=color, alpha=0.85, edgecolors='k', linewidths=0.3)
            ax.set_title(f'{name}  n={mask.sum()}', fontsize=8, color=color, pad=3)
        else:
            ax.text(0, 0, 'no hits', ha='center', va='center',
                    fontsize=8, color='#555555')
            ax.set_title(name, fontsize=8, color=color, pad=3)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def normalise_to_unit_sum(img):
    """Divide image by its total sum so shape comparisons are scale-independent."""
    total = img.sum()
    return img / total if total > 0 else img
