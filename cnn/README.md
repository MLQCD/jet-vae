# Jet Physics — Beta-VAE for Jet Image Generation

Deep learning framework for High Energy Physics (HEP) jet analysis.
This README covers the **Beta-VAE generative model** for jet images, including data preprocessing, model architecture, and training.

---

## Repository Structure

```
jet_physics/
├── code_training/
│   ├── betaVae.py               # Beta-VAE model definition
│   ├── train_betavae.py         # Training script
│   ├── train_betavae_mps.py     # Apple MPS/CPU training script
│   ├── preprocess_jet_images.py # (Optional) Pre-compute dense jet image cache
│   ├── preprocess_jet_images_no_pid.py
│   │                           # Build one-channel no-PID image cache
│   └── visualize_training_log.py
│                               # Plot loss curves and latent activity
├── train_no_pid_mps.sh          # Local no-PID MPS training launcher
├── run_compiled_visualize_app.py # Run compiled visualization app
├── runs/                        # Timestamped local training outputs
├── datas/                       # Raw .dat particle data files
│   ├── g_jet_real_dynamic_sample.dat
│   └── u_jet_real_dynamic_sample.dat
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, NumPy, scikit-learn, matplotlib.
A CUDA-capable GPU is strongly recommended for training.

---

## Data Format

Raw data files are custom `.dat` text files with one jet event per block:

```
<event_begin>:id
pid  px  py  pz  E
pid  px  py  pz  E
...
<event_end>:id
```

- `pid` — particle species ID (integer)
- `px`, `py`, `pz` — momentum components (GeV)
- `E` — energy (GeV)

Two classes are used:

- **u-jet** (quark jet, label `0`)
- **g-jet** (gluon jet, label `1`)

---

## Workflow Overview

```
Raw .dat files
      │
      ▼
preprocess_jet_images.py ─►  dense cache ({split}_images.npy + meta.npz)
      │
      ▼
train_betavae.py  ──►  best_model_betavae.pth
                        training_curves.png
                        generated_samples.png
                        generated.npy
```

---

## No-PID Apple MPS Workflow

Use this workflow for local Apple Silicon training where PID is not used as an
image channel. The no-PID cache has shape `(N, 1, H, W)`: all valid particles
deposit pT into one shared eta/phi image.

Run these commands from:

```bash
cd path/to/jet-vae/cnn
```

### 1. Build the no-PID image cache

`ai4jet/vae_cnn/code_training/preprocess_jet_images_no_pid.py` reads raw
`.dat` files and writes a single-channel cache:

```bash
python ai4jet/vae_cnn/code_training/preprocess_jet_images_no_pid.py \
    --g-jet ../g_jet_real_dynamic_sample.dat \
    --u-jet ../u_jet_real_dynamic_sample.dat \
    --cache-dir ai4jet/vae_cnn/cache_jet_images_32_no_pid \
    --image-size 32 \
    --eta-max 0.5 \
    --phi-max 0.5
```

Expected cache files:

```text
cache_jet_images_32_no_pid/
├── meta.npz
├── train_images.npy
├── train_labels.npy
├── val_images.npy
├── val_labels.npy
├── test_images.npy
└── test_labels.npy
```

`meta.npz` should report `n_channels=1` and
`channel_mode=eta_phi_total_pt_no_pid`.

### 2. Train with Apple MPS or CPU

`ai4jet/vae_cnn/code_training/train_betavae_mps.py` trains from a prebuilt
cache and automatically uses MPS when available:

```bash
python ai4jet/vae_cnn/code_training/train_betavae_mps.py \
    --cache-dir ai4jet/vae_cnn/cache_jet_images_32_no_pid \
    --epochs 600 \
    --batch-size 256 \
    --lr 1e-3 \
    --latent-dim 32 \
    --hidden-dims 32,64,128,256,512 \
    --loss-type H \
    --beta 1.0 \
    --patience 20 \
    --n-generate 64 \
    --output ai4jet/vae_cnn/runs/no_pid_manual/best_model_betavae.pth
```

For the standard local run, use the launcher:

```bash
bash ai4jet/vae_cnn/train_no_pid_mps.sh
```

`train_no_pid_mps.sh` sets the no-PID cache path, training hyperparameters,
creates a timestamped directory under `ai4jet/vae_cnn/runs/`, and saves stdout
to `train.log`.

### 3. Inspect training logs and latent activity

`ai4jet/vae_cnn/code_training/visualize_training_log.py` parses `train.log`,
loads the best checkpoint, computes latent activity on a cache split, and saves
a summary plot plus optional CSV/JSON outputs.

Example for one run directory:

```bash
RUN=ai4jet/vae_cnn/runs/no_pid_beta_1_YYYYMMDD_HHMMSS

python ai4jet/vae_cnn/code_training/visualize_training_log.py \
    --log "$RUN/train.log" \
    --checkpoint "$RUN/best_model_betavae.pth" \
    --cache-dir ai4jet/vae_cnn/cache_jet_images_32_no_pid \
    --split val \
    --output "$RUN/training_losses_and_latent_activity.png" \
    --epoch-csv "$RUN/parsed_epoch_losses.csv" \
    --latent-csv "$RUN/latent_activity_best_model.csv" \
    --summary-json "$RUN/latent_activity_summary.json"
```

### 4. Open the compiled visualization app

`ai4jet/vae_cnn/run_compiled_visualize_app.py` is a fallback launcher for the
compiled visualization app when the source `app.py` and `core.py` files are not
available.

```bash
python ai4jet/vae_cnn/run_compiled_visualize_app.py
```

The script currently loads compiled files from:

```text
visualize/__pycache__/
```

### 5. Run directory contents

Each directory under `ai4jet/vae_cnn/runs/` is one training attempt. A typical
completed no-PID run contains:

```text
runs/no_pid_beta_.../
├── train.log
├── best_model_betavae.pth
├── best_model_betavae_training_curves.png
├── best_model_betavae_generated.npy
├── best_model_betavae_generated_samples.png
├── parsed_epoch_losses.csv
├── latent_activity_best_model.csv
└── latent_activity_summary.json
```

The `runs/` directory is for local experiment output and should not be committed
unless a small, specific result file is intentionally needed.

---

## Step 1 — Pre-build the Image Cache

Pre-computing jet images once avoids rebuilding them every training run.
Use `preprocess_jet_images.py` to create the dense format expected by `train_betavae.py --cache-dir`:

```bash
  python code_training/preprocess_jet_images.py \   
  --g-jet  data/g_jet_CMS_sample.dat \  
  --u-jet  data/u_jet_CMS_sample.dat \  
  --cache-dir /path/to/cache \  
  --image-size 32 \
  --eta-max    0.5 \
  --phi-max    0.5

Or
  python code_training/preprocess_jet_images.py --g-jet data/g_jet_CMS_sample.dat --u-jet data/u_jet_CMS_sample.dat --cache-dir path/to/cache --image-size 32 --eta-max 0.5 --phi-max 0.5
```

change the number to have different image size, but it is only the grid size, the image size remains to be(-0.5, 0.5; -0.5, 0.5)

Cached images are event-level pT normalized:

```text
pixel[c,h,w] = sum_pT[c,h,w] / sum_pT[event]
sum over all channels and pixels = 1
```

No per-pixel mean subtraction or standard deviation scaling is applied.

**Output files:**

## Step 2 — Train the Beta-VAE

Every option below can also be given in a YAML file with `--config` (flags override the file); see `configs/betavae_default.yaml` and `configs/preprocess_default.yaml`.

### Using a pre-built cache

```bash
python code_training/train_betavae.py \
    --cache-dir /path/to/cache \
    --epochs    300 \
    --batch-size 256 \
    --lr        1e-3 \
    --latent-dim 128 \
    --loss-type H \
    --beta      64 \
    --output    results/betavae/best_model.pth

Or
python code_training/train_betavae.py --cache-dir path/to/cache --epochs 300 --batch-size 256 --lr 1e-3 --latent-dim 128 --loss-type H --beta 64 --output results/betavae/best_model.pth
```

For beta annealing, use loss type `A`. This keeps the KLD term unchanged and
linearly ramps the multiplier from `--beta-start` to `--beta`:

```bash
python code_training/train_betavae.py \
    --cache-dir /path/to/cache \
    --epochs 300 \
    --batch-size 256 \
    --lr 1e-3 \
    --latent-dim 128 \
    --loss-type A \
    --recon-loss-weight 1.0 \
    --beta-start 0.0 \
    --beta 1.0 \
    --beta-anneal-iter 60000 \
    --beta-anneal-schedule linear \
    --output results/betavae/best_model.pth
```

### Or Using raw `.dat` files directly (on-the-fly image building)

```bash
python code_training/train_betavae.py \
    --g-jet  datas/g_jet_real_dynamic_sample.dat \
    --u-jet  datas/u_jet_real_dynamic_sample.dat \
    --epochs 300 \
    --batch-size 128 \
    --lr     1e-3 \
    --latent-dim 128 \
    --loss-type B \
    --output results/betavae/best_model.pth
```

### All Training Arguments


| Argument          | Default                  | Description                                                               |
| ----------------- | ------------------------ | ------------------------------------------------------------------------- |
| `--cache-dir`     | `None`                   | Path to dense pre-built cache. If set,`--g-jet`/`--u-jet` are not needed. |
| `--g-jet`         | `None`                   | Path to gluon jet`.dat` file                                              |
| `--u-jet`         | `None`                   | Path to quark jet`.dat` file                                              |
| `--epochs`        | `200`                    | Maximum number of training epochs                                         |
| `--batch-size`    | `64`                     | Mini-batch size                                                           |
| `--lr`            | `1e-3`                   | Initial learning rate (Adam)                                              |
| `--latent-dim`    | `128`                    | Size of the latent space**z**                                             |
| `--hidden-dims`   | `32,64,128,256,512`      | CNN channel sizes; number of values controls the number of conv layers    |
| `--img-size`      | `64`                     | Spatial resolution of jet images; must be divisible by `2^len(hidden_dims)` |
| `--loss-type`     | `B`                      | `H` = Higgins beta-VAE; `B` = Burgess capacity-annealing; `A` = beta annealing |
| `--recon-loss-type` | `mse`                  | Reconstruction loss: `mse`, `l1`, `sqrt_mse`, or `log_mse`                  |
| `--beta`          | `1.0`                    | Final beta weight on KL term for `H` or `A`                               |
| `--recon-loss-weight` | `1.0`                | Multiplier on reconstruction loss before adding the KL term               |
| `--gamma`         | `1000.0`                 | γ weight (used with`--loss-type B`)                                      |
| `--max-capacity`  | `25.0`                   | Maximum information capacity C (nats, used with`--loss-type B`)           |
| `--cap-max-iter`  | `100000`                 | Iterations over which capacity ramps up to`max-capacity`                  |
| `--beta-start`    | `0.0`                    | Initial beta for `--loss-type A`                                          |
| `--beta-anneal-iter` | `100000`              | Iterations over which beta ramps from `beta-start` to `beta`              |
| `--beta-anneal-schedule` | `linear`           | `linear` or `log_linear` beta schedule for `--loss-type A`                |
| `--patience`      | `20`                     | Early-stopping patience (epochs without improvement)                      |
| `--sample-events` | `0`                      | Max events per class (0 = use all) — on-the-fly mode only                |
| `--n-generate`    | `64`                     | Number of jet images to generate after training                           |
| `--output`        | `best_model_betavae.pth` | Path to save the best model checkpoint                                    |

### Training Outputs

After training, the following files are saved next to `--output`:


| File                      | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `best_model_betavae.pth`  | Model weights + metadata (epoch, val_loss, PID map, image stats) |
| `*_training_curves.png`   | Loss / reconstruction loss / KLD curves for train and val sets   |
| `*_generated.npy`         | Generated jet images, shape`(n_generate, C, H, W)`               |
| `*_generated_samples.png` | Visual grid: total pT image + dominant-channel image per sample  |

---

## Beta-VAE Model (`betaVae.py`)

The `BetaVAE` class implements a convolutional VAE with two flavours of the β-VAE objective.

### Architecture

```
Input [N, C, H, W]
  │
  ▼  Encoder
  5× Conv2d(stride=2) + BatchNorm + LeakyReLU     → spatial: H/32 × W/32
  │
  ├─► fc_mu  → μ  (latent_dim,)
  └─► fc_var → log σ²  (latent_dim,)
         │
         ▼  Reparameterization: z = μ + ε·σ
         │
  ▼  Decoder
  Linear → reshape [N, 512, H/32, W/32]
  4× ConvTranspose2d(stride=2) + BatchNorm + LeakyReLU
  1× ConvTranspose2d + BN + LeakyReLU + Conv2d
  Softplus + per-image sum normalization → [N, C, H, W], non-negative, sum = 1
```

- **Default hidden dims:** `[32, 64, 128, 256, 512]`
- **Image size constraint:** must be divisible by 2^len(hidden_dims).

### Loss Functions

For all loss modes, the reconstruction term is multiplied by
`recon_loss_weight` before the KL/capacity term is added.

**Type `H` — Higgins beta-VAE:**

```
L = reconstruction_MSE + β · (current_batch_size / configured_batch_size) · KLD
```

**Type `B` — Burgess capacity annealing:**

```
C(t) = min(C_max, C_max / C_stop_iter · t)
L = reconstruction_MSE + γ · (current_batch_size / configured_batch_size) · |KLD − C(t)|
```

**Type `A` - beta annealing:**

```text
beta(t) = beta_start + (beta - beta_start) * min(t / beta_anneal_iter, 1)
L = recon_loss_weight * reconstruction_loss + beta(t) * (current_batch_size / configured_batch_size) * KLD
```

For log-linear beta annealing, use `--beta-anneal-schedule log_linear`:

```text
s = min(t / beta_anneal_iter, 1)
beta(t) = beta_start * (beta / beta_start) ** s
```

This requires `beta_start > 0` and `beta > 0`.

The reconstruction loss is computed between event-level pT-normalized images, then multiplied by `--recon-loss-weight` before the KL term is added. Use `--recon-loss-type l1` for an absolute-error reconstruction term, `sum(|x - xhat|)` per image.

---

## Loading a Saved Model

```python
import torch
from betaVae import BetaVAE

checkpoint = torch.load("best_model_betavae.pth", map_location="cpu")

model = BetaVAE(
    in_channels=checkpoint['n_channels'],
    latent_dim=checkpoint['latent_dim'],
    image_size=checkpoint['img_size'],
    hidden_dims=checkpoint.get('hidden_dims', [32, 64, 128, 256, 512]),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Metadata stored in the checkpoint:
print(checkpoint['val_loss'])        # best validation loss
print(checkpoint['epoch'])           # epoch at which it was saved
print(checkpoint['pid_channel_map']) # PID → channel mapping
print(checkpoint['normalization'])    # event_pt
```

##
