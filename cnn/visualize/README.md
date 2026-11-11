# `feature/vae-visualize`

This branch adds interactive Streamlit viewers for two jet VAE models in this repository:

- `vae_DNN/visualize/`: latent-space explorer for the MLP constituent-feature beta-VAE
- `vae_cnn/visualize/`: latent-space explorer for the CNN jet-image beta-VAE

The visualization code is intended for checkpoint inspection, latent-dimension probing, reconstruction comparisons, and quick qualitative checks on real jets.

## Repository Layout

```text
jet/
├── README.md
├── vae_cnn/
│   ├── 20260508_021610/
│   ├── bvae_beta_b001_20260508_021616.log
│   └── visualize/
│       ├── app.py
│       └── core.py
└── vae_DNN/
    ├── README.md
    ├── code_training/
    └── visualize/
        ├── app.py
        └── core.py
```

## What This Branch Adds

### `vae_DNN` explorer

The MLP VAE viewer exposes:

- baseline latent selection from `z = 0`, random prior sample, or an encoded real jet
- single-dimension latent sweeps
- manual latent slider control across all dimensions
- impact heatmaps showing which latent dimensions most affect jet observables

The generated jet is shown as an `eta-phi` constituent scatter plot with PID coloring and basic jet observables such as jet `pT`, jet mass, multiplicity, and girth.

### `vae_cnn` explorer

The CNN VAE viewer exposes:

- baseline decoded jet image inspection
- latent sweeps and manual latent control
- dimension-impact heatmaps on image-level observables
- real cached image browsing and reconstruction comparison
- raw `.dat` event browsing, image conversion, and reconstruction comparison
- active-unit analysis from encoded samples
- training-curve viewing from the checkpoint log

The CNN app works directly with multi-channel jet images and provides both total-`pT` views and per-channel grids.

## Requirements

You need a Python environment with at least:

- `streamlit`
- `numpy`
- `torch`
- `matplotlib`

The existing model-specific dependencies in `vae_DNN/requirements.txt` may still be needed for loading checkpoints or dataset utilities.

## Running The Apps

### 1. MLP beta-VAE explorer

From `vae_DNN/`:

```bash
streamlit run visualize/app.py --server.port 8501 --server.headless true
```

To preload a checkpoint:

```bash
streamlit run visualize/app.py --server.port 8501 -- \
  --checkpoint results/betavae/other_run/best_model.pth
```

Notes:

- `--model-module-dir` is optional and can be used if `betaVae.py` is not auto-detected correctly.
- the "encode a real jet" baseline expects `.dat` event files in the same format used by training

### 2. CNN beta-VAE explorer

From `vae_cnn/`:

```bash
streamlit run visualize/app.py -- \
  --checkpoint 20260508_021610/best_model_betavae_cone05_20260508_021610.pth
```

Or from anywhere:

```bash
streamlit run /absolute/path/to/vae_cnn/visualize/app.py -- \
  --checkpoint /absolute/path/to/checkpoint.pth
```

To enable the data-driven tabs, you may also pass:

```bash
--cache-dir /path/to/cache_jet_images_32_real
```

## Remote Usage

If you run Streamlit on a remote host, forward port `8501`:

```bash
ssh -L 8501:localhost:8501 <user>@<host>
```

Then open:

```text
http://localhost:8501
```

## Expected Inputs

### `vae_DNN`

- a `.pth` checkpoint from the constituent-feature beta-VAE
- optionally, `.dat` jet files for "encode a real jet"

### `vae_cnn`

- a `.pth` checkpoint from the CNN beta-VAE
- optionally, a cached image directory for real-image browsing
- optionally, raw `.dat` event files for the raw-event tabs
- optionally, the training log stored near the checkpoint for the curve viewer

## Validation

The visualization modules in this branch were syntax-checked with:

```bash
python -m py_compile \
  vae_cnn/visualize/app.py \
  vae_cnn/visualize/core.py \
  vae_DNN/visualize/app.py \
  vae_DNN/visualize/core.py
```

## Related Docs

- `vae_DNN/README.md`: training, data format, latent analysis, and generation for the MLP baseline
- `vae_DNN/visualize/app.py`: UI entry point for the MLP explorer
- `vae_cnn/visualize/app.py`: UI entry point for the CNN explorer
