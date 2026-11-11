# jet-vae

Variational autoencoders for particle jets (quark vs. gluon).

- **`transformer_vae/`** – a Transformer VAE on shape-only point sets `(z, eta, phi)` with a Sinkhorn-EMD
  reconstruction loss and no particle-ID input.
- **`cnn_vae/`** – a CNN VAE on jet images (see also `cnn_vae/README.md`).

## Install

```bash
pip install -e .                       # Transformer VAE: torch, numpy, scikit-learn, pyyaml
pip install -r cnn_vae/requirements.txt    # CNN VAE
```

## Transformer VAE

```
.dat files ──preprocess──▶ cache/*.npy ──train──▶ best_model.pth ──encode──▶ latents.npz
```

Input `.dat` files: one jet per `<event_begin>:id … <event_end>:id` block, one particle per row
`pid px py pz E`. Quark jets are label 0, gluon jets label 1.

**1. Preprocess.**

```bash
python -m transformer_vae.data.preprocess --u-jet u_jets.dat --g-jet g_jets.dat --cache-dir cache/
```

Writes `{train,val}_{particles,mask,category,labels}.npy` and `meta.json`. The 85/15 split is stratified
by label (`--random-state 42`) and is fixed by the cache; the training `--seed` does not change it.

**2. Train.**

```bash
python -m transformer_vae.train --config transformer_vae/configs/nopid_beta1e-5.yaml \
    --cache-dir cache/ --output runs/s1/best_model.pth --seed 1
```

All hyperparameters (β, the Sinkhorn ε and iteration annealing, model size, optimiser) are set in
`transformer_vae/configs/nopid_beta1e-5.yaml`; `transformer_vae/configs/nopid_beta1e-4.yaml` is the same with β = 1e-4. A run saves
`best_model.pth` (best validation loss), `best_model_last.pth` (resume state, use `--resume`),
`best_model_history.json`, `best_model_config.json` and some generated jets.

**3. Encode.**

```bash
python -m transformer_vae.encode --checkpoint runs/s1/best_model.pth \
    --cache-dir cache/ --split val --out latents_val.npz     # mu, logvar, labels
```

or in Python: `from transformer_vae.encode import load_model, encode`.

## CNN VAE

```
.dat files ──preprocess_jet_images.py──▶ image cache ──train_betavae.py──▶ best_model.pth
```

Same `.dat` input format. Use `preprocess_jet_images_no_pid.py` for a single-channel (no-pid) image.

```bash
python cnn_vae/code_training/preprocess_jet_images.py --config cnn_vae/configs/preprocess_default.yaml \
    --g-jet g_jets.dat --u-jet u_jets.dat --cache-dir cache_images/

python cnn_vae/code_training/train_betavae.py --config cnn_vae/configs/betavae_default.yaml \
    --cache-dir cache_images/ --output results/betavae/best_model.pth
```

All hyperparameters (the KL / annealing mode `loss-type` and its β, γ, capacity settings, the
reconstruction loss, latent size, optimiser) are set in `cnn_vae/configs/betavae_default.yaml`, which writes out
the defaults of `train_betavae.py`; the image window and size are in `cnn_vae/configs/preprocess_default.yaml`.
Command-line flags override the YAML. Loading a saved model and the analysis notebooks are described in
`cnn_vae/README.md`.

## Normalization

**Transformer VAE (point set).** Per jet:
- constituents are sorted by pT (descending) and truncated to 80; padded slots are masked;
- `z = pT / Σ pT` over the kept constituents, so the `z` values of a jet sum to 1;
- `(eta, phi)` are shifted so the jet's massless vector-sum axis is at `(0, 0)`, with `phi` wrapped to
  (−π, π]; the energy `E` in the file is not used;
- the decoder reproduces both constraints by construction: `z` comes from a softmax over the decoder
  slots, and the decoded `(eta, phi)` are re-centred on the decoded jet's own axis.

**CNN VAE (image).** Per jet:
- `(eta, phi)` are shifted to the jet axis, and particles inside `|Δη| < eta_max`, `|Δφ| < phi_max` are
  histogrammed by pT into an image (one channel per particle ID, or a single channel for no-pid);
- the image is divided by its total pT, so all pixels sum to 1; particles outside the window are dropped
  before this normalization;
- the decoder output goes through softplus, is divided by its sum (so it also sums to 1) and is
  re-centred on its own axis; the reconstruction loss (default MSE) is computed between these
  normalized images.

## Weights

Trained weights will be released on Hugging Face (link to be added).

## License

MIT, see `LICENSE`.
