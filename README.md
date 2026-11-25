# jet-vae

Variational autoencoders for particle jets (quark vs. gluon).

- **`transformer_vae/`** – a Transformer VAE on shape-only point sets `(z, eta, phi)` with a Sinkhorn-EMD
  reconstruction loss and no particle-ID input.
- **`cnn_vae/`** – a CNN VAE on jet images (see also `cnn_vae/README.md`).

## Paper checks and reproduction

[Paper requirements](PAPER_REQUIREMENTS.md) tracks the remaining checks, exact-run provenance,
numerical source data and reproduction commands for the paper.

## Data

The jet samples are on Hugging Face: [Shayla216/jet-vae-samples](https://huggingface.co/datasets/Shayla216/jet-vae-samples).

- `training_jets/u_jet.dat`, `training_jets/g_jet.dat`: 100 000 quark and 100 000 gluon jets (the `.dat` format
  read by the Transformer VAE preprocessing).
- `pythia_scan/200GeV_process/`: the Pythia parameter scan (10 000 quark and 10 000 gluon jets for each of 33 settings).

```bash
pip install huggingface_hub
hf download Shayla216/jet-vae-samples --repo-type dataset --local-dir data/
```

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

## Paper figures

`paper_figures/` contains the reproduction scripts for current manuscript Figs. 2-5.
See [the figure instructions](paper_figures/README.md) for commands and expected inputs.
The current appendix RAMBO and observable-correlation figures are in
[paper_figures/Appendix/](paper_figures/Appendix/README.md).

| Figure | Script | Inputs |
|---|---|---|
| Fig. 2, input and reconstructions (three rows, two jet columns) | `fig2_reconstruction.py` | quark / gluon reference jets, both checkpoints |
| Fig. 3, posterior correlations and joint log-variance distributions (both VAEs) | `fig3_posterior_structure.py` | reference jets, both checkpoints |
| Figs. 4 and 5, single-coordinate and collective parameter responses | `fig4_5_parameter_response.py` | Pythia scan files (8 parameters x 4 variations + nominal, 10 000 quark and 10 000 gluon jets each), both checkpoints |

The jet samples are on Hugging Face (see [Data](#data)); the trained checkpoints are not part of this repository. The
scripts take their paths on the command line. The settings of the image-level VAE used in the paper are in `cnn_vae/configs/betavae_paper.yaml`.

## Supplementary model comparison

[model_comparison/](model_comparison/README.md) compares five CNN32 and seven
particle Transformer32 checkpoints across beta settings, with shared dijet samples
and RAMBO controls. It includes [a topic-grouped report](model_comparison/beta_comparison.pdf),
individual figures, numerical plot inputs, checkpoint metadata and reproduction scripts.
The two particle beta=1e-5 training runs are listed separately; current-paper
checkpoints are explicitly marked. This collection has its own topic names and
is separate from the manuscript figure numbering.

## Non-jet benchmarks

The paper compares the posterior-width structure of the jet VAEs with that of VAEs trained on four standard
image datasets. The datasets are not redistributed here; each is subject to the terms of its source page
(CelebA is available for non-commercial research purposes only).

| Dataset | Content | Image used | Source |
|---|---|---|---|
| MNIST | handwritten digits, 60 000 training and 10 000 test images (28 × 28) | 1 × 32 × 32 (resized) | LeCun, Bottou, Bengio, Haffner, *Proc. IEEE* **86**, 2278 (1998), [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) |
| FashionMNIST | clothing items in 10 classes, 60 000 training and 10 000 test images (28 × 28) | 1 × 32 × 32 (resized) | Xiao, Rasul, Vollgraf, [arXiv:1708.07747](https://arxiv.org/abs/1708.07747), [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) |
| CelebA | 202 599 celebrity face photographs with 40 binary attributes | 3 × 64 × 64 | Liu, Luo, Wang, Tang, *Proc. ICCV* (2015), [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| dSprites | 737 280 binary images of 2D shapes, generated from five factors (shape, scale, orientation, x and y position) | 1 × 64 × 64 | Matthey, Higgins, Hassabis, Lerchner (2017), [github.com/deepmind/dsprites-dataset](https://github.com/deepmind/dsprites-dataset/) |

The code is [YannDubs/disentangling-vae](https://github.com/YannDubs/disentangling-vae) (Dubois, Kastanos, Lines,
Melman, 2019; MIT licence), used unmodified; it is not copied into this repository. In the paper table, the rows
marked R use the pre-trained models that ship in the `results/` directory of that repository, and the rows marked O
are models we trained ourselves with the same code. If you use these benchmarks, please also cite the datasets above
and disentangling-vae:

```bibtex
@misc{dubois2019dvae,
  title        = {Disentangling VAE},
  author       = {Dubois, Yann and Kastanos, Alexandros and Lines, Dave and Melman, Bart},
  month        = {march},
  year         = {2019},
  howpublished = {\url{http://github.com/YannDubs/disentangling-vae/}}
}
```

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
