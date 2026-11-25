# Reproducing the paper figures

The scripts below correspond to Figs. 2-5 of the current manuscript.

| Figure | Script | Needs |
|---|---|---|
| Fig. 2: input, particle reconstruction and image reconstruction | `fig2_reconstruction.py` | quark and gluon reference samples, both checkpoints |
| Fig. 3: posterior correlations and three joint log-variance distributions per model | `fig3_posterior_structure.py` | the same reference samples, both checkpoints |
| Figs. 4 and 5: single-coordinate and collective parameter responses | `fig4_5_parameter_response.py` | the Pythia scan files, both checkpoints |

## Inputs and commands

Run these commands from the repository root. Install the dependencies with
`pip install -e .` and `pip install -r cnn_vae/requirements.txt`.

The samples are on Hugging Face: [Shayla216/jet-vae-samples](https://huggingface.co/datasets/Shayla216/jet-vae-samples).
Download them with `hf download Shayla216/jet-vae-samples --repo-type dataset --local-dir data/`.
The commands assume that directory layout; replace the checkpoint paths with your local files.

- Image-level model: no-PID CNN32, effective beta = 2e-6, checkpoint selected at epoch 739.
  Its configuration is [betavae_paper.yaml](../cnn_vae/configs/betavae_paper.yaml).
- Particle-level model: no-PID Transformer32, beta = 1e-5, checkpoint selected at epoch 134.

```bash
python paper_figures/fig2_reconstruction.py \
    --u-jet data/training_jets/u_jet.dat --g-jet data/training_jets/g_jet.dat \
    --cnn-checkpoint cnn_best_model.pth --transformer-checkpoint transformer_best_model.pth \
    --out figs/fig2_reconstruction.pdf

python paper_figures/fig3_posterior_structure.py \
    --u-jet data/training_jets/u_jet.dat --g-jet data/training_jets/g_jet.dat \
    --cnn-checkpoint cnn_best_model.pth --transformer-checkpoint transformer_best_model.pth \
    --out figs/fig3_posterior_structure.pdf

python paper_figures/fig4_5_parameter_response.py \
    --scan-dir data/pythia_scan/200GeV_process \
    --cnn-checkpoint cnn_best_model.pth --transformer-checkpoint transformer_best_model.pth \
    --out-dir figs/
```

The last command produces `fig4_alpha_response.pdf` and `fig5_parameter_responses.pdf`.

## Figure settings

- **Fig. 2:** file-local, zero-based event indices 45050 (quark) and 77395 (gluon),
  selected with `--u-index` and `--g-index`. The three rows show the input, particle-level
  reconstruction and image-level reconstruction. Both reconstructions use deterministic
  `decode(mu)`. CPU inference is the default used for the paper's maps.
- **Fig. 3:** 5000 jets per class from the reference training samples, selected with seed 42
  (`--n-per-class`, `--seed`). Both models use the same event indices. The correlation matrices
  include all active dimensions: 32 particle-level and 29 image-level. Each model also displays
  the same three original-index pairs: (1, 2), (16, 20), (3, 29). The image collective-width
  statistic excludes indices 0, 11, 17, 30, leaving 25 dimensions; the particle statistic uses all 32.
  The relative loading spreads are approximately 0.1020 (image) and 0.0723 (particle).
- **Fig. 4:** original latent index j=1 for both models (`--latent-index`). The curves connect
  the five measured shifts, with a shared nominal point at zero.
- **Fig. 5:** all 29 image-level and 32 particle-level active dimensions appear in the response
  curves. For the image alpha_s summary, four weak-response dimensions (0, 11, 17, 30) are excluded
  using `--weak-threshold 0.1`; the other summaries use all active dimensions. Fits use a free
  intercept and generalized least squares on the four nonzero variations, with fixed equal class
  weights and shared-nominal covariance. Error bars show population standard deviation across
  coordinate slopes. The alpha_s mean slopes are approximately -1.379752 (25 image dimensions)
  and -0.970129 (32 particle dimensions).

## Scan files and cache

The response script reads `{u_jet,g_jet}/nominal.npz` and
`{u_jet,g_jet}/<parameter>/{minus_20pct,minus_10pct,plus_10pct,plus_20pct}.npz`.
The eight parameters are `alphaS`, `aLund`, `bLund`, `sigma`, `pTmin`, `mesonUDvector`,
`probStoUD` and `probQQtoQ`. Each file contains 10 000 jets and is checked against its metadata.

The per-setting means and Monte Carlo variances are saved in `scan_means.npz`.
The cache records checkpoint hashes and input-file identities; use `--recompute` when these change.
`--device cuda` enables GPU encoding. The `pTmin` label denotes the generator input rather than
its internal effective shower cutoff.

Reference samples may also be supplied as `.npz` files with `particles80`, `mask80`, `p4`, `pid`
and `mask` arrays. Replacing the reference training samples with a different sample changes the
posterior statistics and does not reproduce the same Fig. 3.
