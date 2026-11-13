# Reproducing the paper figures

| Figure | Script | Needs |
|---|---|---|
| Fig. 2: input vs. reconstruction, both VAEs | `fig2_reconstruction.py` | quark and gluon reference `.dat` files, both checkpoints |
| Fig. 3: correlation of `mu_j` and `ln sigma_j^2` | `fig3_posterior_structure.py` | reference jets (`.dat` or `.npz`), Transformer checkpoint |
| Fig. 4: joint distributions of `ln sigma^2` pairs | `fig4_logvar_pairs.py` | same as Fig. 3 |
| Figs. 5, 6: response of `sigma`-space to Pythia parameters | `fig5_6_parameter_response.py` | the Pythia scan files (below), both checkpoints |

The samples are on Hugging Face ([Shayla216/jet-vae-samples](https://huggingface.co/datasets/Shayla216/jet-vae-samples)); download them with
`hf download Shayla216/jet-vae-samples --repo-type dataset --local-dir data/` (the commands below assume this `data/`
folder, relative to the repository root). Run from this directory (the scripts import `common.py`). Every script prints the numbers quoted in the text
next to the plot. Install the dependencies of both models plus `matplotlib`:
`pip install -e .` and `pip install -r cnn_vae/requirements.txt`.

```bash
python fig2_reconstruction.py --u-jet ../data/training_jets/u_jet.dat --g-jet ../data/training_jets/g_jet.dat \
    --cnn-checkpoint cnn_best_model.pth --transformer-checkpoint transformer_best_model.pth --out fig2.pdf

python fig3_posterior_structure.py --u-jet ../data/training_jets/u_jet.dat --g-jet ../data/training_jets/g_jet.dat \
    --transformer-checkpoint transformer_best_model.pth --out fig3.pdf     # prints active dims and spread of b_j
python fig4_logvar_pairs.py        --u-jet ../data/training_jets/u_jet.dat --g-jet ../data/training_jets/g_jet.dat \
    --transformer-checkpoint transformer_best_model.pth --out fig4.pdf

python fig5_6_parameter_response.py --scan-dir ../data/pythia_scan/200GeV_process \
    --cnn-checkpoint cnn_best_model.pth --transformer-checkpoint transformer_best_model.pth --out-dir figs/
```

- **Fig. 2** uses event 4 of the quark file and event 77395 of the gluon file (`--u-index`, `--g-index`).
- **Figs. 3 and 4** draw 5000 jets per class (`--n-per-class`, `--seed 42`). The nominal files of the scan
  (`pythia_scan/200GeV_process/{u_jet,g_jet}/nominal.npz`) can be used instead of the `.dat` files.
- **Figs. 5 and 6** read `{u_jet,g_jet}/<parameter>/{minus_20pct,minus_10pct,plus_10pct,plus_20pct}.npz`
  and `{u_jet,g_jet}/nominal.npz` (8 parameters `alphaS, aLund, bLund, sigma, pTmin, mesonUDvector, probStoUD,
  probQQtoQ`; 10 000 jets per file). Each file is checked against its own metadata. The 33 per-setting means are cached in
  `scan_means.npz`, so replotting is instant (`--recompute` to redo). A GPU makes the encoding much faster
  (`--device cuda`).
- Checkpoints: the particle-level VAE is the epoch-134 no-PID model; the image-level VAE is the 32-dimensional
  model whose settings are in `cnn_vae/configs/betavae_paper.yaml`. They will be released on Hugging Face.

Reference values printed by the scripts for the released checkpoints: 32 active dimensions and a relative spread of
the scaling coefficients `b_j` of about 0.072 (Fig. 3); 30 (image-level) and 32 (particle-level) active dimensions and
mean `alpha_s` slopes of about -1.09 and -0.97 (Fig. 6).
