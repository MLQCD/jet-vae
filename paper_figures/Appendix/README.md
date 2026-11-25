# Current appendix figures

This directory reproduces the current manuscript's Figs. 9--11.

| Figure | Script | Content |
|---|---|---|
| 9 | [RAMBO/fig9_multiplicity_comparison.py](RAMBO/fig9_multiplicity_comparison.py) | Common posterior width versus multiplicity for quark jets, gluon jets and RAMBO |
| 10 | [RAMBO/fig10_multiplicity_response.py](RAMBO/fig10_multiplicity_response.py) | RAMBO responses with full QCD multiplicity distributions matched |
| 11 | [fig11_observable_correlations.py](fig11_observable_correlations.py) | 32 posterior means and log variance 0 versus nine physical observables |

## Redraw the published numerical results

Run from the repository root. These commands use the included small JSON plot
inputs and need neither a checkpoint nor raw events. Each saves a PDF and PNG.

```bash
pip install -r paper_figures/Appendix/requirements.txt
python paper_figures/Appendix/RAMBO/fig9_multiplicity_comparison.py
python paper_figures/Appendix/RAMBO/fig10_multiplicity_response.py
python paper_figures/Appendix/fig11_observable_correlations.py
```

Outputs go to `figs/`. Use `--data FILE.json --out FILE.pdf` for regenerated
inputs or a different destination. The JSON files contain integer histogram
counts, conditional statistics, response means, or correlations, not trained
weights or event-level samples. [data/provenance.json](data/provenance.json)
records the source-array and plot-data hashes. Plot-data hashes normalize
line endings to LF so that validation works on Windows, Linux and macOS.

## Recompute the shared QCD reference and Fig. 11

For inference, also install the existing repository dependencies:
`pip install -e .` and `pip install -r cnn_vae/requirements.txt`.
Use the same raw reference samples as Fig. 3 and the no-PID particle Transformer32
checkpoint, beta=1e-5, epoch 134. The checkpoint SHA256 is
`e8e59c4f836b2c1df728a3970439ecde1db3f889208556a4b4eaccc7f24a42b2`.
The scripts check this identity before inference. Supply your local checkpoint.

```bash
python paper_figures/Appendix/prepare_reference.py \
    --u-jet data/training_jets/u_jet.dat --g-jet data/training_jets/g_jet.dat \
    --checkpoint transformer_best_model.pth --device cuda \
    --out-dir appendix_output/reference
python paper_figures/Appendix/fig11_observable_correlations.py \
    --data appendix_output/reference/fig11_correlations.json
```

The reference selects 5,000 events per class, with seed 42 and sorted file-local
indices, exactly as Fig. 3. It exports the reference posteriors, observables,
QCD width distributions, QCD PCA basis, and all 297 signed Pearson correlations.
The reference encoder batch size is 256; RAMBO uses 128.
GPU inference was used for the stored appendix data; `--device cpu` is supported
but small floating-point differences can occur across hardware/PyTorch versions.

Fig. 11 uses every positive-pT constituent for the observables, before the model's
top-80 truncation. The summed constituent four-momentum sets the jet axis and
mass. With `z_i=pT_i/sum(pT)`, `theta_i=DeltaR_iJ/0.4`, and
`theta_ij=DeltaR_ij/0.4`, the columns are:

| Observable | Definition |
|---|---|
| Multiplicity | Number of positive-pT constituents |
| pTD | `sqrt(sum(z_i^2))` |
| Entropy | `-sum(z_i log(z_i))` |
| LHA | `sum(z_i sqrt(theta_i))` |
| Girth | `sum(z_i theta_i)` |
| Second angularity | `sum(z_i theta_i^2)` |
| e2, exponent 1/2 | `sum(i<j, z_i z_j sqrt(theta_ij))` |
| e2, exponent 1 | `sum(i<j, z_i z_j theta_ij)` |
| Log mass squared | `log(m_J^2 / GeV^2)` |

The calculation retains the numerical conventions used for the paper:
`log(z+1e-12)` in entropy and `log(max(mass2,0)+1e-12)` in GeV units.
The correlation uses `mu_0,...,mu_31,logvar_0`, directly from the encoder;
log variance is not logged again. The figure has no horizontal-axis title.
It uses Times New Roman when installed, with STIXGeneral as a portable fallback.

See [RAMBO/README.md](RAMBO/README.md) for generator, matching and inference commands.
