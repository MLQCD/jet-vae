# Model comparison across beta settings

This collection compares five image-level CNN32 checkpoints and seven particle-level
Transformer32 checkpoints. It is a supplementary checkpoint study. The current
manuscript figures remain in [paper_figures/](../paper_figures/README.md).

- [Comparison report](beta_comparison.pdf): grouped by analysis topic, one model per page.
- [Individual figures and page index](figures/README.md).
- [Model registry](models.json): checkpoint hashes, representation, beta, seed, epoch and paper-reference status.
- [Numerical results](data/): small JSON plot inputs; weights and event-level caches are not included.
- [Validation](VALIDATION.md): cross-checks and hashes of the published numerical inputs.

## Model scope

| Representation | Beta | Seed | Selected epoch | Role |
|---|---|---|---|---|
| Image | 1e-6 | not recorded in checkpoint | 520 | comparison |
| Image | 2e-6 | not recorded in checkpoint | 739 | current paper |
| Image | 3e-6 | not recorded in checkpoint | 800 | comparison |
| Image | 4e-6 | not recorded in checkpoint | 750 | comparison |
| Image | 5e-6 | not recorded in checkpoint | 755 | comparison |
| Particle | 1e-5 | 42 | 134 | current paper |
| Particle | 1e-5 | 1 | 141 | comparison |
| Particle | 1e-4 | 1 | 199 | comparison |
| Particle | 2e-4 | 1 | 195 | comparison |
| Particle | 3e-4 | 1 | 199 | comparison |
| Particle | 4e-4 | 1 | 174 | comparison |
| Particle | 5e-4 | 1 | 194 | comparison |

The two particle beta=1e-5 checkpoints are separate training runs. The study does
not average over seeds or isolate beta as the sole cause of differences between
all checkpoints. Image beta means `checkpoint beta / reconstruction weight`;
particle beta is the checkpoint's coefficient of KL relative to Sinkhorn EMD.
The two representations have different reconstruction objectives, so equal beta
values would not represent equal regularization strength.

The present collection covers dijet and RAMBO results. It contains no Z+jet results
and no particle beta=1e-3 benchmark. Topics replace manuscript figure numbers.

## Redraw all figures without checkpoints

From the repository root:

```bash
pip install -r model_comparison/requirements.txt
python model_comparison/scripts/plot_all.py
```

This rebuilds `beta_comparison.pdf`, individual PDFs/PNGs in `figures/`, and a
page index. `--data-dir`, `--out-dir` and `--report` select alternative locations.
The plotting stage needs neither PyTorch nor event samples. The data tables retain
histogram counts, correlation matrices, response means/variances, selected indices,
reconstruction images, training curves and source hashes.

## Recompute a checkpoint

Install the existing inference dependencies with `pip install -e .` and
`pip install -r cnn_vae/requirements.txt`. Supply local checkpoint files matching
[models.json](models.json); download links for weights are outside this update.
The raw samples and scan layout are described in the [paper instructions](../paper_figures/README.md).

Prepare the shared reference once:

```bash
python model_comparison/scripts/prepare_inputs.py \
    --u-jet data/training_jets/u_jet.dat --g-jet data/training_jets/g_jet.dat \
    --out comparison_work/reference.npz
```

Then analyze any registered model, for example:

```bash
python model_comparison/scripts/analyze_model.py \
    --model particle_beta2e-4_seed1 --checkpoint /path/to/best_model.pth \
    --reference comparison_work/reference.npz \
    --scan-dir data/pythia_scan/200GeV_process \
    --work-dir comparison_work/particle_beta2e-4_seed1 \
    --out model_comparison/data/particle_beta2e-4_seed1.json --device cuda
```

Use a different work directory for each model. Intermediate posterior/statistic
caches are tied to checkpoint and reference hashes; scan caches also verify each
source file. No training is performed. CPU inference is supported; small
floating-point differences can occur across devices or PyTorch versions.

The files in `configs/` record the available checkpoint settings. They are
provenance records, not complete replacement training configurations; fields
missing from a checkpoint are not inferred.

## Definitions and comparison rules

**Reference and reconstruction.** The same 5,000 u and 5,000 g jets are selected
with seed 42 and sorted file-local indices, as in the paper. Examples use u index
45050 and g index 77395. Illustrated reconstructions use CPU `decode(mu)`.
The common pixel SSE is the unweighted sum of squared differences between 32x32
images, averaged over all 10,000 reference jets. Particle decodings are rendered
into the same image window, without renormalizing particles outside that window.
This diagnostic is separate from the models' training losses.

**Active coordinates and correlations.** Active means
`Var(mu_j) / mean(exp(logvar_j)) > 1`. All active coordinates appear in posterior
correlation matrices. A width coordinate is eligible for the joint-distribution
and observable panel when its signed correlation exceeds 0.8 with more than half
of the other active widths. Up to five disjoint pairs are spread evenly through
the eligible coordinates in original-index order. The observable panel uses the
first eligible width. If none qualifies, the panel shows means only and the
selection is recorded as null. Identical index numbers in separately trained
models do not identify the same physical coordinate.

**Parameter responses.** Each setting has 10,000 u and 10,000 g jets; the eight
Pythia parameters have -20%, -10%, nominal, +10%, +20% settings. Activity is measured
on the nominal scan mixture, independently of the reference sample. Fits reuse
the paper's free-intercept GLS on the four nonzero variations, with fixed 50:50
class weights and shared-nominal MC covariance. Error bars are population SD
across coordinate slopes, not uncertainty of the average. Only the image alpha_s
summary excludes coordinates with maximum absolute response below 0.1; all active
curves remain visible. Particle summaries use every active coordinate. The pTmin
label denotes the generator input, not the internal effective shower cutoff.

**Observable definitions.** Particle panels use the nine raw-constituent
observables in the [appendix](../paper_figures/Appendix/README.md), with all
positive-pT constituents and R_ref=0.4. Image panels use eight observables computed
from normalized pixels: pTD, entropy, three angularities, two energy correlators,
and log normalized image mass squared. Image mass is dimensionless and differs
from particle jet mass in GeV; image panels do not include particle multiplicity.
The code for pixel observables is in `scripts/shared.py`.

**RAMBO.** All seven particle checkpoints use the existing
[RAMBO generator](../paper_figures/Appendix/RAMBO/README.md), M=20 GeV and pT=200 GeV,
without shower, clustering or angular selection. The scan has 1,000 events per
N=2..80, seed sequence `[20261115,N]`. Five additional ensembles match the full
QCD multiplicity histograms after excluding N=1, with seed sequence
`[20261116,3906,group_index,N]`, totaling 99,975 events. The same events and target
counts are used for every model. Each model fits its own common-width covariance
PCA on QCD and keeps that basis fixed on RAMBO, with mean loading one. Widths are
not refitted or recentered on RAMBO. Profiles require at least 20 events per N;
means use SEM-weighted smoothing (s=0.6 times the bin count), while the 16th--84th
percentile event bands are unsmoothed. Common display ranges span the data of all
checkpoints; they may therefore differ from the manuscript's display limits.

**Training histories.** Particle curves use recorded per-epoch histories, including
the selected checkpoint marker. Image curves are approximate digitizations from
archived PNGs, explicitly labeled in every panel. Missing or unresolved digitized
values remain gaps. These curves are not exact epoch logs; reconstruction,
correlation and response results come from model inference, not digitization.

## Extending the comparison

Add a distinct model ID and checkpoint hash to `models.json`, record available
settings in `configs/`, and run the same analysis on the shared reference and scan.
Keep seed/checkpoint variants distinct even when their beta agrees. Add a matching
`data/<model_id>_training.json` with its source kind and recorded values or clearly
labeled digitization. `plot_all.py` automatically includes registered models.

```bash
python -m unittest discover -s model_comparison/tests -v
```
