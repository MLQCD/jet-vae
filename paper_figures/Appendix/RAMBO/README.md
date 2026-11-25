# RAMBO: current Figs. 9 and 10

`rambo.py` implements massless RAMBO with equal event weights at fixed N.
The algorithm is due to [Kleiss, Stirling and Ellis](https://doi.org/10.1016/0010-4655(86)90119-0).
Four-vectors are float64 `(E, px, py, pz)` in GeV. Events have labeled particles;
no identical-particle symmetry factor is included in the phase-space volume.

The paper uses invariant mass 20 GeV, total pT 200 GeV, rapidity 0 and azimuth 0:
`P=(sqrt(200^2+20^2),200,0,0)`. No shower, clustering or angular selection is applied.
All generated constituents are passed to the frozen no-PID particle encoder;
the paper samples never exceed 80 particles, so no truncation occurs.

## Generate four-vectors only

```bash
pip install -r paper_figures/Appendix/RAMBO/requirements.txt
python paper_figures/Appendix/RAMBO/rambo.py \
    --mass 20 --pt 200 --n-particles 2 20 80 --n-events 1000 \
    --seed 20261115 --output appendix_output/rambo_events
```

Each multiplicity gets an NPZ sample and JSON settings/kinematic validation.
Optional `--write-dat` writes `pid px py pz E` rows with placeholder PID 0.
These generated NPZ files use **E first**, unlike the repository's QCD scan
NPZ files. Use the RAMBO inference scripts below for the no-PID preprocessing;
PID 0 should not be sent to the QCD species-category reader.

## Regenerate Fig. 9

Install the inference dependencies and supply the checkpoint described in
[the appendix README](../README.md). From the repository root:

```bash
python paper_figures/Appendix/RAMBO/generate_scan.py \
    --checkpoint transformer_best_model.pth --device cuda \
    --out-dir appendix_output/rambo_scan
python paper_figures/Appendix/RAMBO/fig9_multiplicity_comparison.py \
    --data appendix_output/rambo_scan/fig9_distributions.json
```

This generates 1,000 events for each N=2,...,80 (79,000 total), using
`SeedSequence([20261115,N])`. The QCD-derived basis is held fixed:
`u(x) = (logvar_active-a_QCD) @ b_QCD / (b_QCD @ b_QCD)`.
The basis is the leading covariance principal component of centered QCD log
variances, normalized to `mean(b)=1`; it is not refitted or recentered on RAMBO.
The default basis and QCD distributions are included under `data/`.
To use a rebuilt reference, add
`--basis appendix_output/reference/reference_basis.json`
and `--qcd-data appendix_output/reference/qcd_distributions.json`.

Density panels use integer-N bins and 75 width bins from -1.5 to 6,
normalized as the fraction of all sample events per 2D bin. The displayed range
is -1.5 to 4, with a logarithmic color scale from 1e-5 to 1e-2.
Profiles require at least 20 events at a given N. The displayed mean uses a
SEM-weighted cubic smoothing spline (`w=1/SEM`, `s=0.6 * number_of_bins`),
without extrapolation. The bands retain the unsmoothed 16th--84th event
percentiles; they are not uncertainty bands on the mean. The JSON includes
unsmoothed means, SEMs, bands and exact histogram counts.

## Regenerate Fig. 10

```bash
python paper_figures/Appendix/RAMBO/generate_matched.py \
    --checkpoint transformer_best_model.pth --device cuda \
    --out-dir appendix_output/rambo_matched
python paper_figures/Appendix/RAMBO/fig10_multiplicity_response.py \
    --data appendix_output/rambo_matched/fig10_response.json
```

Five fresh ensembles match the entire empirical multiplicity distributions
of the QCD alpha_s settings (-20%, -10%, nominal, +10%, +20%). Each QCD target
starts with 10,000 quark and 10,000 gluon jets. Remove N=1 and retain every
remaining integer-N count exactly. This gives 99,975 RAMBO events altogether.
The alpha_s values label the QCD targets; RAMBO itself has no alpha_s parameter.

Use `SeedSequence([20261116,3906,group_index,N])`, with group_index=0,...,4.
The plotted responses are each sample's mean log variance minus the newly
generated nominal-target mean, for all 32 coordinates. No PCA is fitted to
these ensembles. The x coordinates are the actual post-cut mean N; all curves
pass through zero at nominal index 2. The plot retains the paper's vertical
limits +/-0.2983057705650325.

The exact target counts are included in `data/multiplicity_targets.json`.
They can also be reconstructed from the existing dijet scan inputs:

```bash
python paper_figures/Appendix/RAMBO/prepare_multiplicity_targets.py \
    --scan-dir data/pythia_scan/200GeV_process \
    --out appendix_output/multiplicity_targets.json
```

Pass this file to `generate_matched.py --targets ...`. The reader uses the full
constituent mask, checks 10,000 jets per class and N<=80, and never uses padding
or a truncated multiplicity as the target.

## Validation

```bash
python -m unittest discover -s paper_figures/Appendix/RAMBO/tests -v
python -m unittest discover -s paper_figures/Appendix/tests -v
```

Generator tests cover conservation, the mass shell, analytic energy spectra,
rotational symmetry, boosts, random streams, phase-space volume and file round trips.
