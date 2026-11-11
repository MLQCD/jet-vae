# Paper checks and reproduction package

This paper presents correlations and approximate common scaling of VAE posterior variances, together with their response to generator-coupling variations. The tasks below connect the retained results to their exact runs and make the numerical evidence reproducible. Start from completed runs and saved outputs; identify a missing input before proposing additional computation.

Suggested coordination: Peiran assembles the paper package, and Sitong supplies and verifies the sample/model comparisons she performed. Record the producer of each result and agree on any remaining assignments together.

## Immediate paper checks

Complete these checks before freezing the numerical results. For each item, add the output path, producer and any remaining limitation here. A missing artifact should be reported explicitly rather than replaced with a similar run.

### 1. Identify the exact paper runs

- [ ] Add a figure-to-run manifest covering every retained figure and quoted numerical result. Include the original code commit, checkpoint location and checksum, sample and split identifiers, preprocessing settings, training configuration, and analysis/plotting command.
- [ ] Supply the exact CNN paper configuration with 32 latent dimensions. The repository's generic 128-dimensional configuration is not the paper recipe. Also identify the exact particle-model configuration; do not infer it from example defaults.
- [ ] Record the effective reconstruction and KL loss normalization, including reductions over events, pixels/particles and latent dimensions. State whether diagnostic samples contain training events.
- [ ] Record the active-dimension rule and indices for each model, determined from the default reference sample. Keep this activity definition separate from any subsequent selection by coupling response.

**Done when:** each paper result can be traced to a specific saved run and sample. Use this manifest to reconcile the model, training and sample descriptions in the methods and appendices.

### 2. Quantify the common-width pattern on those runs

- [ ] On the samples used in the correlation/scaling figures, export the active count, leading principal-component variance fraction, equal-loading-mode variance fraction, loading spread and a defined residual measure for each model.
- [ ] Include the script and numerical output, with definitions of centering, any coordinate rescaling and the residual normalization. Distinguish `logvar = log(sigma^2)` from `log(sigma)` consistently in arrays, plots and text.
- [ ] Use the same checkpoint, sample and active set throughout. Explain any difference from earlier notebook numbers before replacing a manuscript value.

**Done when:** the common-scaling discussion and its figures use reproducible, consistently defined statistics. These are empirical properties of the learned posterior; posterior samples conditioned on one realized jet should not be identified with an ensemble of physical shower outcomes.

### 3. Make the coupling-response summaries consistent

- [ ] From the stored scans, provide signed per-coordinate slopes and summaries over all active dimensions for the model/sample combinations already retained in the parton-pair and Z+jet analyses. No new model/sample comparison is requested. The current paper identifies 30 active CNN dimensions and 32 active particle-model dimensions; verify these against item 1 and include weak-response dimensions.
- [ ] Replace the primary response-selected CNN summaries with all-active summaries. If a selected subset is retained as a secondary diagnostic, give its selection rule and indices explicitly. Apply the same convention in the main text and Z+jet appendix.
- [ ] Supply scan values, reference values, per-coordinate fit inputs, fit range and variable definitions. Distinguish the dispersion of slopes across coordinates from the statistical uncertainty on a slope or summary.
- [ ] Confirm the uncertainty calculation actually used, including covariance induced by a shared default reference and any pairing of events between samples. Make the response-statistics text agree with the implementation; if a needed covariance contribution was omitted, assess its effect using the saved outputs.

**Done when:** the response figures, captions, quoted means/dispersions and statistical-method description agree with the exported inputs and code for both sample setups.

In the interpretation, distinguish the parton-pair variation of the final-state shower input `TimeShower:alphaSvalue` from the Z+jet joint variation of shower, hard-process and MPI coupling inputs, with the PDF coupling held fixed. Retain the effective-cutoff qualification: the existing independent cutoff scan does not isolate the full positive-coupling cutoff effect. These scans are generator-parameter variations, not a measurement of the running coupling. No new matched-cutoff generation is requested here.

## Numerical data and reproduction commands

### 4. Package the retained evidence

- [ ] First provide an inventory of available paper assets: producer, originating run, format, compressed size and any missing input or redistribution restriction.
- [ ] Export original machine-readable source data for every retained numerical figure and table, together with plotting scripts. Do not digitize rendered plots when the original arrays exist.
- [ ] Include the compact latent/scan outputs needed to recompute the central statistics in items 2–3, where available. Report any gap between reproducing a plot and independently recomputing its underlying statistic.

| Result | Source data to include |
|---|---|
| Reconstruction examples | Exact displayed input and reconstruction arrays, event identifiers, axes/bin edges, normalization and plotting transformations. |
| Correlation matrices and pair plots | Numerical matrices and latent indices; density-bin edges/counts or underlying values. Include the corresponding eventwise posterior outputs needed to recompute correlations and the activity criterion. |
| Common-mode/scaling statistics | Eventwise log variances for the retained sample, active indices, centering/scaling definitions, loadings, variance fractions and residual outputs. |
| Coupling-response curves and summaries | Exact generator settings/defaults, sample sizes, per-coordinate scan means, uncertainty/covariance inputs and fitted slopes. Identify the all-active and any selected sets separately for each sample setup. |

Use code, manifests and small curated tables in this repository. Larger arrays and checkpoints can live in a linked, versioned archive with checksums; choose the host after checking actual sizes. If curated data files are added here, make narrow exceptions to the current data-file ignore rules while keeping raw caches and training outputs ignored.

For encoder-level checks, inventory the final checkpoints, exact preprocessing/model configurations and evaluation/scan jet inputs. If original inputs cannot be distributed, supply the generator recipe and state what regeneration can and cannot reproduce. Full raw training samples and intermediate checkpoints are not prerequisites for the first package; preserve available originals and provenance.

### 5. Provide a runnable paper recipe and verify it

- [ ] Add a paper-specific entry point listing the environment, generator cards or equivalent settings, saved-input locations and commands for every retained figure. Clearly distinguish this recipe from generic training examples.
- [ ] Have a collaborator regenerate the retained figures and quoted statistics from the documented saved inputs without retraining. Record the package/code version, commands, numerical agreement or tolerances, and anything that could not be checked.
- [ ] Before an external release, resolve artifact access and licensing for code, data and weights, replace pending download links with working versioned links, and record the frozen release commit/tag in the paper.

**Done when:** another collaborator can reproduce the retained numerical evidence from the supplied package, with any remaining limitations stated accurately. Reproducing plots from source tables is distinct from rerunning the encoder or reproducing training.

Preparing these files in the private collaboration repository does not itself set a public-release date or change licenses. Public availability and artifact-specific terms should be settled before external distribution.

## Scope and handoff

The flavor-label exploration is outside this paper. No new flavor-classification study, broad seed/beta sweep, symbolic-regression study or curvature-mechanism campaign is requested. Preserve the original research commits; there is no need to merge all exploratory `ai4jet` branches into this repository.

Please return completed items with their paths and a short list of blockers. For each blocker, identify the affected figure or claim, the missing evidence, the smallest useful check and an effort estimate. If existing outputs cannot support a retained claim, flag it for qualification or removal before expanding the research scope.
