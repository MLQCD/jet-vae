# Paper checks and reproduction package

This paper presents correlations and approximate common scaling of VAE posterior variances, together with their response to generator-coupling variations. The tasks below connect the retained results to their exact runs and make the numerical evidence reproducible. Start from completed runs and saved outputs; identify a missing input before proposing additional computation.

## Assignment and order

Owners below are proposed for assignment, not confirmed commitments. Peiran coordinates the paper package; Sitong supplies the sample/model comparisons she performed. Each contributor owns the provenance of their own results. Reassign a task if its original producer is better placed to complete it.

| Task | Proposed lead / support | Dependency | Deliverable | Status |
|---|---|---|---|---|
| T1. Recover exact paper inputs | Peiran coordinates; Sitong supplies her comparison inputs | None; start here | Figure/run manifest, exact configs, asset inventory | Unassigned |
| T2. Quantify common scaling | Peiran; original analysis producer supplies outputs | Relevant T1 inputs | Script and per-model statistics table | Unassigned |
| T3. Recompute all-active responses | Peiran; Sitong supplies comparisons she produced | Relevant T1 scan inputs | Per-coordinate slopes, all-active summary and revised figures | Unassigned |
| T4. Verify fit uncertainties | Original fit producer; Peiran integrates | T1 fit inputs; work alongside T3 | Method check, corrected fit outputs if needed, draft-ready methods text | Unassigned |
| T5. Assemble numerical source data | Each figure producer; Peiran consolidates | Start with T1; finalize after T2–T4 | Source-data inventory, arrays/tables and plotting commands | Unassigned |
| T6. Reproduce the retained results | Sitong for results she did not produce; another collaborator for her own results | T2–T5 | Reproduction log and resolved discrepancies | Unassigned |
| T7. Close the manuscript | Zhen with Peiran | T2–T4 for numerical updates; T6 for final sign-off | Consistent text, figures and captions; reviewed PDF | Unassigned |
| T8. Prepare external release | Peiran packages; Zhen coordinates release decisions | Size inventory from T1/T5; verified package from T6 | Versioned links, checksums, agreed terms and release identifier | Unassigned |

**Start with T1 and the available parts of T5.** T2, T3 and T4 can proceed in parallel as soon as their inputs are identified; do not wait for unrelated figures. T7 can proceed alongside the numerical checks. T8 hosting/licensing logistics can proceed in parallel and are distinct from verifying the scientific results.

For each task, replace `Unassigned` with the agreed owner and target date, then record `In progress`, `Blocked` or `Done` with an output link/commit. The paths below are proposed destinations for new deliverables, not files that already exist. Reuse an existing equivalent path and link it here rather than duplicating outputs.

## Immediate paper checks

Complete these checks before freezing the numerical results. For each item, add the output path, producer and any remaining limitation here. A missing artifact should be reported explicitly rather than replaced with a similar run.

### T1. Identify the exact paper runs

- [ ] Add a figure-to-run manifest covering every retained figure and quoted numerical result. Include the original code commit, checkpoint location and checksum, sample and split identifiers, preprocessing settings, training configuration, and analysis/plotting command.
- [ ] Supply the exact CNN paper configuration with 32 latent dimensions. The repository's generic 128-dimensional configuration is not the paper recipe. Also identify the exact particle-model configuration; do not infer it from example defaults.
- [ ] Record the effective reconstruction and KL loss normalization, including reductions over events, pixels/particles and latent dimensions. State whether diagnostic samples contain training events.
- [ ] Record the active-dimension rule and indices for each model, determined from the default reference sample. Keep this activity definition separate from any subsequent selection by coupling response.

**Output:** `paper/figure_manifest.csv`, `paper/configs/` and `paper/asset_inventory.csv` (including compressed sizes, availability and missing inputs).

**Done when:** each paper result can be traced to a specific saved run and sample. Use this manifest to reconcile the model, training and sample descriptions in the methods and appendices.

### T2. Quantify the common-width pattern on those runs

- [ ] On the samples used in the correlation/scaling figures, export the active count, leading principal-component variance fraction, equal-loading-mode variance fraction, loading spread and a defined residual measure for each model.
- [ ] Include the script and numerical output, with definitions of centering, any coordinate rescaling and the residual normalization. Distinguish `logvar = log(sigma^2)` from `log(sigma)` consistently in arrays, plots and text.
- [ ] Use the same checkpoint, sample and active set throughout. Explain any difference from earlier notebook numbers before replacing a manuscript value.

**Output:** `paper/analyze_common_widths.py` (or the existing equivalent), `paper/results/common_widths.csv`, and a short definitions note. Include one row per retained model/sample combination, with the sample size, active indices and all requested metrics.

For a concrete common-scaling diagnostic, let each column of `X` be one active coordinate's eventwise `logvar`, centered by its sample mean, without dividing by its coordinate standard deviation. With covariance `C` and normalized equal-loading vector `e = (1,...,1)/sqrt(d)`, report `lambda_max(C)/trace(C)` and `(e^T C e)/trace(C)`. Report the unit-normalized leading eigenvector (choose its sign so the sum of loadings is positive), its RMS loading spread about its mean, and the normalized equal-mode residual `||X - (X e)e^T||_F / ||X||_F`. The last quantity equals `sqrt(1 - equal_mode_fraction)` and is a useful consistency check, not an independent statistic. If an existing paper diagnostic uses another convention, retain and label it separately, explain the difference, and reconcile the manuscript definition. This adds a short analysis of saved arrays, not a new model fit.

**Done when:** the common-scaling discussion and its figures use reproducible, consistently defined statistics. These are empirical properties of the learned posterior; posterior samples conditioned on one realized jet should not be identified with an ensemble of physical shower outcomes.

### T3. Make the coupling-response summaries consistent

- [ ] From the stored scans, provide signed per-coordinate slopes and summaries over all active dimensions for the model/sample combinations already retained in the parton-pair and Z+jet analyses. No new model/sample comparison is requested. The current paper identifies 30 active CNN dimensions and 32 active particle-model dimensions; verify these against T1 and include weak-response dimensions.
- [ ] Replace the primary response-selected CNN summaries with all-active summaries. If a selected subset is retained as a secondary diagnostic, give its selection rule and indices explicitly. Apply the same convention in the main text and Z+jet appendix.
- [ ] Supply scan values, reference values, per-coordinate fit inputs, fit range and variable definitions. Distinguish the dispersion of slopes across coordinates from the statistical uncertainty on a slope or summary.

**Output:** `paper/results/response_slopes.csv` (one row per setup/model/parameter/coordinate, including activity and any secondary-selection flag), `paper/results/response_summary.csv`, the generating script and replacement figure files. Give the exact slope convention and dispersion definition. Reconcile the final uncertainties with T4.

**Done when:** the response figures, captions and quoted means/dispersions agree with the all-active outputs for both retained sample setups.

### T4. Verify the fit and uncertainty method

- [ ] Trace the reported slopes/error bars back to the code actually used. Record the fit variable, range, weighting, sample sizes and meaning of each error bar.
- [ ] Check covariance induced by a shared default reference and any pairing of events between samples. If a needed contribution was omitted, assess its effect using saved eventwise outputs or sufficient statistics; identify missing information if these are unavailable.
- [ ] Compare the old and verified results, recording whether central values, uncertainties or interpretation change. Supply a concise methods paragraph matching the implemented procedure.

**Output:** `paper/fit_method.md`, the fit script, and `paper/results/fit_check.csv` with old/new results where a correction is needed. Link unchanged results with an explanation when no correction is needed.

**Done when:** Peiran and the fit producer can account for the stated uncertainties, and the response-statistics subsection and captions describe the verified implementation. This check does not require new event generation unless the required information cannot be recovered.

In the interpretation, distinguish the parton-pair variation of the final-state shower input `TimeShower:alphaSvalue` from the Z+jet joint variation of shower, hard-process and MPI coupling inputs, with the PDF coupling held fixed. Retain the effective-cutoff qualification: the existing independent cutoff scan does not isolate the full positive-coupling cutoff effect. These scans are generator-parameter variations, not a measurement of the running coupling. No new matched-cutoff generation is requested here.

## Numerical data and reproduction commands

### T5. Package the retained evidence

- [ ] First provide an inventory of available paper assets: producer, originating run, format, compressed size and any missing input or redistribution restriction.
- [ ] Export original machine-readable source data for every retained numerical figure and table, together with plotting scripts. Do not digitize rendered plots when the original arrays exist.
- [ ] Include the compact latent/scan outputs needed to recompute the central statistics in T2–T4, where available. Report any gap between reproducing a plot and independently recomputing its underlying statistic.

| Result | Source data to include |
|---|---|
| Reconstruction examples | Exact displayed input and reconstruction arrays, event identifiers, axes/bin edges, normalization and plotting transformations. |
| Correlation matrices and pair plots | Numerical matrices and latent indices; density-bin edges/counts or underlying values. Include the corresponding eventwise posterior outputs needed to recompute correlations and the activity criterion. |
| Common-mode/scaling statistics | Eventwise log variances for the retained sample, active indices, centering/scaling definitions, loadings, variance fractions and residual outputs. |
| Coupling-response curves and summaries | Exact generator settings/defaults, sample sizes, per-coordinate scan means, uncertainty/covariance inputs and fitted slopes. Identify the all-active and any selected sets separately for each sample setup. |

**Output:** `paper/source_data_manifest.csv`, with one entry per figure panel/table, data and script locations, checksums and reproduction scope. Put small curated data under `paper/data/`; list larger artifact locations without adding them to Git.

Use code, manifests and small curated tables in this repository. Larger arrays and checkpoints can live in a linked, versioned archive with checksums; choose the host after checking actual sizes. If curated data files are added here, make narrow exceptions to the current data-file ignore rules while keeping raw caches and training outputs ignored.

For encoder-level checks, inventory the final checkpoints, exact preprocessing/model configurations and evaluation/scan jet inputs. If original inputs cannot be distributed, supply the generator recipe and state what regeneration can and cannot reproduce. Full raw training samples and intermediate checkpoints are not prerequisites for the first package; preserve available originals and provenance.

### T6. Provide a runnable paper recipe and verify it

- [ ] Add a paper-specific entry point listing the environment, generator cards or equivalent settings, saved-input locations and commands for every retained figure. Clearly distinguish this recipe from generic training examples.
- [ ] Have a collaborator regenerate the retained figures and quoted statistics from the documented saved inputs without retraining. Record the package/code version, commands, numerical agreement or tolerances, and anything that could not be checked.

**Output:** `paper/README.md` with commands and environment information, and `paper/reproduction_check.md` with the tester, code/input versions, command results, numerical comparisons and any unresolved discrepancy.

**Done when:** another collaborator can reproduce the retained numerical evidence from the supplied package, with any remaining limitations stated accurately. Reproducing plots from source tables is distinct from rerunning the encoder or reproducing training.

### T7. Integrate the verified results into the manuscript

- [ ] Update the methods/configuration description from T1, common-scaling numbers from T2, and response figures, captions and uncertainties from T3–T4. Check each changed number against its source output.
- [ ] Retain the generator-coupling and cutoff qualifications above. Keep primary claims about the observed correlations and approximate common scaling aligned with the verified results.
- [ ] Defer PID input/sample comparisons to future work: they did not improve benchmark performance and would complicate this presentation; they do not weaken the central observation. No further PID runs or dedicated PID section are required. Avoid a general claim that PID can never be useful in jet tasks.
- [ ] Review transitions, acknowledgments, repository link, data/code availability wording and the final compiled PDF. Resolve any discrepancy raised by T6 before treating the corresponding numerical result as final.

**Output:** revised manuscript source/figures and matching PDF through the established manuscript workflow, plus a short list mapping numerical changes to T2–T4 outputs. Keep manuscript files in the manuscript project; this code repository need only link the associated release/version.

**Done when:** Zhen and Peiran have reviewed a consistent final draft whose central numbers have passed T6. Repository cleanup alone is not numerical validation.

### T8. Prepare the external release

- [ ] Use the measured file sizes to choose storage for larger arrays/checkpoints. Keep code, manifests and small curated data here; provide versioned artifact links and checksums.
- [ ] Resolve access and artifact-specific licensing for code, data and weights before external distribution. Replace pending download links with working links and identify anything that cannot be released.
- [ ] Freeze the tested code/data package, record its commit/tag (and archival identifier if used), and put that identifier in the manuscript's availability statement.

**Output:** completed download instructions, artifact checksums and agreed release terms, tied to the tested package version.

**Done when:** the intended release files are accessible under the agreed terms and the availability statement is accurate. Internal verification can finish before public hosting is ready. Preparing these files in the private repository does not itself authorize public distribution or change licenses.

## Scope and handoff

The flavor-label exploration is outside this paper. No new flavor-classification study, broad seed/beta sweep, symbolic-regression study or curvature-mechanism campaign is requested. Preserve the original research commits; there is no need to merge all exploratory `ai4jet` branches into this repository.

## When additional computation is needed

The default is to reuse saved outputs. If latent arrays are missing but the exact checkpoints and jet inputs exist, rerun encoder inference only. Refit or recompute uncertainties from saved scan outputs where possible. Before proposing retraining or new event generation, report the missing artifact, affected result, why recovery/reanalysis is insufficient, the smallest useful run and an effort estimate; agree on it before expanding the work.

Please return completed items with their paths and a short list of blockers. For each blocker, identify the affected figure or claim, the missing evidence, the smallest useful check and an effort estimate. If existing outputs cannot support a retained claim, flag it for qualification or removal before expanding the research scope.
