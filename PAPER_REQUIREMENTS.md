# Minimal paper-completion checklist

The aim is to finish the paper using the existing results, with only the checks and corrections needed to support its claims. Reuse existing scripts, notebooks and saved outputs. Completed work does not need to be repeated or reformatted into new reports.

Peiran and Sitong should agree on the practical division of work, keeping each result with the person who produced it where possible. Peiran coordinates the physics results and manuscript updates. Sitong coordinates the code and numerical-data package and verifies the comparisons she performed. They can adjust this split together.

## Proposed schedule and tracker

Targets below are proposed for the week of November 11, 2026; Peiran and Sitong should confirm feasibility and adjust the split. Update the last column with `Done` plus a file/commit link, or a one-line blocker. No separate report is needed.

| Action | Lead | Target | Completion evidence / current status |
|---|---|---|---|
| Confirm inputs and practical split | Peiran and Sitong | Thu Nov 12, end of day CT | Paper config and Figs. 2–6 scripts added in `30c3086`; confirm exact checkpoint/sample locations and any missing inputs. |
| Resolve essential numerical checks and circulate stable PDF | Peiran with original producers; Sitong verifies her dataset comparisons | Fri Nov 13, end of day CT | All-active summaries, common-width numbers, fit/error wording and existing comparison tables need confirmation. Appendix C (Tables II–III) is prepared in Zhen's revised working draft; shared-draft integration is pending. |
| Collect existing plot data and commands | Sitong coordinates; producers supply outputs | Fri Nov 13 for available files and remaining gaps | `paper_figures/README.md` and scripts exist; data locations/exports and final numerical agreement need confirmation. Larger-file hosting may follow. |
| Final proofreading and integration | Zhen, Peiran, Nadja, Qianwen and Sitong; Peiran consolidates | Mon Nov 16, noon CT, if stable PDF is ready Friday | Each replies with corrections or no further corrections; availability and completion pending. |

**Friday decision:** if the numerical checks are resolved and the stable draft is circulating, aim for submission Monday or Tuesday, November 16–17, after all five authors have reviewed it and necessary corrections are incorporated. If an essential input, substantive discrepancy or coauthor review remains outstanding, continue at normal speed. The submission target does not guarantee an arXiv announcement date. Large-data hosting and cosmetic repository cleanup should not delay a supported v1; its availability statement must be accurate.

## 1. Confirm the inputs already used

- [ ] Identify the saved runs and samples behind the retained figures. Existing paths, notebook references and configuration files are sufficient; a short README list can connect them.
- [ ] Confirm the actual paper model settings, preprocessing, loss normalization and active-dimension definition. The 32-dimensional CNN paper configuration is in `cnn_vae/configs/betavae_paper.yaml`, and the generic CNN default was also changed to 32 in `65023b1`; confirm the settings against the actual run. Do not infer training settings absent from the checkpoint. Correct the manuscript or paper instructions wherever they differ from the actual run.

No new training is needed merely to make the research history more uniform or to match generic repository defaults.

## 2. Complete only the necessary numerical checks

- [ ] Confirm that the quoted correlation/common-scaling statistics come from the same saved runs and samples as the figures, with consistent `log(sigma^2)` versus `log(sigma)` conventions. Reuse the existing quantitative evidence. Additional PCA, equal-loading, loading-spread or residual diagnostics are not required unless a specific retained claim lacks support.
- [ ] Using the existing scans, make the primary response summaries use all active dimensions, including weak-response coordinates, for the model/sample combinations already in the paper. Verify the current counts of 30 active CNN and 32 active particle-model dimensions. The current `paper_figures/fig5_6_parameter_response.py` removes weak-response coordinates from the alpha_s summary; use all active coordinates for the primary summary. Check the existing Z+jet summary consistently as well, using its original analysis. Any retained response-selected subset must be clearly labeled as secondary. Update only the affected numbers, figures and captions.
- [ ] Check that the slope fits, error bars and response-statistics text describe the implemented calculation. Distinguish coordinate-to-coordinate dispersion from statistical uncertainty, and check the treatment of the shared default reference where it affects reported errors. The current Figs. 5–6 script fits with a free intercept and displays the standard deviation across coordinate slopes; confirm that the manuscript describes this correctly. Recompute only if an actual inconsistency or omission is found.

- [ ] **Sitong, with Peiran: verify the existing cross-dataset comparison for the supplementary appendix.** Start from slides 1 and 3 of the [shared progress deck](https://docs.google.com/presentation/d/10rJW42gAueKff9ojvMnDuL-QWZcEI0hqWhiLS_gT_hM/edit). The working draft retains 27 non-PID rows in the effective-rank comparison and 14 in the additional-regularization comparison. Confirm the exact jet run and beta for the reported rank-1.17 row, whose original beta cell lists several values; identify the `ours`/`repo` run labels; and state the active-coordinate rule, correlation summary (including diagonal treatment), effective-rank formula and covariance-versus-correlation choice. Give the model settings and reconstruction/KL loss reductions needed to interpret the comparison. Update only incorrect or outdated entries from existing outputs. The revised tables, short definitions/caption and existing script/output links are sufficient completion evidence; no new training, beta sweep or separate report is requested.

Return the relevant existing script/output and any corrected numbers. Keep the requested cross-dataset tables in the paper; additional standalone metric reports or a new analysis framework are not required. Keep the established generator-coupling and cutoff qualifications in the manuscript.

## 3. Collect the existing evidence for sharing

- [ ] Save the original machine-readable data underlying the retained plots/tables, with the scripts or notebooks that produced them. Include enough labels, settings and brief instructions to interpret the files and regenerate the plots; reuse existing documentation.
- [ ] Check that the supplied commands work for the affected final plots and central numerical results. There is no separate requirement for a full independent rerun of every analysis or training run. If a result remains unclear, check that result specifically.
- [ ] Keep small files in this repository and identify the sizes/locations of larger arrays or checkpoints for suitable external storage. Full raw training samples and intermediate checkpoints are not required for this package. Agree on public access and artifact licensing before external release; this checklist does not change licenses.

A brief paper-reproduction section in the README is enough. No separate inventories, status reports or duplicate manifests are required if the existing files and instructions already provide the needed information. Describe accurately what the shared package reproduces.

## 4. Integrate corrections and finish proofreading

- [ ] Peiran incorporates the necessary numerical/text corrections, with Zhen on scientific wording. Keep the final figures, captions and quoted numbers consistent. Retain the existing cross-dataset comparison as a supplementary appendix with a main-text reference; use the prepared Appendix C as the starting point, resolve its two `\ZL{...}` comments with Sitong, replace provisional run labels and add the dataset/reference-implementation citations. Include the smaller-beta non-jet results as well as the effective-rank table. Explain the dataset and regularization dependence, and connect the jet result's physical relevance to its QCD parameter response; correlation alone does not establish jet exclusivity. Keep effective rank distinct from principal-component variance fractions, and do not infer missing ranks from correlation values.
- [ ] Zhen, Peiran, Nadja, Qianwen and Sitong each proofread the same final PDF and return corrections or confirm that they have none. Peiran consolidates the corrections; substantive changes go back to the relevant coauthors.

## Scope

PID comparisons are deferred to future work because they did not improve benchmark performance and would complicate the presentation; they do not weaken the central observation. No further PID runs are requested. The flavor-label exploration also remains outside this paper.

The existing non-jet benchmark comparisons belong in this paper's supplementary material; updating and integrating them does not call for a new benchmark campaign. No new model/sample comparisons, broad seed/beta sweeps, matched-cutoff generation, symbolic-regression or curvature studies are requested. If an essential saved output is missing, first try to recover it or rerun inference with the existing checkpoint and sample. Discuss new training or event generation only if a specific problem with a retained result cannot otherwise be resolved.

The work is complete when the retained claims are supported, the identified inconsistencies are resolved, the existing plot evidence is collected, and all five coauthors have proofread the final paper. Additional analyses and infrastructure should not be added without a concrete publication need.
