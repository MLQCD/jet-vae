# Validation of the published comparison

The [validation record](validation.json) gives model-level results and SHA-256
hashes of the numerical inputs (UTF-8 text with LF line endings) and report.

- All 12 checkpoints passed the registry, finite-value, correlation-matrix,
  histogram-count and shared-nominal checks in `tests/test_comparison.py`.
- Independently solving the GLS normal equations recovered every published
  parameter slope within 1.8e-15 of the whitened least-squares result.
- All 80,000 reference image-observable values matched the existing analysis.
- The five CNN posterior correlation matrices matched their existing results.
  Common pixel SSE agreed within 1.2e-10. Original slope tables were checked for
  image beta=2e-6 through 5e-6, with agreement within 1.6e-15; the image beta=1e-6
  slope table was not separately compared with an archived table.
- The current-paper particle checkpoint recovered all 297 appendix observable
  correlations within 1e-13, the exact multiplicity-matched RAMBO responses, and
  the exact QCD/RAMBO histogram counts.
- Reused scan inputs were checked against all 66 source-file hashes and particle
  arrays. Three image conversions per source file matched the prepared images.
  Reused posteriors were checked against the corresponding checkpoint hashes.
- Each particle checkpoint used 79,000 RAMBO multiplicity-scan events and 99,975
  multiplicity-matched events. Generation checks conserve the target four-vector
  and verify massless particles. Published histograms retain every event.
- The final report has 87 pages, nine topic bookmarks and 86 model bookmarks.
  Model labels and page-index links were checked, and every page was rendered
  and visually reviewed.

These checks establish consistency of this release. They do not estimate
training-seed uncertainty or equate the CNN and particle reconstruction losses.

Run the portable checks from the repository root:

```bash
python -m unittest discover -s model_comparison/tests -v
```
