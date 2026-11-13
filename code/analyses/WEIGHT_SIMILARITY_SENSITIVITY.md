# Weight similarity sensitivity: one post-training workflow

Run from any working directory using the repository's Python environment:

```bash
.venv/bin/python code/analyses/weight_similarity_sensitivity.py
```

This is the maintained entry point for the **pairwise scaling** sensitivity table
and empirical figures developed in this analysis. It does not run training,
minimum-norm balancing, subject bootstrap, or activity comparisons. Earlier
experimental runners and their outputs are retained as historical records; this
workflow does not load their analysis directories.

## Inputs and selection

Default run: `nested_DA_last10_P100_S32_B8`; default hidden size: 2.

The script reads:

- `data/rnns/<run>/_aws/run-config.json` for frozen subject/fold settings.
- `data/analysis/performance/performance_df.htsv` for saved fit paths and
  combined training-and-validation NLL. Use `--performance-table` to select
  another table for the same run. If the table is missing, build it using the
  shared `analysis.get_analysis_df` and `performance.get_performance_df` helpers first.
- The selected original model pickles, loaded on CPU.

It uses `performance.select_best_outer`, selecting the lowest `train_val_CE`
inner fit per subject × model × outer fold. Exact ties use the lower inner-fold
index. The selected scores, endpoints, absolute pickle paths and model SHA256s
are saved. Subject/fold coverage and matched architecture/activation/bias
conditions are checked. Models from another run are rejected.

For the default dataset this selects 1,320 fits: 11 subjects × 10 outer folds ×
12 two-unit variants, yielding 71,940 fit pairs. No held-out performance is used
for selection.

## Computations

All pairs within each model variant are compared in the same orientation:
subject then outer-fold sorting, with reference A fixed and B aligned to A.

- **ReLU baseline:** enumerate permutations and the Cartesian product of
  `{0.5, 0.875, 1, 1.25, 1.625, 2}` per unit.
- **ReLU refined:** start at the winning grid scales and hold that permutation
  fixed. Use analytical-Jacobian trust-region reflective least squares in log
  scales, bounds `[-20,20]`, `ftol=xtol=gtol=1e-10`, at most 300 residual
  evaluations. Retain the grid candidate if refinement fails or does not improve
  the objective. This uses `measures.mechanistic_variability.align_rnns`.
- **Tanh baseline:** enumerate permutations and signs, without positive scaling.
- **Hypothetical scaled tanh:** refine positive scales in every permutation/sign
  branch starting from unit scale, then select the lowest distance. This uses
  the reusable `measures.scaling_sensitivity.fit_pair` helper and the same TRF
  tolerances/bounds. It deliberately changes model function and is not the
  function-preserving main tanh method.

The alignment objective contains input weights, hidden bias, recurrent weights
and decoder weights. The reported cosine contains **all trainable parameters**,
including gates and initial states. Improving the former can lower the latter.
Neither continuous search is a guarantee of a global optimum.

First average 45 fit pairs per within-subject block or 100 per across-subject
dyad. Then average the 11 subjects or 55 dyads equally. Percentage change is
`100 * (scaled mean - baseline mean) / abs(baseline mean)`; the summary is a
ratio of means, not a mean of pairwise percentages. Ranges are across variants,
not uncertainty intervals. Dyads share subjects and are not independent.

## Outputs

Tables, cached computations and provenance:

```
data/analysis/weight_similarity_sensitivity/
```

- `selected_models.htsv`: exact selected fits, source scores and pickle hashes.
- `pairwise_scaling.htsv`: original and scaled cosine/distance, endpoints and
  optimizer diagnostics for every fit pair.
- `subject_pair_scaling.htsv`: subject/dyad means and percentage changes.
- `per_variant_summary.htsv`, `scaling_ranges.htsv`: the sensitivity
  table, per-variant values and methods explanation.
- `activation_contrasts_discrete.htsv`: original discrete-ReLU minus unscaled-tanh
  empirical contrasts, reproducing the previous standard-method plot.
- `activation_contrasts_continuous.htsv`: grid-refined ReLU minus unscaled-tanh
  contrasts, using the agreed continuous refinement for ReLU.
- `manifest.json`, `groups/*.json`: settings, source/model hashes, cache identity
  and convergence/bound diagnostics.

PNG and SVG figures:

```
results/figures/weight_similarity_sensitivity_<run>_h2/
```

- `relu_scaling_sensitivity`, `tanh_scaling_sensitivity`: empirical changes from
  scaling for each subject/dyad, with overall mean markers.
- `activation_contrasts_discrete`, `activation_contrasts_continuous`: empirical
  ReLU-minus-tanh differences, with within/across panels side by side.
- `combined_discrete`, `combined_continuous`: tanh-scaling sensitivity on top,
  activation contrasts below. All four panels have identical x limits and ticks.
  The upper title explicitly names tanh; lower captions identify which ReLU
  condition was used. **Use the discrete version to reproduce the earlier
  figure; use the continuous version for the adopted refined alignment.**

## Manuscript exports

The default two-unit run also exports the requested manuscript files:

- `results/appendix/figure_weight_sensitivity.png`: the combined figure with
  continuously refined ReLU alignment in the lower panels.
- `results/appendix/table_weight_sensitivity.csv`: the two-row percentage-range
  summary (comparison, within subjects, across subjects).

Plot runs refresh both files; report-only runs refresh the table. Other runs
and hidden sizes do not overwrite these default manuscript exports.

## Stages, reuse and parallelism

```bash
# Compute, report, and plot (reuse groups only after provenance validation):
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --jobs 6

# Rebuild all pair scores:
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --recompute

# Rerun just a stage from verified frozen computation:
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --stage report
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --stage plots

# Other run or one-unit models:
.venv/bin/python code/analyses/weight_similarity_sensitivity.py --folder RUN --hidden-size 1
```

Group reuse requires the same selected endpoints, model bytes, settings and
alignment/study source code, plus matching output checksums and complete pair
coverage. Group files are written atomically; interrupted runs can reuse fully
completed matching groups. Report/plot-only modes verify frozen selection and
pair-table hashes and never load models or rerun fitting/alignment. The study
writes separate outputs and does not replace notebook caches or earlier figures.

Narrative reports (`REPORT.md` and `activation_associated_variance.md`) are
written to `results/appendix/weight_similarity_sensitivity/`. The intermediate
folder holds one active dataset/hidden-size output set; later stages check
these settings against `manifest.json`.
