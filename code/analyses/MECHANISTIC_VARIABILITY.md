# Mechanistic variability: one post-training workflow

Start here for the baseline mechanistic analyses. From the repository root:

```bash
.venv/bin/python code/analyses/mechanistic_variability.py
```

Default run: `nested_DA_last10_P100_S32_B8`. The command performs, in order:

1. **Compute:** load experiment/model/performance dataframes, select exactly
   `performance.select_best_outer(performance_df)`, recover each reference subject's saved
   trials, compute similarities, and publish the notebook-compatible cache.
2. **Statistics:** aggregate fold pairs per subject/subject pair and compare every
   two-unit configuration against GRU/tanh/decoder-bias-on using the existing
   descriptive paired dz, subject-deletion jackknife and global Holm correction.
3. **Plots:** export all-configuration two-unit violin panels, individual scores,
   a readout zoom and GRU-reference tables (including across/within side by side).

No models are trained and no AWS/SLURM jobs are submitted by this study. It uses
imported baseline model artifacts and their frozen `_aws/run-config.json`.
It is a standalone post-training entry point, separate from the training task
registry used by `code/analyses/run.py`.

## Rerun only what you need

```bash
# Force fresh similarity computation, then statistics and plots:
.venv/bin/python code/analyses/mechanistic_variability.py --recompute

# Individual stages:
.venv/bin/python code/analyses/mechanistic_variability.py --stage compute
.venv/bin/python code/analyses/mechanistic_variability.py --stage statistics
.venv/bin/python code/analyses/mechanistic_variability.py --stage plots

# Different imported run / parallelism:
.venv/bin/python code/analyses/mechanistic_variability.py \
  --folder nested_DA_last10_P100_S32_B8 --jobs 3
```

The default full run reuses computation only if its selected endpoints, model
bytes, subject-specific reference inputs, frozen settings and measures code match. Statistics and
plots are then regenerated. Later stages consume the frozen computation tables,
checking their hashes; plot-only runs never perform model forward passes.

## Where code belongs

| Responsibility | File |
|---|---|
| Selection, stage order, caching, dataframes and provenance | `code/analyses/mechanistic_variability.py` |
| Reusable alignment, block probes, scores and statistical computations | `code/measures/mechanistic_variability.py` |
| Reusable figure functions; no loading or CLI | `code/figures/mechanistic_variability.py` |

The separate `rebuild_similarity_cache.py`, `baseline_similarity_statistics.py`
and `appendix_baseline_similarity_table.py` runners have been retired. Older
`baseline_analysis.py` imports are compatibility aliases; executing it forwards
to the study. Performance plotting and experimental diagnostics remain separate;
they are not dependencies of the maintained study.

## Outputs

All study dataframes and provenance:

```
data/analysis/mechanistic_variability/
```

- `experiment_df.htsv`: model/subject/outer manifest.
- The shared inner-fit path and performance tables live in
  `data/analysis/performance/analysis_df.htsv` and `performance_df.htsv`; they
  are not duplicated in this analysis folder.
- `selected_models.htsv`: one train+validation inner winner per outer fold.
- `reference_subjects.htsv`, `reference_inputs.npz`: each subject’s trial counts, source path and complete input sequence.
- `pairwise_similarities.htsv`: within/across-subject fold-model comparisons.
- `subject_pair_similarity.htsv`, `similarity_summary.htsv`: aggregate scores.
- `similarity_vs_standard_gru.htsv`: means, raw contrasts, paired dz, CIs, p-values.
- `manifest.json`, `computation_inputs.json`: stage provenance and fingerprints.
- `groups/`: per-configuration computation outputs retained for inspection.

Figures and figure summary tables:

```
results/figures/appendix_mechanistic_variability_<run>/
```

All figures have `appendix_` names and PNG/PDF/SVG exports. The study also updates
`data/analysis/mechanistic_variability/pairwise_similarities.htsv` for the
existing notebook API, with selected-model source checks and `notebook_cache.json` provenance.
Superseded notebook caches are preserved under `data/analysis/.local/`.

## Scientific policies retained

- Minimum combined `train_val_CE`, exactly as the notebook selector implements it.
  That selection score comes from continuous trial exports; this is distinct from
  the block-reset inputs used for similarity.
- Every selected outer-fold winner is included. For the current run: 24 variants,
  110 models each (11 subjects × ten outers), 5995 pairs per variant, 143880 total.
- **Restored notebook reference method:** within-subject pairs run both models
  on that subject's own data. Across-subject pairs run both models on subject B's
  data, comparing A transferred to B against B on its own data. `trial_subject`
  contains the real subject ID and equals `subject_B` for every saved pair.
- Uses the complete saved trial sequence, with the notebook's original continuous
  state handling (including across sessions within that saved sequence). This is
  distinct from the 32-step resets used during training. No pooled/balanced probe
  set is substituted. Forced-choice steps are included in similarity.
- A/B follows the original sorted subject/outer-fold pair ordering. The primary
  table does not silently average the reverse direction; A-on-B and B-on-A may
  differ. Unordered subject pairs contribute equally to aggregate means, but
  sorted pair orientation gives subjects unequal frequency as the reference B.
- Existing canonical parameter alignment; parameter cosine; unitwise-averaged
  hidden Pearson; **flattened** gate/readout Pearson. Missing reset gates remain
  missing. Parameter comparisons are only within matching configurations.
- Pair plots are descriptive. Within grey dots = 11 subject means over 45 fold
  pairs each; across dots = 55 subject-pair means over 100 fold combinations each.
- Reference tables retain approximate subject-deletion jackknife t inference and
  Holm correction across all 94 available two-unit contrasts. Only 11 subjects:
  exploratory inference. CIs are pointwise; stars use adjusted p-values.

The study now reproduces the original notebook's subject-B/continuous-trial
protocol using batched evaluation and cached correlations. Direct regression
checks compare it with `_process_group` for all three architectures and both
nonlinearities. Cached loads validate selected endpoint IDs and pair counts.

Superseded pooled-reference tables and figures are preserved with the suffix
`_shared_reference_previous`. Their scores must not be mixed with the restored
subject-specific outputs. `reference_protocol_comparison.htsv`, when present,
records before/after changes; it reflects both reference-data and state-handling
differences, not an isolated test of pooling.

## Preliminary scientific checks

Quick exploratory checks live under **`code/analyses/preliminary/`**, which is
explicitly gitignored along with its local results. For example:

```bash
.venv/bin/python code/analyses/preliminary/check_reference_direction.py
```

The quick directionality check uses outer fold 1 from all 11 subjects for standard
two-unit GRU and ReLU/no-bias monoGRU: 55 subject pairs per architecture, each
scored on A and on B. Alignment is held fixed to isolate the reference data;
this is not a full ten-fold inferential analysis. Its local manifest, CSVs and
plot record that scope. Implementation regression tests remain in `code/tests`.

## Main GRU–monoGRU result

The full workflow (or `--stage plots` after computing statistics) also writes:

- `results/main/figure_GRU_monoGRU_mech_var.svg` (SVG only).
- `results/main/table_GRU_monoGRU_mech_var.csv`.

These compare two-unit GRU/tanh/decoder-bias-on (`GRU`) against two-unit
monoGRU/ReLU/decoder-bias-off (`monoGRU+BC-DB`). The outlined violins contain
individual outer-winner model pairs, matching the notebook: weights, logit,
then hidden state; across subjects left, within subject right. The 12-row CSV
contains counts, mean, SD, median and quartiles of these same distributions,
plus the existing subject-level contrasts against GRU. Its `p_holm` retains
correction over the full study's configuration/metric comparisons; model-pair
counts are descriptive sample counts, not independent inferential replicates.
The generic filenames are overwritten for the run passed to `--folder`; its
study manifest records output hashes and the source run.

The intermediate folder contains one active run, with its dataset identifier
in `manifest.json` rather than its path. Plot/statistics-only stages require
`--folder` to match that manifest. See [data layout](../../data/analysis/README.md).

## Full-page appendix figure

`--stage plots` also exports `results/appendix/figure_all_models_mech_var.svg`
(SVG only, A4 portrait). All 12 two-unit architecture/activation/decoder-bias
configurations appear in each panel. Five rows show weights, logits, hidden
states, update gates and reset gates; columns show across and within subjects.
Axes span −1 to 1 with a dashed ceiling at 1. Colours identify architectures;
bottom labels identify activation and decoder bias. Reset gates absent from
lightGRU/monoGRU are labelled explicitly. Violin observations and winner
selection match the main GRU–monoGRU figure.

## Cohen's d appendix panels

`--stage plots` also exports `results/appendix/figure_mech_var_cohens_d.svg`
and `table_mech_var_cohens_d.csv`, using the saved
`similarity_vs_standard_gru.htsv` contrasts. All 12 two-unit configurations
are compared against GRU/tanh/decoder-bias-on, with across and within panels
sharing a symmetric colour scale. Columns match the violin figure's order.
Cells show paired descriptive Cohen's dz; positive means higher similarity.
Missing reset gates are N/A. Black outlines/stars retain the existing global
Holm-adjusted subject-deletion inference, rather than treating fold pairs as
independent observations. Across-subject dz uses incident-dyad subject summaries,
which share dyads; it is descriptive and does not directly supply the p-values.
