# Mechanistic variability

The maintained entry point is `code/analyses/mechanistic_variability.py`.
Its `METHODS` dictionary defines the reporting policy and is exported as
`data/analysis/mechanistic_variability/methods.json` and `methods.md`, plus the
supplementary `results/appendix/gru_monogru/README.md` and `methods.json`.

## Reproduce

From the repository root:

```bash
.venv/bin/python code/analyses/mechanistic_variability.py
```

Default source: `data/rnns/nested_DA_S32_B8` (patience 20, S32/B8).
Use `--stage statistics` then `--stage plots` to refresh reports from verified
pairwise scores. `--stage plots` alone verifies existing statistics and renders.
`--recompute` forces fresh metric computation. No training or server jobs launch.

## Selection and scoring

`performance.select_best_outer()` selects one saved inner winner per outer fold
using minimum combined **block-reset free-choice train+validation NLL**. For S32,
33 source rows provide 32 prediction targets, with the learned initial state
reset per block. Losses pool by free-target count; test losses and penalties do
not select winners. Missing block scores raise rather than fall back. Explicit
`selection='continuous'` retains the old analysis criterion. Training checkpoint
and hyperparameter selection are unchanged.

Similarity is measured on each reference subject A's complete saved input
sequence, continuously across blocks and sessions, including forced trials.
B is aligned to A; both models receive A's inputs. Subject/model pairs are sorted
and evaluated once; reverse reference directions are not averaged.
Tanh uses permutations/signs; ReLU uses grid search followed by positive continuous
scale refinement of the winning permutation. Alignment minimizes distance over
candidate input/recurrent weights, candidate bias and decoder weights. Gates and
learned initial states transform consistently. Alignment preserves logits.

Scores: all-trainable-parameter cosine; mean corresponding-unit hidden Pearson
(excluding nearly constant units); flattened gate Pearson; Pearson of logit0−logit1.
Scalar monoGRU and vector GRU update gates use their respective flattened vectors.
Reset-gate comparisons are absent where the architecture has no reset gate.

## Statistical units and inference

Each of 11 subjects has ten outer winners. Within each configuration, 45 fold
pairs are averaged to one subject mean. Across subjects, 100 fold combinations
are averaged to one dyad mean: 55 dyads share the same 11 subjects.
Fold pairs and across-subject dyads are not independent replicates.

- **Within:** exact two-sided sign test on independent subject-level paired
  differences. It tests direction/median, not mean magnitude. Zero differences
  are omitted. No normality pretest chooses the test.
- **Across:** mean paired dyad difference, subject-deletion jackknife SE and
  approximate two-sided t with 10 degrees of freedom. Each deletion removes
  every incident dyad. All 11 subjects enter; none is lost by subject_A grouping.
- **Focused GRU/monoGRU:** BH correction across eight available comparisons
  (four measures × two scopes). This family supplies Table 1 and focused plots.
- **Exploratory all-configuration contrasts:** same tests, separate BH family
  across 94 non-reference comparisons. Its stars can differ from Table 1 because
  the question and multiplicity family differ. Reference/absent-gate cells are excluded.

Main Table 1 reports raw mean similarities and median paired differences. Within
stars annotate the median direction contrast; across stars annotate the model
mean because across p-values test mean differences, not medians.

Supplementary Cohen dz is mean-based: paired subject mean difference / SD of
subject differences. Across dz uses overlapping incident-dyad subject means and
is descriptive only; it does not supply the jackknife p-value. Deletion ranges
are influence diagnostics, not confidence intervals. Within ranges recompute the
median; across tables separately label mean and median deletion ranges.
Small sample size, dependent dyads and approximate across inference remain limits.

## Outputs

- `results/main/table_1.svg` and CSV: concise primary table.
- `results/appendix/gru_monogru/table_within.svg` and CSV: raw subject-summary
  mean ± SD, mean/median differences, median deletion range, dz, signs and p/q.
- `results/appendix/gru_monogru/table_across.svg` and CSV: corresponding dyad
  summaries, mean/median deletion ranges, descriptive dz and jackknife p/q.
- `results/appendix/gru_monogru/figure_paired_<metric>.svg`: four figures, each
  showing raw paired values and paired differences within/across subjects.
  Each plot labels its actual test and the focused eight-test BH family.
- `results/appendix/figure_all_models_mech_var.svg`: all-configuration violin page.
- `results/appendix/figure_mech_var_cohens_d.svg` and CSV: effect-size panels with
  sign-within/jackknife-across inference and exploratory 94-test BH correction.
- `results/appendix/mechanistic_variability/`: additional component panels/tables.

Intermediate data and stage/source/output hashes live under
`data/analysis/mechanistic_variability/`; performance and winner tables under
`data/analysis/performance/`. Notebook-compatible caches verify selected models.
The focused report's intermediate unit-level data are in `gru_monogru_report/`.

The old all-jackknife/Holm table is retained as
`similarity_vs_standard_gru_jackknife_sensitivity.htsv`. Superseded publication
layouts are archived under `data/analysis/.local/mechanistic_variability/`.
They are not used for current stars. Historical scripts and exported examples
retain their recorded selection; ten-winner-only exports cannot be reselected
without the original inner fits.

Plot implementations remain in `code/figures/`, metric implementations in
`code/measures/`, and focused report assembly in `code/analyses/gru_monogru_report.py`.
The maintained workflow orchestrates all current outputs and records their policy.
