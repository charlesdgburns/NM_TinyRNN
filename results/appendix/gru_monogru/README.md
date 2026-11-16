# Mechanistic variability: current reporting methods

## Selection

Minimum free-choice train+validation NLL on original saved blocks; reset to learned initial state per block; no test targets or penalties.

## Evaluation

Continuous complete subject A saved inputs, including forced trials; B aligned to A; no block/session resets during similarity evaluation.

## Alignment

Tanh permutation/sign search; ReLU grid winner then continuous positive-scale refinement; main-parameter Euclidean objective.

## Scores

All-trainable-parameter cosine; mean valid unitwise hidden Pearson; flattened update/reset gates and logit-difference Pearson.

## Aggregation

Within: 45 fold pairs -> one mean per subject (11); across: 100 fold pairs -> one mean per dyad (55, sharing 11 subjects).

## Within Test

Exact two-sided sign test of direction/median of independent subject paired differences; not a mean test.

## Across Test

Mean dyad difference; delete each subject and all incident dyads; jackknife SE; approximate two-sided t(10).

## Corrections

BH across 8 focused GRU/monoGRU tests; separate BH family of 94 exploratory all-configuration contrasts; no reference/absent-gate tests.

## Effect Size

Supplementary Cohen dz: paired mean/SD of 11 subject differences; across uses 11 overlapping incident-dyad means and is descriptive only.

## Deletion Ranges

Within median and across mean deletion ranges, plus explicit across median deletion range; sensitivity ranges, not confidence intervals.

## Highlighting

Within stars annotate median direction difference; across stars annotate mean similarity, not median differences.

## Limitations

11 subjects; shared outer-fold models; across inference approximate; exploratory comparisons; no normality pretest is used to select tests.

## Reproduction

From the repository root: `.venv/bin/python code/analyses/mechanistic_variability.py --stage statistics`, then `--stage plots`. Omit --stage for the full verified workflow.

## Outputs

Main: results/main/table_1.svg. Focused supplementary tables and four paired plots: results/appendix/gru_monogru/. All-configuration violin page and Cohen dz matrix: results/appendix/. Extra per-component panels: results/appendix/mechanistic_variability/. CSV data and manifests accompany the figures.

The 8-test and 94-test families answer different scoped questions and can yield different q-values for the same raw contrast. Do not mix significance annotations between them. The all-jackknife/Holm legacy table is retained under similarity_vs_standard_gru_jackknife_sensitivity.htsv and is not used for current stars.
