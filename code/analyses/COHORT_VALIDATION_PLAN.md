# Cohort-2 replication: analysis plan (frozen before inspecting cohort-2 similarity results)

Written: 2026-11-19T00:06:31Z (UTC). At the time of writing, no author or assistant had inspected cohort-2
similarity, performance or summary outputs. Only the subject count (22) and file names were seen.

## Data and models
- Cohort 2: 22 mice, same probabilistic reversal-learning task (run `nested_5HT_S32_B8`, `data/analyses/cohort_validation/`).
- Contrast: 2-unit GRU (tanh, decoder bias) vs 2-unit monoGRU+BC-DB (ReLU, energy + sparsity, no reset gate, scalar update gate, no decoder bias).
- Fitting, nested CV, block-reset selection, lambda grid, seeds, alignment and similarity metrics: identical to cohort 1 (`nested_DA_S32_B8`); no changes after this date.

## Outcomes and tests (same code as the cohort-1 main and appendix tables)
Similarity per fit pair on the raw scale; within subject = mean over outer-fold pairs per mouse (22 values);
across subjects = mean per subject pair (231 dyads).

| Component | Level | Test | Sidedness |
|---|---|---|---|
| Weight | within | Wilcoxon signed-rank on 22 subject differences | one-sided, H1: delta > 0 |
| Hidden state | within | Wilcoxon signed-rank | one-sided, H1: delta > 0 |
| Weight | across | delete-one-subject jackknife, t(21) | one-sided, H1: delta > 0 |
| Hidden state | across | delete-one-subject jackknife, t(21) | one-sided, H1: delta > 0 |
| Logit | within | Wilcoxon signed-rank | two-sided (significant in cohort 1, but its null calibration was poor and it is not a claim) |
| Logit | across | jackknife t(21) | two-sided (not significant in cohort 1) |
| Update gate | within / across | Wilcoxon / jackknife | two-sided; reported with 95% CI, no equivalence claim |

- Rule: one-sided only for comparisons significant in cohort 1 that carry a claim (weights, hidden states).
  An effect in the opposite direction counts as a failed replication, not a finding.
- Multiple comparisons: Benjamini-Hochberg over the same 8 tests as the cohort-1 table.
- Primary endpoint: across-subject hidden-state similarity. Within-subject comparisons are secondary and
  underpowered (see below); a non-significant within-subject result is not evidence against the claim.
- Effect sizes: delta [95% CI] and d_z = delta/(jackknife SE * sqrt(n)), reported next to cohort-1 values.

## Behavioural prediction
- Held-out NLL per mouse (free-choice-weighted eval_CE_computed over outer winners); delta = monoGRU - GRU.
- TOST with a fixed margin of +/-0.0074 nats/trial (5% of the cohort-1 GRU gain over chance, 0.148 nats);
  equivalence if the 90% CI lies inside the margin. Mean delta, 95% CI and k/22 mice also reported.

## Design sensitivity (safeguard power; Perugini, Gallucci & Costantini, 2014)
Effect = lower bound of the 60% CI of each cohort-1 effect. Simulated with the tests above (4,000 replicates),
resampling cohort-1 subject differences (within) or subject effects and residuals (across), N = 22, alpha = 0.05 one-sided:
weights within 86% (delta 0.11), hidden within 73% (0.045), weights across >99% (0.28), hidden across 93% (0.15).
NLL TOST power at the cohort-1 mean difference: ~99%.
