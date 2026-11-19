# Average activation contrast: ReLU minus tanh

Primary analysis: continuously refined ReLU and permutation/sign-only tanh, without balancing. Average the six matched architecture/decoder-bias configurations equally within each subject or dyad; then average subjects or dyads equally. Positive differences indicate higher ReLU similarity.

Within-subject inference is the paired subject-mean t procedure. Across-subject inference uses delete-one-subject jackknife SE: sqrt((n-1)/n * sum((deleted_mean - mean(deleted_means))**2)); deleting a subject removes all incident dyads across all configurations. Both use t(n-1) intervals (n=11). Across-subject inference is approximate, assumes independent sampled subjects, and may be unreliable with few subjects or a degenerate dyadic kernel. Folds, configurations and the 55 dyads are not treated as independent replicates. Inference concerns the average across these six fixed configurations, not all possible architectures.

The superiority test is one-sided H0: delta <= 0. Report individual 95% and 90% intervals; p-values are also Holm adjusted across the two scopes separately within each alignment version. Discrete alignment is a sensitivity analysis, not a separate confirmatory replication.

TOST margin: 0.05. For a supplied positive margin m, test delta > -m and delta < m; p_TOST is the larger one-sided p-value. Individual equivalence at alpha=.05 requires the 90% interval entirely within (-m,m). Across-subject TOST inherits the jackknife approximation. The margin was supplied by the user for this exploratory analysis, after prior summaries had been inspected; it is not preregistered. Equivalence of the mean contrast does not establish metric equivalence or equivalence for every model configuration.

| Scope | Mean difference | 95% CI | 90% CI | One-sided p (ReLU greater) | TOST p |
|---|---:|---|---|---:|---:|
| across_subject | +0.03363 | [-0.15225, +0.21951] | [-0.11757, +0.18483] | 0.3477 | 0.4242 |
| within_subject | -0.04173 | [-0.14140, +0.05793] | [-0.12281, +0.03934] | 0.8136 | 0.4285 |

TOST reference: https://aaroncaldwell.us/TOSTERpkg/reference/t_TOST.html

Reproduce: `.venv/bin/python code/analyses/weight_similarity_sensitivity.py --stage contrasts --equivalence-margin 0.05`
