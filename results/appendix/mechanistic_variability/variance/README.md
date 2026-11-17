# Mechanistic similarity variance

Dataset: nested_DA_S32_B8; block-reset-selected outer winners.
Standard GRU (tanh, decoder bias) versus monoGRU (ReLU, no decoder bias).

- figure_variance_breakdown.svg: exact descriptive raw-similarity variance;
  percentage breakdown above absolute variance, with shared absolute axes.
- figure_hierarchical_variance.svg: proportional scatterplots above absolute
  scatterplots, with posterior medians and 95% credible intervals. All absolute
  panels share a y-axis; no tables are embedded.
- figure_bayesian_evidence.svg: prior-to-posterior updating for subject variance
  exceeding fitted-solution variance. Bayes factors concern this inequality,
  not exact-zero component inclusion.
- bayesian/figure_question1_variance_contributions.svg: proportional and absolute
  component estimates with 95% credible intervals for each similarity measure.

Hierarchical absolute units are squared atanh(similarity), not raw-score
variance. Outer-fold fit combines common and role-specific fitted effects;
model-fit pair is the residual level, not a demonstrated causal mechanism.
These exploratory Gaussian models leave posterior predictive discrepancies.
Similarity-score variance is not a direct measure of parameter reproducibility.

Numerical summaries, posterior draws, diagnostics and provenance remain in
`data/analyses/mechanistic_variability/appendix/variance/`. See
`code/analyses/MECHANISTIC_VARIABILITY_VARIANCE.md` for the model specification.
Generate these figures using:
`.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections variance`
