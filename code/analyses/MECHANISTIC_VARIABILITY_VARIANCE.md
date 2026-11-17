# Variance in mechanistic similarities

Run from the repository root:

```bash
.venv/bin/python -m pip install -r code/analyses/requirements-variance.txt
.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections variance
# If any saved sampling diagnostics fail, retry only those fits:
.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections variance --variance-stage retry
```

## Maintained publication outputs

Only these four figures are generated in
`results/appendix/mechanistic_variability/variance/`:

- `figure_variance_breakdown.svg`: exact raw-score proportions above absolute variances.
- `figure_hierarchical_variance.svg`: proportional scatterplots above absolute
  scatterplots, showing posterior medians and 95% credible intervals.
- `figure_bayesian_evidence.svg`: updating for subject variance exceeding fitted variance.
- `bayesian/figure_question1_variance_contributions.svg`: all component shares and
  absolute values, with credible intervals.

All numerical summaries, posterior arrays and sampling/predictive diagnostics
remain in `data/analyses/mechanistic_variability/appendix/variance/`. Plotting
writes `bayesian_summary.csv`, `hierarchical_absolute_variance.csv`,
`question1_variance_contributions.csv` there.
It does not export duplicate CSVs or diagnostic figures to `results/`.
Previously computed robust summaries and inclusion evidence remain available as
intermediate data; the standard report no longer recomputes or plots them.
Default appendix execution reuses saved posterior draws. To explicitly fit,
use `--variance-stage all`; to retry failed fits, use `--variance-stage retry`.

The default is four NUTS chains, 1,000 warmup and 1,000 retained draws per chain,
target acceptance .95, seed 20261116. The appendix script contains
`run_variance(argv)` for focused numerical calls with `--stage descriptive`,
`bayesian` and `plots`. The publication CLI uses `--variance-stage` instead.
The numerical function's `--metric` restricts Bayesian fits;
it does not restrict the descriptive analysis. Identical input/code/settings
fingerprints reuse completed fits. A restricted Bayesian run replaces summary
CSVs with that run's subset; use a separate `--output` for pilot runs.
Fits with divergent retained transitions are automatically retried at target
acceptance .99; the actual target is recorded per fit. Across-subject fits use
exact Gaussian marginalisation for sampling efficiency. Each balanced 10x10
dyad matrix is orthogonally decomposed into its grand mean, row/column contrasts
and interaction residual. Gaussian effects and the intercept are integrated out;
only five SDs are sampled. This preserves the specified likelihood and priors.
A test compares this likelihood against a direct dense covariance calculation.
Conditional Gaussian effects are drawn after sampling for predictive checks.
The retry stage keeps the original input fingerprint and refits only failed
cases with doubled warmup/draws and target .99, recording retry settings and
the updated implementation hash separately.

## Inputs and selection

Requires `nested_DA_S32_B8`, standard two-unit GRU (tanh, decoder bias) and
monoGRU (ReLU, no decoder bias), 11 subjects and ten outer winners each.
All pair endpoints are checked against the saved inner-fold winners and the
block-reset selection metadata. No reselection, training or modification of the
primary similarity cache occurs. Input hashes, selected-model and pair snapshots
are recorded in `data/analyses/mechanistic_variability/appendix/variance/`.

Within: 495 scores/configuration, 45/subject. Across: 5,500 scores/configuration,
100/subject dyad. The four shared metrics are weight cosine, logit Pearson,
mean unitwise hidden Pearson and flattened update-gate Pearson. These are not
interchangeable measures. Gate dimensions differ between architectures.

## Exact descriptive decomposition

On the original score scale:

`Var(score) = Var(group mean) + mean(within-group squared deviation)`.

All variances use denominator N; group weighting is by count, preserving this
identity even for unequal group sizes. Within groups are subjects; across groups
are ordered-as-saved subject dyads. Report the mean similarity alongside the
variance: consistently low similarity can have little variance. Subject-deletion
ranges are descriptive sensitivity ranges, not confidence intervals. Endpoint
role summaries can be joined to selected NLL/regularisation/seed metadata.

The within-group part is further decomposed by an exact orthogonal projection:

- Within each subject's complete 10-model graph, fit `score_ij=mean+u_i+u_j+e_ij`
  with sum(u)=0. Endpoint effects are `u_i=9/8*(mean incident score_i-grand mean)`.
- Within each across-subject 10x10 matrix, fit grand mean + row effect + column
  effect + interaction. Row/column effects are estimated separately per dyad.

Between-group, additive-endpoint and residual-interaction squared contributions
sum exactly to the total observed variance. This projection is finite-sample and
descriptive: estimated endpoint effects contain sampling/pair variation. Across
endpoint effects can vary with partner subject, unlike the global fitted-model
effects in the Bayesian model. No population variance claims are attached to it.

The descriptive plots label these components "Between subjects", "Outer fold
effects" and "Pair-specific effects". Across-subject groups are subject pairs.
The outer-fold label refers to variation among selected fitted solutions, not
isolated training-split or seed variation. `figure_variance_breakdown.svg` places
the percentage decomposition above the absolute variances with matching bar
positions. All absolute panels use the same y-axis; the common limit is derived from the
largest observed total variance.
Model hue is blue for GRU and orange for monoGRU. Dark/medium/light shades
distinguish between-subject, outer-fold and pair-specific contributions,
respectively, using white-blended tints equivalent to opacity 1/.65/.30.
The legend is a two-row by three-column colour grid, with GRU/monoGRU row
labels and bold variance-component column headings.
## Hierarchical model

The likelihood is Gaussian on `atanh(score)` (clipped only at 1-1e-7 if necessary).
This is a bounded-score modelling transformation, **not** the textbook Fisher
correlation sampling-error model: no trial-count SE is assigned, and it is also
applied to cosine and averaged correlations. Population variance fractions are
on this transformed scale and must not be equated with raw descriptive fractions.

Within: `z_sij = mu + a_s + u_si + u_sj + v_si - v_sj + error`.

Across: `z_stij = mu + a_s + a_t + c_st + u_si + u_tj + v_si - v_tj + error`.

All latent effects are independent zero-mean Gaussian draws with separate SDs:
subject a, subject-dyad c, fitted-model common effect u, fitted-model role contrast
v, pair residual. A fitted model has the same identity when it appears as A or B.
The u/v parameterisation permits positive or negative covariance between its
reference-role and candidate-role effects, while assuming equal role variances.
Population score-variance contributions are respectively
`[sd_a², 0, 2sd_u², 2sd_v², sd_error²]` within, and
`[2sd_a², sd_c², 2sd_u², 2sd_v², sd_error²]` across.
Fractions divide each contribution by their sum. These are population-model
contributions, not an exact ANOVA of the finite observed subjects.

Priors: intercept Normal(0,3); each component SD HalfNormal(1) in transformed
units. Standardised latent effects Normal(0,1). Independent fits by architecture,
scope and metric. Saved posterior arrays retain chain and draw dimensions for
audit. Report medians and equal-tail 95% credible intervals, sampling diagnostics
and conditional posterior predictive checks of raw mean/SD/quantiles/skewness and
between-group variance fraction. Sampling adequacy requires zero divergences,
max Rhat<=1.01 and minimum bulk ESS>=400 over all monitored parameters.

## Evidence and sensitivity

The implemented Bayes factors concern **interval hypotheses**, not exact zero
variance. The regions are subject or fitted-solution variance fraction exceeding
10%, 25%, or 50%, and subject fraction exceeding fitted-solution fraction. The
fitted-solution contribution combines u and v. The BF for a region versus its
complement is posterior odds divided by prior odds under the encompassing model,
equivalent to comparing priors truncated and normalised to the two regions.
Thresholds are exploratory sensitivity choices, not established meaningful cutoffs.

Prior probabilities use 200,000 independent draws. No infinite or zero BFs are
reported from empty Monte Carlo tails: both regions must have at least 20
effective expected draws, otherwise the BF is omitted. These are not point-null
Bayes factors for a variance component's existence. Posterior probabilities alone
are not called BFs. No p-values or BH correction are applied.

SD prior scales .5 and 2 are evaluated by importance reweighting, holding the
intercept/latent priors fixed. ESS must be >=400 and >=20% of retained draws for
a sensitivity result to be considered usable. This is approximate reweighting,
not independent refitting; failed overlap requires a fresh fit before inference.
Architecture comparisons report the monoGRU/GRU fitted-solution variance ratio
and the difference in its variance fraction, with intervals. Draws are paired
independently across architecture fits; no claim of paired subject-level
architecture-effect inference is made by this comparison.

## Interpretation and limits

Outer winners differ in overlapping training data, selected inner fold,
regularisation and initialisation. Fitted-solution variation does **not** isolate
random-seed or pure fold effects. Fold numbers are not common experimental
conditions across subjects. Residuals may include alignment, pair interactions,
unmodelled split dependencies and nonlinear mechanistic classes.

The saved direction is B aligned to A, both evaluated on A's saved sequence.
Across subject order is fixed; the dyad effect includes reference-data and
directional compatibility. Subject effects assume exchangeable endpoints after
the dyad term; these data cannot identify reference-data versus biological
subject effects. The role contrast allows endpoint-specific fitted-model
behaviour but does not replace an A-on-B/B-on-A sensitivity analysis. Greater
mechanistic variation may coexist with equivalent predictive performance.

There are only 11 subjects. Credible intervals/BFs are conditional on likelihood,
exchangeability and priors, not assumption-free evidence. Poor predictive checks
make hierarchical results provisional even when MCMC converges. In particular,
Gaussian transformed scores can miss multimodal gate mechanisms. Start with the
exact descriptive decomposition; use a mixture or richer residual structure
only when the diagnostics justify it.

The four maintained SVGs live in
`results/appendix/mechanistic_variability/variance/`; intermediates and posterior
arrays live only in the analysis data directory. Existing main tables are unchanged.

## Bayesian presentation

`figure_bayesian_evidence.svg` asks whether subject variance exceeds the combined
fitted-solution variance. Dashed lines show the prior probability (18.5% within,
29.3% across); points show the posterior probability. Bayes factors quantify the
change in odds, not the posterior odds alone. Thus BF>1 need not imply posterior
probability>50%. Undefined finite-tail estimates remain marked unresolved.
These figures remain exploratory because of the predictive discrepancies.

`bayesian/figure_question1_variance_contributions.svg` retains all sources in a compact layout,
combining fitted common and role-specific variance into "Outer-fold fit".
This labels variation among selected fitted solutions, including split,
selection and initialisation differences, rather than an isolated split effect.
Combining the display does not change the fitted model. Within
subjects, model-pair deviations are represented by the residual. With one score
per model pair, an additional independent Gaussian pair intercept cannot be
separated from Gaussian residual variance: only their sum would be identifiable.
The across-subject dyad component instead refers to a pair of subjects, shared
by 100 fitted-model comparisons. Structured compatibility among fitted-model
pairs is not separately modelled by the current independent residual.
The compact plot labels this component "Model-fit pair (residual)". This names
the level of the remaining deviations; it does not attribute all of them to a
single known mechanism. Pair compatibility, alignment behaviour, model
misspecification and reference-input dependence can contribute. A structured
pair predictor (for example, a mechanistic-class matching term) could explain
part of it, leaving a smaller residual. Renaming the plot does not add such a
predictor or change the fitted likelihood.
The contribution figure places within/across proportional point-and-interval
panels above matching absolute point-and-interval panels. Points are posterior
medians and intervals are 95% credible intervals. All absolute panels share one
y-axis in z², z=atanh(similarity); these are not percentages or raw-score
variances. Numerical tables are separate CSV outputs. The dedicated question-1
CSV includes both the fraction and absolute variance summaries.

Absolute values use the same posterior draws before division by total variance.
Their units are squared atanh(similarity); applying tanh to a variance does not
recover a raw-scale variance contribution. Numerical values remain in
`hierarchical_absolute_variance.csv` in the intermediate data folder.

`figure_hierarchical_variance.svg` places proportional point-and-interval panels
above matching absolute point-and-interval panels, with within/across subjects
shown separately. Each point is a posterior median with an equal-tail 95%
credible interval. All absolute panels share a y-axis in squared
atanh(similarity). Tables are not embedded. Numerical summaries remain in the
intermediate data folder.

The question-1 figure covers all four shared similarities and both scopes.
Points and error bars show posterior medians and equal-tail 95% credible
intervals; its absolute panels share a scale. Numerical values are in
`question1_variance_contributions.csv` in the intermediate data folder.

## Inclusion-versus-exclusion Bayes factors

```bash
.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections evidence
```

This optional numerical analysis is not part of routine figure generation.
These comparisons test exact absence of components, unlike the earlier interval
hypotheses. For each architecture, scope and measure the full model is compared
against:

- `no_fitted_solution`: both shared fitted-model and role-contrast SDs fixed to zero;
  subject and (across) dyad effects retained.
- `no_additive_subject`: additive subject SD fixed to zero; fitted effects and
  (across) dyad effects retained. This is not removal of every form of subject structure.
- `no_subject_or_dyad` (across only): both additive subject and dyad SDs fixed to zero;
  fitted-model effects retained.

Retained SDs have the same independent HalfNormal(1) priors in every model.
The intercept has the same Normal(0,3) prior and is integrated out. Dropped
components are absent parameters, not merely assigned a small positive SD.

Gaussian effects are integrated analytically. Orthogonal projections compress
the within complete graphs and across bipartite matrices without discarding
likelihood constants or residual degrees of freedom. Tests compare the result
against direct dense Gaussian densities for full and all reduced models.
Nested sampling integrates only the remaining 2–5 component SDs, using
`dynesty==3.2.0`, 400 live points, rwalk proposals, multi-ellipsoid bounds, and
stopping dlogZ=.05. Every evidence is estimated twice with independent seeds.
Combined evidence averages Z, not logZ. Repeat spread is checked against three
times the quadrature sum of reported nested-sampling logZ errors.

`log10_BF = (logZ_full-logZ_reduced)/log(10)`: positive favours inclusion,
negative favours exclusion. Numerical SEs combine the full/reduced evidence
errors; these quantify integration uncertainty, not scientific uncertainty or
model adequacy. Huge BFs are stored on the log scale to avoid overflow.
The GRU/across/weights/no-subject-or-dyad case narrowly failed initial repeat
agreement and was repeated with 800 live points, dlogZ=.025 and independent seed
20261117. Those two higher-accuracy runs supersede its initial runs; actual
settings are recorded in the NPZ metadata, CSV and `refinements.json`.
Sensitivity to HalfNormal(.5) and HalfNormal(2) priors uses the weighted nested
posterior draws to reweight evidence. Require effective posterior weights >=400
and >=20% of baseline effective weights; this sensitivity calculation is an
approximation, not independent nested runs under each alternative prior.

Intermediates and raw runs live in
`data/analyses/mechanistic_variability/appendix/variance/inclusion_evidence/`.
Inclusion-evidence CSVs remain in that intermediate folder; the maintained
figure set does not export inclusion-evidence tables.

As an additional numerical check, a small two-SD model was integrated by adaptive
two-dimensional quadrature: logZ=-10.79110, versus nested sampling -10.75102 with
reported SE .05746 (difference .70 SE). This check and its synthetic data are
preliminary validation, not an analysis of the neural-network fits.
Full-model nested-sampling posterior mean variance fractions also agreed with
the independent NUTS estimates to within .0095 (less than one percentage point)
over all architectures, scopes, measures and components. Numerical agreement
does not resolve the predictive model-fit discrepancies.

All evidence remains conditional on the same Gaussian transformed-score model
that showed predictive discrepancies. Inclusion BFs do not establish that this
likelihood is correct or that a statistical component has a biological cause.

Method: [Speagle (2020), dynesty](https://arxiv.org/abs/1904.02180);
[nested-sampling error documentation](https://dynesty.readthedocs.io/en/latest/errors.html).

References: [PyMC sampling](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html),
[multiple-membership effects](https://paulbuerkner.com/brms/reference/mm.html),
[prior sensitivity of Bayes factors](https://arxiv.org/abs/2206.05210).
