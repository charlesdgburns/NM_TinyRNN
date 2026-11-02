# Changes relative to the original local RL folder

## Release review, 2 November 2026 (1.1.0)

- Exposed the existing three mean-forgetting variants through the public API
  and registry/CLI, and added unbiased Q_basic to the registry. Defaults remain
  the same eight primary models; the underlying value equations are unchanged.
- Added the two already-supported matched-zero/no-decrement drift variants to
  DRIFT_MODELS so that this opt-in set covers all ten drift registry names.
- Corrected AIC/AICc/BIC for equal-bound controls to count only remaining free
  parameters. Full parameter-vector metadata and unrestricted fits are unchanged.
  Added n_ic_params; saved earlier scores are not rewritten.
- Recorded drift rule, reward decrement flag and mean initialization in run
  manifests. Corrected the zero-drift source alias to MB1.
- Added regression tests for the penalty bug, mean equations/source parity,
  mask/reset/export behavior, and saved ML/MAP/LOSO workflows.
- Added a complete model catalogue and clean release packaging. No new animal
  fits or RNN operations were run during this release review.

## Additional drift models, 29 October 2026

- Independently implemented Ji-An et al. (2025) drift-zero and drift-other
  equations, retaining zero initialization and the source's unclipped states.
- Added a matched rewarded-rule omission-zero control and optional side bias,
  single history or full signed EMA history. These extensions are separated
  from the published cores in `DRIFT_MODELS.md`.
- Added equation, source-step, parameter-domain, timing and equivalence tests.
  The existing default comparison set and all prior likelihoods are unchanged.

## Initial clean subset

2026-10-29: additive, self-contained publication subset.

- Retained only the requested Q and 2AB Bayesian model families, plus the
  consistently parameterized reward-EMA/history/forgetting comparison families.
- Original fitted equations, parameter orders, priors, likelihood floor and
  legacy half-scale Q EMA are preserved and checked against unedited source.
- `BayesAsym` aliases `BayesAsym_2AB`; no unrelated Bayes/BRL/OpAL/meta classes.
- Corrected Bayesian variable-export probabilities to use the fitted belief
  contrast, and replaced the asymmetric return-Q export's repeated final belief
  with actual trial-specific expected values. These exporter changes do not
  change likelihoods. All trial-variable DataFrames now use the documented
  common `('model_vars',variable,'')` group; old Bayesian-specific group names
  are an intentional schema migration, not claimed export-schema parity.
- Added a shared forward path: pre-choice and post-input states, full logits,
  likelihood probabilities and exported variables cannot disagree silently.
- Added explicit binary/mask/identity/domain checks. Contradictory history flags,
  missing input fields and invalid data raise rather than assume plausible values.
- Added portable array/CSV Sessions instead of importing private data classes.
- Kept forced inputs while respecting target masks; information criteria count
  actual scored targets rather than all nominal free trials.
- Added deterministic local RNG and complete optimization diagnostics. Random
  starts share the original distributions, not necessarily its exact RNG stream.
- Grouped LOSO by physical recording, keeping all its reset blocks together.
  CV now honors the requested MAP_fit/restarts/options; the old fit_sessions
  helper silently requested ML CV even from a MAP call.
- Added matched-target/hash checks, subject-unit SEM, raw/per-trial metrics and
  explicit descriptive paired-test labels. Comparison returns inspectable tables
  instead of automatically creating a plot; fit_subjects(plot=True) is rejected
  with an explicit message to use a separate rendering stage.
- Removed absolute developer paths, import-time plotting, unsafe restoration,
  circular fitting/comparison imports and mutable default parameters.

These changes do not reproduce the RNN data preparation/selection pipeline or
retrain any RNN. The previous animal fits are unchanged. Packaging tests use
synthetic behavior; equations/parity tests do not establish global optimization,
model identifiability or exact monoGRU representability of the full models.
