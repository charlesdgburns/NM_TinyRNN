# Review and validation

## Release review: 2 November 2026 (1.1.0)

The current review passes **59 tests, zero skipped**, with ORIGINAL_RL_SOURCE
and TINY_RNN_SOURCE enabled. It covers source likelihood parity, independent
equations, ML/MAP, recording-LOSO, held-out scoring, masks/reset identities,
information criteria and the public mean-model workflows. The successful log
is `validation/release_tests_20261102.log`. This is synthetic/software validation;
no new animal-data fits, RNN loading/training or scientific plots were run.

The pre-change package passed 50 tests with three optional source tests skipped
(source environment variables were not supplied). Two new equal-bound penalty
regressions then failed: the old AIC counted a two-entry fixed vector as two
free parameters, and a one-free-parameter vector as two. The fix uses the
remaining free dimension for AIC/AICc/BIC and records n_ic_params. The same
regressions now pass. Unrestricted likelihoods, parameter vectors and penalties
are unchanged; earlier saved fits/scores have not been rewritten.

The three mean-forgetting classes already existed, but were absent from the
registry/CLI. They now share the public fitting route and have independent
equation, released-step parity, endpoint, mask/reset/export and ML/MAP/LOSO
tests. The default eight-model comparison set remains unchanged. The zero-drift
source alias is corrected from the older unverified MFdec label to MB1 after
checking the saved PRLCogAgent source.

Initial reads stalled on iCloud placeholders. Sandboxed brctl download failed
with "Trying to invoke brctl from a sandboxed process". Authorized brctl
hydration outside the sandbox made the same files readable; subsequent source
tests ran in the original sandbox/interpreter. No cold-state iCloud repair is
claimed. Original files/hashes, failure logs and review changes are preserved
outside the upload folder in the dated workspace review directory.

Python source parsing and the 120-column ceiling pass. Dependency lower bounds
and all operating systems are not exhaustively tested. Finite multistart fits
do not establish global optimality; paired tests remain descriptive.

## Historical review: 29 October 2026

The deliverable is an additive `code/models/rl_models` folder. Original source
files and earlier animal/RNN fits are unchanged; no GitHub push was performed.

## Additional drift-model validation

The extended package passed **50 tests**, with both original-local-source and
Ji-An source-step audits enabled, zero skipped. The log is
`validation/final_tests.log`. Independent project-level tests add 16 passing
checks for accelerated recurrences, target timing, pooled subject scores,
source manifests, fit audits and guarded resume.

The isolated installed package also completed 32 synthetic ML/MAP fits across
all eight explicit drift registry variants, with 64 starts and no selected
nonconvergence. Dependencies remain NumPy/pandas/SciPy; PyTorch was not imported.
See `validation/drift_package_smoke.json` for that executed scope.

The five new experimental variants were separately fit to all 11 approved
`nested_DA_8` mice: full-ten-session, fixed-eight-session original-block, and
ten nine-session LOSO contexts, each with ML/MAP. All 1,320 selected fits
converged; 13,200 random starts plus five feasible same-training starts were
retained. Previous fits and raw/RNN files were not edited. These project result
tables and plots are saved separately from this portable code folder.

The project comparison reuses frozen monoGRU predictions on exactly matched
targets, with fresh source validation covering the 252 selected networks.
Initial file reads stalled and one initial metadata check failed; final checks
passed outside the sandbox after same-source checksum hydration. The cause
remains unconfirmed; no cold-state repair is claimed. The integer-manifest guard
is protective, not a demonstrated fix of that initial failure. The project
validation record retains these qualifications and the first failed analysis.

The original subset checks below remain historical evidence; its initial
validation report is preserved as `validation/REVIEW_initial_20261029.json`.

## Executed scientific checks

- **39 unittest tests passed**, with both optional original-source parity
  audits enabled and no skipped tests. See `validation/tests.log`.
- Original likelihood/metadata parity covers 21 configurations with 3 random
  parameter vectors each, plus neutral asymmetric Bayesian settings for all
  three history modes. Export-only corrections are deliberately excluded from
  claims of unchanged export-schema behavior.
- Independent equations verify reward-EMA/forgetting nesting, eta=1/single
  equivalence, removal of history, half/full history scaling, pre/post timing,
  forced-input histories, masks, Bayesian hazard/evidence order and exporters.
- ML/MAP objectives, priors/bounds, actual scored-target AIC/BIC, fixed-bound
  optimization, deterministic local RNG, recording-grouped LOSO, identity joins,
  explicit held-out scoring and subject-unit comparisons were exercised.
- The complete default eight-model synthetic example ran **ML and MAP** with
  two starts: 48 full-data fits plus 144 recording-LOSO fits, **192 selected fits**
  and 384 optimizer starts. All 192 selected fits converged; 380/384 individual starts
  converged. Unsuccessful starts remain in their attempt diagnostics.
- That example used 3 synthetic subjects, 3 recordings each, 60 trials/recording:
  540 input trials and 486 eligible free targets/model. Inputs retain forced trials;
  values/history reset each recording. Ratios pool trials within subject before
  equal-subject means/sample SEM. This is software validation, not an animal
  result or a parameter-recovery study.
- All ten supported metric families and both paired-test choices were evaluated;
  raw and per-trial extensive metrics remain separate. Small-sample undefined
  AICc and one-subject uncertainty remain explicit.
- Source hashes were rechecked against `PROVENANCE.json`. Python syntax and
  the 120-column ceiling were checked. The small final formatting change after
  the full synthetic run preserved an identical executable AST; the final
  source version also passed the 39-test workflow.

## Publication-layout and package checks

The folder was tested when copied to the intended nested repository layout,
including `NM_TinyRNN.code.models.rl_models` imports, package/flat CLI entry
points, standalone wheel construction and isolated temporary-target installation.
The same approved scientific interpreter/dependencies were used; no dependency
installation changed the analysis environment. Final machine-readable status
and installed-artifact checks are recorded in `validation/REVIEW.json`.

## Failures, fixes and changed conditions

- An uninstalled-package test launched from an unrelated working directory could
  not import `rl_models`. The documented parent-directory invocation succeeds;
  arbitrary-directory use requires installation or an explicit parent PYTHONPATH.
  The original unrelated-directory command is not claimed repaired without that
  condition. Copied/installed package paths were tested separately.
- Early direct-Session validation stringified missing IDs. Validation now rejects
  missing/non-scalar/empty identities before conversion; regressions pass.
- A too-strict Bayesian environment guard initially rejected valid neutral .5/.5
  probabilities. The guard now permits equality; source-parity regressions pass.
- The first full ML/MAP example completed with a pandas concatenation warning.
  Explicit numeric dtypes fix the cause. The **same input** was rerun with zero
  warnings and identical full-data/CV likelihoods; original output was retained.
- Pip reported an unavailable default cache and disabled it itself. No alternate
  cache, scientific kernel or input was selected to make a failing test pass.
- This workspace is not a Git checkout; Git-status inspection is unavailable.
  The deliverable is a reviewed filesystem staging folder, not a committed branch.

## Limits

The initial portable subset checks did not run experimental-data fits, RNN
training/replay, notebooks or plots. The separate later drift experiment scope
is recorded above. The existing PyTorch/OpenMP issue is irrelevant to this folder,
which never imports PyTorch. Finite multistart fitting is not a proof of global
optimality; behavioral parity is not an architectural embedding proof.

The validated environment is named `cortical_photometry_1`; exact Python,
NumPy/pandas/SciPy versions and file hashes are in `validation/REVIEW.json`.
Dependency lower bounds were not exhaustively tested across versions. Black/ruff
were unavailable in that interpreter; syntax/import and manual line/style checks
were used. Paired P values are descriptive, selected-reference and uncorrected;
no scientific significance claim is made from the synthetic example.
