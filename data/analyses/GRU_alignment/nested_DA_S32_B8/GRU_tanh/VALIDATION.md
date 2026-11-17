# Additions-only GRU publication validation

The publication adds GRU files only. No pre-existing GitHub file path, data,
code, monoGRU artifact or source checkpoint is replaced or deleted. Baseline:
`f36568933d5cd937e3944092c63998e46a458819` (main), whose parent is the requested
`d5e02f6390e911f0eef6cc830f0dc2b2454e7af7` export commit.

## Data use and methods

Source run is `nested_DA_S32_B8`, published `subject_examples_latest`. Subjects
are WS01, WS02, WS05, WS08, WS09, WS10, WS13, WS14, WS16, WS20 and WS22.
All 110 two-unit tanh GRUs with decoder bias are retained: ten published outer
winners per mouse, with exact stored inner indices and original minimum
block-reset train+validation selection. No refitting, test-based reselection,
missing-fit substitution or exclusions are performed.

Each trial row is an after-current-input readout in one exact fitted-model
trajectory, predicting the next choice. Whole-subject histories remain
continuous across session, block and split boundaries. There are 84,861 distinct
subject trials and 848,610 rows across the ten separate fits, with no lags,
resets or trial aggregation. All source artifact and raw-session hashes and
original trial/split identities are verified.

Every model has positive A1-minus-A2 decoder contrast for h1, and negative
contrast for h2, matching h1 -> A1/left/0 and h2 -> A2/right/1. Alignment searches
only the two signed permutations satisfying those labels, using the native core
parameter-distance objective and equal total subject weights. No positive
ReLU gain balancing is applied to tanh. Candidate, recurrent, initial-state,
gate and decoder parameters transform together; both gate outputs are permuted
without sign flips, and decoder bias stays unchanged.

The medoid is WS10, outer fold 8, stored inner index 4. All 110 fits were
reference-eligible. The remaining globally equivalent action-labelled
orientation is fixed by fewest reference-source sign flips and stable
permutation order. This is a declared coordinate convention.

The 44-entry parameter summary averages folds within mouse, then mice equally.
Subject SD/SEM describe the eleven mouse means. The parameter-mean pickle is
descriptive, not a fit or an ensemble; no inferential analysis is performed.

## Additions-only implementation

The initial local export used an optional constraint parameter added to the
existing alignment function. For this publication the identical function body
is placed in the new `code/measures/gru_alignment_constraints.py` module, which
imports the unchanged native gate-transform helper. New GRU scripts/tests import
that module. All 31 existing upstream Python files were checked against their
original Git blob hashes; `mechanistic_variability.py` is unchanged.

The original local export and ZIP remain preserved. The publication was rebuilt
in a separate staging directory using the same interpreter, source bytes,
cohort, fit identities, scientific objective, transforms and parameters. Its
entire 110-row alignment audit table exactly matches the original validated
table, including permutations, signs, reference distances and synthetic errors.

## Executed checks

Interpreter: `cortical_photometry_1` Python 3.11. Commands run sequentially from
the publication repository root:

```bash
python -m unittest discover -s code/tests -v
python code/analyses/GRU_alignment.py
python code/analyses/export_aligned_GRU.py --jobs 4
python code/tests/verify_gru_release.py
```

All 16 focused tests passed, including original monoGRU alignment/export tests,
joint hard constraints and their identity fallback, equal-subject medoid
weights, global-orientation invariance, both GRU gates, learned initialization,
decoder bias, serialization and all eight synthetic signed permutations.

All 110 regenerated exports passed hidden-state, both-gate, probability/logit,
trial/split identity and serialization checks. All 336 manifest-listed export
artifacts were independently reloaded and hash-verified; all original source
hashes remain unchanged. Maximum absolute source-versus-regenerated errors:

| Variable | Maximum error |
|---|---:|
| Hidden states | 1.34535012e-6 |
| Update gates | 8.98742676e-7 |
| Reset gates | 3.81800843e-7 |
| Logit | 2.03460389e-6 |
| Probability | 4.26805878e-7 |

Tolerance is atol 5e-5 / rtol 2e-4 for full-history source comparisons. Synthetic
symmetry checks had maximum error 5.96e-8. The minimum h1 decoder contrast is
0.89283943 and the maximum h2 contrast is -0.23699442. No model violates the
required labels.

## Observed interruptions and limits

Earlier source downloads needed DNS retries, using unchanged inputs and hash
checks. During publication staging, sandboxed reads and macOS fast-copy stalled
on iCloud placeholders; those incomplete copies were interrupted. Host access
and explicit parallel reads copied the same original paths, and source hashes
were rechecked before analysis. This is staging under changed filesystem access,
not a claim that the original sandboxed copy path passed. One premature test
discovery, while staging was incomplete, found zero tests and was not counted
as validation; the final complete build ran all 16 tests above.

GitHub CLI authentication appeared invalid within the network-restricted
sandbox; the same account passed authentication and write-access checks with
host network access. No credentials were changed.

Matplotlib/Fontconfig cache-permission warnings remain visible. No cache setting
was changed, no figures were rendered, and no plotting validation is claimed.
Black was unavailable in the approved environment. Validation covers this exact
cohort/source/configuration, not unrequested architectures or training paths.
The separate modelling-reversal aligned-example importer remains monoGRU-only;
these native GRU pickle/HTSV files can be loaded directly.
