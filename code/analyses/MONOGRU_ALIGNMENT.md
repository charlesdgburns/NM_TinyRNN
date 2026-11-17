# A balanced reference for two-unit monoGRU +BC-DB models (ReLU, no decoder bias)

Run from the repository root:

```bash
.venv/bin/python code/analyses/monoGRU_alignment.py
```

The default source is `data/analyses/mechanistic_variability/selected_models.htsv`, filtered to two-unit monoGRU/ReLU with decoder bias **off**. It contains 110 selected models from `nested_DA_S32_B8`: 11 subjects × ten outer folds. Selection is the existing block-reset train+validation criterion. This script neither selects on test behavior nor trains models. It rejects a run-name mismatch, missing folds, bias-on model objects, and duplicate winners.

For another source, provide **both** its selected table and run name:

```bash
.venv/bin/python code/analyses/monoGRU_alignment.py \
  --folder nested_DA_last10_P100_S32_B8 \
  --selected-table /absolute/path/to/that/run/selected_models.htsv
```

The monoGRU mechanisms notebook currently requests the patience-100 run. Do not assume its cohort matches the default S32/B8 cohort in this script. Shared performance caches can point to other runs, so this script reads the explicit selected table and source model files directly.

## Recommendation and justification

Use independent minimum-norm scaling, a fixed readout-based unit label, and a data-driven **medoid** as the representative reference. These solve different problems:

1. **Fix scale independently of the example.** ReLU allows positive hidden-unit scaling with compensating changes to input, recurrent, readout, gate-input and initial-state parameters. We minimize the sum of squares of **all trainable parameters** along each model's scale orbit using the convex log-scale canonicalizer extracted from `canonical_weight_comparison.py` into the reusable `code/measures/monogru_balancing.py` helper. This prevents an arbitrary scale of one notebook model from determining the coordinates of the entire cohort. It does not imply the resulting scale is biologically privileged. Invariant parameters add constants to the objective; the numerical tradeoff otherwise uses the repository's parameter units and input encoding.
2. **Give units a semantic label.** Sort units by decoder contrast `decoder.weight[1,j] - decoder.weight[0,j]`, after balancing. When contrasts have opposite signs, h1 supports choice 1 and h2 supports choice 0, matching the attached diagram's decoder column convention. Decoder common offsets cancel in this contrast. No output labels or ReLU signs are flipped. Neither positive scaling nor a permutation can manufacture opposing signs when the model does not have them. Same-sign/zero-contrast models are retained and flagged, with a deterministic descending-contrast order but no claim that both semantic roles exist. Near-zero contrasts use an absolute 1e-7 threshold for the opponent flag; exact ordering ties retain the original order.
3. **Choose a real central model.** Among finite-balanced opponent models with both units active on fixed synthetic probes, choose the weighted medoid: the actual model minimizing weighted mean Euclidean distance to **all** cohort models in balanced, labeled coordinates. Every subject has equal total weight and each outer fold has equal weight within its subject. The metric uses the same core blocks as `mechanistic_variability.align_rnns`: candidate input, candidate bias, recurrence, and decoder weights. Gate parameters and initial states are balanced and transformed, but excluded from this distance, as in the existing alignment objective. Decoder bias is also excluded from that distance, and is absent from every model in this cohort. Mean *distance*, rather than squared distance, reduces leverage of very distant fits. An actual medoid avoids presenting an average of different recurrent systems as a fitted model.
4. **Fit each fold to the frozen balanced medoid before averaging.** After choosing the medoid in independently balanced, semantically labeled coordinates, call `mechanistic_variability.align_rnns` with its normal positive-scale grid and continuous TRF refinement. The decoder-based unit permutation is constrained so h1/h2 keep their semantic roles. The objective is Euclidean distance over candidate-input, candidate-bias, recurrence and decoder weights. Gate parameters and initial hidden states are transformed consistently, although excluded from that distance. The medoid itself is held fixed. Final scales compose the initial balancing and reference-fitting steps, and are recorded with the original-to-final permutation.

This **reference-fitting step is now performed before averaging**. Initial balancing defines the medoid-selection space and provides a deterministic starting representation; the final exported models need not remain at their independent minimum-norm scales. A different reference can change final hidden scales and mean parameters. Model predictions are invariant under these positive rescalings. Prior exports before protocol `balanced_medoid_reference_fit_v2` used independent balancing and semantic labeling only; those are preserved in a local backup and superseded by the current exports.

A hand-selected scale is reasonable only when there is an independently motivated physical unit or explicit visualization requirement. Here it would introduce an arbitrary convention. Data-driven parameter averaging could be a useful descriptive summary **after** balancing and labeling; it is not a fitted or function-equivalent model and is not used as the reference.

## Degeneracy and preservation checks

The full-norm minimizer is finite and unique only with sufficient nonzero incoming and outgoing scale anchors. The reused canonicalizer detects missing positive/negative per-unit anchors. Such models receive bounded log scales in [-8,8], are flagged `unanchored_bounded`, and cannot be reference candidates. This approximation is explicitly not a unique scale-invariant canonical form. Opponent/activity flags are eligibility criteria, not exclusions from the population distance objective. A model with no finite parameters or failed numerical checks stops the run.

Each original/aligned pair is checked on 32 deterministic synthetic binary histories of length 64: 24 pseudorandom histories plus all eight constant forced-choice/action/outcome inputs. This uses no observed held-out outcomes. Logits and update gates must be unchanged, and hidden states must equal the saved positive-scale/permutation transformation. Synthetic checks are numerical checks of an algebraic symmetry, not exhaustive behavioral validation. The tolerances are atol=rtol=2e-4 in float32; actual errors are recorded. All source files are unchanged.

Tests cover invariance to prior scaling/permutation, nonzero learned initial state and gate transformations, unanchored models, equal cohort weights, and hard permutation constraints. Existing alignment refinement tests are also run.

## Outputs and reuse

Outputs live in `data/analyses/monoGRU_alignment/<folder>/monoGRU_relu_ndb/`:

- `reference_model.pickle`: the frozen balanced medoid, with original learned gate biases intact and no decoder bias.
- `aligned/`: transformed copies of all source models, separated by subject, configuration and outer fold.
- `aligned_models.htsv`: source identities, output paths, composed original-to-final permutations/scales, separate balancing and reference-fitting scales, before/after reference distances, norms, readout flags and numerical validation errors. Scales are ordered in **output** unit order, so `h_aligned = h_original[..., perm] * scales`.
- `balanced_core_parameters.npz`: labeled parameter vectors used in the medoid objective.
- `leave_subject_out.htsv`: medoid identity and distance to the full-cohort reference after removing each subject and recomputing equal weights.
- `reference_parameters.png` / `.svg`: candidate, gate and decoder parameter overview.
- `manifest.json`: source-run identity, methods, cohort/code/model hashes and reference selection.

Reusing an existing output directory with a different selected-table hash or different model bytes is rejected. Choose a new `--output` to build a new cohort reference; preserve the old reference for longitudinal comparisons. Importable notebook API:

```python
from pathlib import Path
from NM_TinyRNN.code.analyses.monoGRU_alignment import load_model, align_model

alignment_dir = Path('/home/charlesdgburns/Coding/NM_TinyRNN/data/analyses/monoGRU_alignment/nested_DA_S32_B8/monoGRU_relu_ndb')
reference = load_model(alignment_dir / 'reference_model.pickle')
aligned, audit = align_model(model, reference)
```

For the fitted cohort, load each `aligned_model_path` from the table instead of repeating notebook alignment to a hand-picked `model`. All exported models belong to monoGRU +BC-DB; the validation API rejects decoder-bias-on models. The API returns an ambiguity/degeneracy audit; examine it for new models as well. No notebook cells are changed by this script.

This is a descriptive reference estimated from the whole selected cohort. An independent mechanistic generalization claim requires estimating the reference/convention from the appropriate training cohort and freezing it before evaluating new held-out subjects. Do not treat folds as independent animals, or compare these scaling-dependent hidden statistics with earlier unaligned statistics without explicitly declaring the coordinate change. Function-preserving rescaling leaves predictive NLL unchanged but does not preserve participation ratio, activity norm, or parameter regularization penalties.

The earlier mixed-bias run in the parent output directory is superseded for this task; use only the `monoGRU_relu_ndb/` reference and aligned-model table.

## Export examples and average parameters

After alignment, run `.venv/bin/python code/analyses/export_aligned_monoGRU.py --jobs -1`.
This regenerates all 110 full-subject trial rollouts from raw session data,
preserves exact saved fold labels, and writes conventional per-subject/outer-fold
example files under `data/rnns/subject_examples_aligned`. It uses the shared
`mechanistic_variability._run_model_on_trials` implementation. Every exported
trajectory is checked against the source model's predictions and expected
hidden transformation. Equal-subject average parameters are saved as a table
and a descriptive mean-model pickle; the notebook equation-style drawing helper
is reused from `code/figures/monoGRU_parameters.py`. The folder README records
layout, interpretation, and reproducibility details.

Trial regeneration uses joblib process workers across subjects (`--jobs -1`, the default, uses all available CPUs). Each worker runs one CPU thread and reuses its subject data across ten folds. Use `--jobs 1` for serial execution. The mean-parameter reduction and plotting happen after workers finish, in deterministic subject/fold order.
