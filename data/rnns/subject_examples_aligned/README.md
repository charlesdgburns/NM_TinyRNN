# Aligned monoGRU +BC-DB subject examples

110 two-unit monoGRU ReLU models **without decoder bias**, selected from `nested_DA_S32_B8` by block-reset train+validation NLL (11 subjects × ten outer folds). No outer-test loss was used for selection.

Each model is initially minimum-norm balanced under positive ReLU scaling and labeled so h1 favors choice 1 and h2 favors choice 0. After medoid selection, its positive hidden-unit scales are numerically optimized to minimize core-parameter distance to the fixed balanced medoid, with unit roles preserved. These reference-fitted models are averaged and exported (protocol `balanced_medoid_reference_fit_v2`). The population medoid is WS14, outer fold 3, inner fold 4. These are function-preserving transformations of fitted models, not newly trained networks.

The layout matches `subject_examples_latest`:

```text
<subject>/monoGRU/outer_fold_<N>/
  2_unit_monoGRU_relu_unipolar_forced_ndb_info.json
  2_unit_monoGRU_relu_unipolar_forced_ndb_model.pickle
  2_unit_monoGRU_relu_unipolar_forced_ndb_trials_data.htsv
outer_fold_winners.htsv
export_manifest.json
```

The trial tables were **regenerated from raw subject session files**, through `mechanistic_variability._run_model_on_trials`, using each aligned model. They contain hidden states, scalar update gate, logits, probabilities, original trial identifiers and the original train/validation/evaluation labels. Inference runs continuously through the whole subject history as in the source examples; it does not reset at block, session or split boundaries. Raw trial identities and split labels must match the original export before regeneration. Predicted probabilities/logits/gates must agree with the original model, and hidden states must match its documented scale/permutation. These comparisons and float32 serialization checks ran for every exported model. `regeneration_checks.htsv` records the actual errors. These continuous trial probabilities are distinct from the original block-reset NLL values retained in the info files.

Root-level summary files:

- `balanced_population_medoid_model.pickle`: the frozen balanced reference.
- `average_aligned_model.pickle`: descriptive parameter-mean model, **not** an ensemble or an independently trained fit.
- `average_parameters.htsv` / `.md`: all 25 parameter entries in stored orientation, with mean and between-subject SD/SEM. Mean each subject's ten folds, then mean the 11 subjects equally. `fit_sd` is descriptive only because folds share data. Initial-state parameters are included; `z_0` is stored but unused by the default scalar-gate recurrence.
- `average_parameters_equations.png` / `.svg` / `.pdf`: the notebook's equation-style figure, with a shared zero-centered parameter color scale. Candidate activation is ReLU, gate is sigmoid, and the decoder uses current h_t. Initial states are reported in the table rather than in these three equations.
- `export_manifest.json`: source session hashes, code provenance, model identity, regenerated-file hashes and prediction-preservation errors.

Reproduce from the repository root:

```bash
.venv/bin/python code/analyses/monoGRU_alignment.py
.venv/bin/python code/analyses/export_aligned_monoGRU.py --jobs -1
```

The alignment script's source models must be locally available under the selected training run. The exporter checks those checkpoint hashes and the frozen alignment manifest. The exported model pickles and trial tables can be loaded directly without rerunning training or export. See `code/analyses/MONOGRU_ALIGNMENT.md` for the alignment rationale and source-run guardrails. Source checkpoints, `subject_examples_latest`, and notebook cells were not overwritten.

Trial regeneration uses joblib process workers across subjects (`--jobs -1`, the default, uses all available CPUs). Each worker runs one CPU thread and reuses its subject data across ten folds. Use `--jobs 1` for serial execution. The mean-parameter reduction and plotting happen after workers finish, in deterministic subject/fold order.
