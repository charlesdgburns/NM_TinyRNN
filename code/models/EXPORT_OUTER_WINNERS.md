# Export all outer-fold winners

From the repository root, in the training/analysis Python environment:

```bash
python code/models/export_outer_winners.py --folder_name nested_DA_S32_B8
```

Defaults select two configurations: `2_unit_GRU_tanh_unipolar_forced` and
`2_unit_monoGRU_relu_unipolar_forced_ndb`. Override them with repeated
`--model_id` arguments. Use `--folder_name nested_DA_last10_P100_S32_B8`
to reproduce the earlier patience-100 export instead of the latest default run.

Selection calls `performance.select_best_outer`: lowest combined training and
validation CE among inner fits, separately for every subject/configuration/outer
fold. All ten outer folds are retained, including fold 10; outer-test performance
is not used for selection. Every requested outer fold must have complete saved
training artifacts and a finite selection score.

Outputs default to `data/rnns/subject_examples_latest` in this repository:

```text
<subject>/<architecture>/outer_fold_<N>/<model_id>_info.json
<subject>/<architecture>/outer_fold_<N>/<model_id>_model.pickle
<subject>/<architecture>/outer_fold_<N>/<model_id>_trials_data.htsv
outer_fold_winners.htsv
export_manifest.json
```

The table preserves source paths and inner-fold indices, plus absolute paths to
the exported files. The manifest records source/destination SHA256 checksums.
Existing files at matching export paths are replaced; unrelated files remain.
Choose `--output_dir data/rnns/subject_examples_S32_B8` to keep exports separate.
Relative output paths are resolved from the calling directory. Source paths use
the parent-of-repository convention internally, so the script can be launched
from any working directory. No training or SLURM jobs are launched.
