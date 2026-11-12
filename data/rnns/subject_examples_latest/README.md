# Current DA baseline: all ten outer-fold winners

Source: nested_DA_last10_P100_S32_B8 (latest ten sessions, patience 100, S32/B8).
Exactly `best_model_df = performance.select_best_outer(performance_df)` selects
one inner fit per subject/model/outer fold by lowest combined train_val_CE.
All ten outer winners are exported; no median/min representative filter and no
exclusion of outer fold 10 are applied.

220 saved models = 11 subjects × 2 model configurations × 10 outer folds.
Each model has info JSON, model pickle and saved trial-data HTSV (660 files).
Only two-unit standard GRU (tanh, decoder bias on) and two-unit monoGRU
(ReLU, decoder bias off) are included. Layout:

```
<subject>/<architecture>/outer_fold_<1..10>/<model_id>_info.json
<subject>/<architecture>/outer_fold_<1..10>/<model_id>_model.pickle
<subject>/<architecture>/outer_fold_<1..10>/<model_id>_trials_data.htsv
```

`outer_fold_winners.htsv` retains source paths, training/performance metadata,
selected inner index, and portable `example_info_path`,
`example_model_pickle_path`, `example_trials_data_path` columns for these copies.
The original inner index is also recorded in info JSON. Files are unmodified,
with copied/source SHA256 verification recorded in export_manifest.json.
Trial data retain the existing continuous-subject export protocol; this is not
a regenerated block-reset prediction table. No models were retrained.

```python
from pathlib import Path
import pandas as pd
root = Path('NM_TinyRNN/data/rnns/subject_examples_latest')
examples = pd.read_csv(root / 'outer_fold_winners.htsv', sep='\t')
rows = examples.query('subject_id == "WS20" and model_id == "2_unit_monoGRU_relu_unipolar_forced_ndb"')
# rows contains all ten outer-fold winners
trials = pd.read_csv(rows.iloc[0].example_trials_data_path, sep='\t')
```

Reproduce from repository root:

```bash
.venv/bin/python code/models/save_examples.py \
  --folder_name nested_DA_last10_P100_S32_B8 \
  --all_outer_folds \
  --model_id 2_unit_GRU_tanh_unipolar_forced \
  --model_id 2_unit_monoGRU_relu_unipolar_forced_ndb \
  --output_dir data/rnns/subject_examples_latest
```

Paths in the dataframe and manifest are relative to the parent of the
NM_TinyRNN repository, matching the notebooks. Run the Python example from
that parent directory. This export includes only the two requested configurations.
