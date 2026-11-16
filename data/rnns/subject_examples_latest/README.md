# Latest subject examples

Source run: nested_DA_S32_B8 (S32, batch 8, patience 20).
11 subjects × two model configurations × ten outer folds = 220 winners.

- Standard two-unit GRU: tanh, decoder bias on.
- Two-unit monoGRU: ReLU, decoder bias off.

Exactly the shared analysis winners, selected by minimum block-reset combined
free-choice training+validation NLL among the nine saved inner-fold fits.
All ten outer winners are retained; no outer-test filtering or ranking.
Trial exports are copied unchanged and contain the original continuous-model
outputs, not newly block-reset trajectories.

outer_fold_winners.htsv preserves source paths and adds portable example paths.
export_manifest.json records source/table hashes and hashes of all 660 artifacts.
Layout: <subject>/<architecture>/outer_fold_N/<model_id>_{info.json,model.pickle,trials_data.htsv}.

Reproduce to a new output directory from the repository root:

```bash
.venv/bin/python code/models/save_examples.py \
  --folder_name nested_DA_S32_B8 \
  --selected_models data/analysis/mechanistic_variability/selected_models.htsv \
  --selection block_reset --all_outer_folds \
  --model_id 2_unit_GRU_tanh_unipolar_forced \
  --model_id 2_unit_monoGRU_relu_unipolar_forced_ndb \
  --output_dir /path/to/new/examples
```
