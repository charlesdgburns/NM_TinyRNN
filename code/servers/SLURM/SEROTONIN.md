# Serotonin training

From the repository root on the cluster, in the training Python environment:

```sh
python code/servers/SLURM/serotonin.py --dry-run
python code/servers/SLURM/serotonin.py
```

This submits 44 jobs: one per subject/model for the 21 serotonin subjects added
in commit `ed8734e`, plus WS18. Each job trains all outer folds using the existing nested-CV
defaults, two hidden units, unipolar inputs and forced-choice inputs:

- Standard GRU: tanh, decoder bias, sparsity penalty.
- monoGRU: ReLU, no decoder bias, energy penalty.

Results go to `data/rnns/nested_5HT_S32_B8`. Completed configurations are skipped
only when all outer folds are complete. Optional `--sequence-length`,
`--batch-size`, `--partition` and `--cpu-only` flags are supported. Use
`--subjects WS18` to submit only WS18 without duplicating other active jobs.

The existing SLURM generator supplies resources (10 CPUs, 32 GB, 23:59:00),
log locations and the cluster conda environment
`/nfs/nhome/live/cburns/miniconda3/envs/NM_TinyRNN`.

## Local completion and cohort validation

From the repository root, after downloading the native fits:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python code/analyses/cohort_validation.py --train-missing --training-jobs 6
# Rerun analysis only:
.venv/bin/python code/analyses/cohort_validation.py --jobs 3
```

The single analysis script selects block-reset train/validation-NLL winners and
computes the main GRU/monoGRU mechanistic similarities and eight-test inference.
Outputs are isolated under `data/analyses/cohort_validation/nested_5HT_S32_B8/`
and `results/main/nested_5HT_S32_B8/`. All 22 serotonin subjects, including WS18,
are now represented by 90 inner fits per model and ten selected outer winners.
The minimal main table is `results/main/nested_5HT_S32_B8/table_1.svg`.
Raw fitted models and local execution logs are not included in Git.
