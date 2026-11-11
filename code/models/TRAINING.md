# Sequence length, batch size, and training output folders

`datasets.SEQUENCE_LENGTH` is the default number of next-choice predictions per
sequence (default at 32). Pass `sequence_length` explicitly to vary it without
editing a global. `training.BATCH_SIZE` is the shared training minibatch default
(default at 8 sequences).

Each sequence uses `sequence_length + 1` consecutive source trials because
inputs and targets are shifted by one trial. Blocks never cross session
boundaries; incomplete remainders are discarded. Hidden state resets per block.
Changing sequence length changes block counts and remainder exclusions, but does
not randomize block construction. Nested CV still assigns whole sessions to
folds using its existing deterministic splitting policy.

## Experiment dataframes and job submission

`code/models/experiments.py` builds the shared backbone table, conventionally
named `experiment_df`. One row describes a subject/group, model configuration
and outer fold, with data/result paths, completion status, sequence length and
batch size. The existing `get_*_info_df` function names are retained.

`code/servers/SLURM/submit_jobs.py` consumes that table to write and submit jobs.
SLURM resource settings stay in the submission layer, not in the manifest.

```python
from NM_TinyRNN.code.models import experiments
from NM_TinyRNN.code.servers.SLURM import submit_jobs

experiment_df = experiments.get_DA_info_df(sequence_length=64, batch_size=16)
submit_jobs.run_training(experiment_df)
```

A supplied dataframe is used without rebuilding a preset or imposing the default
test-mode row limit. Its sequence/batch settings come from its own rows.
`overwrite=False` skips completed configurations. Each submitted job trains all
outer folds, so the existing dispatcher uses the manifest's outer-fold-1 rows.
The caller's dataframe is not modified.

The old keyword-based convenience call still works:

```python
submit_jobs.run_training(test=False, sequence_length=64, batch_size=16)
```

Submission modules re-export the original builder names for compatibility, and
`code.models.submit_jobs` still aliases the SLURM module. New analysis code can
import `experiments` without depending on server tooling.

This writes results below:

```text
data/rnns/nested_DA_S64_B16/<subject>/<model_type>/<constraint>/
    outer_fold_N/inner_fold_M/<model_id>_...
```

`S64` means 64 predictions per sequence; `B16` means 16 sequences per minibatch.
The default DA folder is `nested_DA_S32_B8` with the current globals. The name
records configuration; it does not select an architecture set or infer a batch
size. The default DA architectures remain GRU and monoGRU.

All four experiment-manifest factories accept `sequence_length` and `batch_size` and store
both in every job row. Their automatic folder prefixes are:

| Factory | Automatic folder |
| --- | --- |
| `get_DA_info_df` | `nested_DA_S{S}_B{B}` |
| `get_job_info_df` | `nested_DA_S{S}_B{B}` |
| `get_test_info_df` | `nested_DA_S{S}_B{B}` |
| `get_ws_all_info_df` | `test_input_encodings_S{S}_B{B}` |

To inspect a job table without submitting anything:

```python
experiment_df = experiments.get_DA_info_df(sequence_length=64, batch_size=16)
print(experiment_df[['sequence_length', 'batch_size', 'save_path']].head())
```

SLURM commands pass both values explicitly to `nested_cv.train_outers`.
Script/log filenames include subject, model, constraint, S, and B. Fold info JSON
records `sequence_length` and `hparam_ranges.batch_size`.

An explicit `folder_name` overrides the automatic folder **exactly**. This keeps
historical analysis calls such as `get_DA_info_df(folder_name='nested_DA_8')`
working; old folders are not renamed. Those names are not parsed into settings.
The legacy `nested_DA_128` architecture list remains supported for existing
notebooks; new callers can specify `model_types` directly instead of encoding an
architecture selection in a folder name. For historical job tables, the new
settings columns describe supplied/default settings, not verified metadata from
old saved runs.

## Analysis from the same backbone

```python
from NM_TinyRNN.code.measures import analysis, performance

# Rebuild after training to refresh the filesystem-derived completion flags.
experiment_df = experiments.get_DA_info_df(sequence_length=64, batch_size=16)
analysis_df = analysis.get_analysis_df(experiment_df)
performance_df = performance.get_performance_df(analysis_df)
```

`analysis_df` expands the manifest into one row per saved inner-fold model, with
artifact paths. `performance_df` adds measured scores; other measures can use the
same manifest/path tables. Analysis still accepts the legacy `info_df=` keyword.
Experiment definitions, server execution, and derived measurements thus remain
separate responsibilities. No dataframe serialization format is imposed.

## Direct training and nested CV

```python
from NM_TinyRNN.code.models import datasets, nested_cv, rnns, training

dataset = datasets.AB_Dataset(subject_path, sequence_length=64)
model = rnns.TinyRNN(input_size=dataset.inputs.shape[-1])
trainer = training.Trainer(batch_size=16)
state, config, losses = trainer.fit(model, dataset)

# Or construct the dataset and train all outer folds through this entry point:
nested_cv.train_outers(subject_path, save_path, 'GRU',
                      sequence_length=64, batch_size=16)
```

`run_outer_fold(model, dataset, ...)` uses the sequence length of its supplied
dataset; pass batch size through `trainer_kwargs`. `Trainer.fit` likewise uses
the blocks already constructed in its dataset. Neither reconstructs sequences.
An explicitly supplied `save_path` is used unchanged.

Seed semantics differ by entry point: direct `Trainer.fit` uses `train_seed`
for train/validation session assignment and training randomness. Nested CV uses
predefined session folds and derives per-fold training RNG/shuffle seeds.
`seed_split` is the dataset's session-assignment argument; the loader's generator
seed controls block order, not block construction.

## Local runners and probes

`session_learning_curve.py` and `pooled_ws_cv.py` both accept
`--sequence-length` and `--batch-size`, defaulting to the same globals. Their
existing output directories remain unchanged; their run manifests record these
settings and reject incompatible resumes. Use a separate `--output` directory
when comparing sequence lengths. Their block-level CV policies are unchanged.
`run_session_learning_curve.sh` forwards these CLI options unchanged.

The session mechanistic-variability script also accepts `--sequence-length` for
its probe blocks, defaults to `datasets.SEQUENCE_LENGTH`, and records the actual
probe length in its analysis manifest and figure caption.

## Sanity checks

```sh
.venv/bin/python -m unittest discover -s code/tests -p 'test_*.py'
```

The tests use temporary synthetic data. They check block sizes, shifted target
and forced-choice masks, session boundaries, defaults/overrides, job folder names,
SLURM command propagation, local CLI controls, a pooled dry run, and a small
nested-CV training/save run. No HPC jobs are submitted.
