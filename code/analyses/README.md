# Scientific analyses

Analyses own experiment planning, fitting orchestration, result selection, and
study-specific reports. Models/trainers stay in `code/models`; reusable metrics
stay in `code/measures`; AWS/SLURM code handles execution and transport.
Publication placement (main text or appendix) does not determine source layout.

## Repository layout

- `code/analyses/`: named training and post-training workflows and their documentation.
- `code/measures/`: reusable metrics and statistical computations.
- `code/figures/`: reusable plotting functions.
- `data/analysis/`: one folder per analysis for intermediate dataframes, caches and provenance; see [layout](../../data/analysis/README.md). Dataset identifiers live in configuration/metadata.
- `results/`: final figures and tables, including `main/` and `appendix/` outputs.
- `data/rnns/`: fitted model artifacts consumed by the analyses.
- `code/analyses/preliminary/`: gitignored exploratory checks and their local outputs.

The package was renamed from `code/studies` to `code/analyses`. Update imports
and CLI paths accordingly. Existing serialized `study`/`workflow` configuration
fields, `--study` options and `get_study()` registry remain unchanged so saved
plans keep their scientific meaning. Previously prepared AWS archives are
self-contained and keep their original source layout; new bundles use
`code/analyses`. Historical manifests and audit notes retain their original
paths and hashes. Local source-bound plans must be recreated after a source
move; their identity checks intentionally reject changed source locations.

| Name | Definition | Split policy |
| --- | --- | --- |
| `baseline` | Current `models.experiments.get_DA_info_df()` | Per-subject, whole-session nested CV from `nested_cv.py` |
| `pooled_ws` | The existing eight two-unit pooled WS model variants | Blocks partitioned within subject before pooling |
| `data_scaling` | Two-unit GRU/tanh/bias-on and monoGRU/ReLU/bias-off | Fixed latest-session holdout; recency-based exact training budgets |

Baseline reads the live experiment definition, including its subject/model
selection. It does not reproduce a separate hard-coded list in AWS code.
`--models` can restrict baseline model IDs; `--model-types` is passed to the
existing experiment builder. Pooled model keys retain their existing names,
such as `GRU_tanh_biasoff`. The two studies' split policies are not interchangeable.

## Post-training mechanistic variability

Run `.venv/bin/python code/analyses/mechanistic_variability.py` for the maintained
computation → statistics → plots workflow. Use `--stage statistics` or
`--stage plots` to rerun later stages, and `--recompute` to force fresh scores.
See [MECHANISTIC_VARIABILITY.md](MECHANISTIC_VARIABILITY.md) for methods, caching,
notebook compatibility and output tables. This standalone analysis consumes
existing fitted models; it is not an AWS training task.

## Weight-similarity scaling sensitivity

Run `.venv/bin/python code/analyses/weight_similarity_sensitivity.py` for the
pairwise discrete/continuous scaling comparisons, percentage-change table,
and empirical within/across-subject figures. The combined figure is exported
with both the earlier discrete-ReLU contrast and the adopted continuously
refined ReLU contrast, clearly labelled. No balancing or bootstrap is performed.
Use `--stage report` or `--stage plots` to reuse verified computations.
See [WEIGHT_SIMILARITY_SENSITIVITY.md](WEIGHT_SIMILARITY_SENSITIVITY.md) for inputs,
selection, caching, methods and output locations. This is a standalone
post-training workflow, not an AWS training task.

## Interface and tables

Each implemented study exposes ordinary functions:

```python
plan = study.build_plan(config, data_root)
context = study.init_context(plan['config'], payload, data_root, smoke=False)
study.run_task(task, plan['config'], task_output, context)
results = study.collect_results(plan['config'], payload, output, context)
study.analyse_results(results, output / 'reduced')
```

`common.plan_payload(plan)` creates the JSON-serializable payload. Shared
`execution.py` supplies the task metadata, identity checks, and complete-run
collection; callers normally use those helpers instead of calling the low-level
functions directly.

- `experiment_df`: one configuration/subject-or-group/outer-fold row, retaining
  model settings and logical data/result paths.
- `task_df`: one inner fit per row. Each fit uses the existing vectorized
  hyperparameter/weight-seed search. `index` is the stable global task ID and
  `experiment_id` links it back to the backbone table.
- `analysis_df`: one completed inner model per row, with paths to its native
  model, info JSON, and trial-table artifacts.
- Fold/summary results: measured held-out NLL and selected inner-fold metadata.

Task indices are zero-based. `outer` is zero-based in task tables, while the
existing experiment `outer_loop_n` remains one-based. `inner` keeps its study's
native coordinate: baseline validation position 0..8, pooled held-out chunk
0..9 excluding outer. Exported `inner_loop_idx` retains the existing compact
0..8 directory convention.

Plans freeze the concrete experiment/task rows and fold membership. Workers do
not reconstruct a different preset from whatever happens to exist on a server.
Data/code/config identities prevent mixing results from different plans.

## Local execution

The shared local CLI uses no server tools. For example, create a JSON file:

```json
{
  "models": ["2_unit_GRU_tanh_unipolar_forced"],
  "sequence_length": 32,
  "trainer": {"batch_size": 8}
}
```

From the repository root:

```bash
.venv/bin/python code/analyses/run.py plan --study baseline \
  --config /path/to/config.json --data-root /path/to/AB_behaviour \
  --output /tmp/baseline-check

# Execute one planned task, or a reduced-grid smoke task:
.venv/bin/python code/analyses/run.py task --index 0 --output /tmp/baseline-check
.venv/bin/python code/analyses/run.py task --index 0 --smoke --output /tmp/baseline-check

# After every full task is present:
.venv/bin/python code/analyses/run.py collect --output /tmp/baseline-check
```

`plan` writes configuration, portable plan tables, and local input/source hashes.
`task` writes `task-NNNN/`; smoke outputs go under a separate `smoke/` directory.
`collect` requires every full task and verifies identity/coordinates/versions
before selecting winners and writing reports. It does not refit a final model
or average inner-fold scores. Local plans bind their input/source locations;
AWS bundles independently package those files for portable execution.

Sequence length defaults to `datasets.SEQUENCE_LENGTH`, batch size to
`training.BATCH_SIZE`. Other defaults remain study-specific: baseline uses the
current Trainer defaults (including weight seeds 0..10); pooled keeps its prior
weight seeds 1..10 and training grid. For baseline sparsity-labelled runs,
energy penalties remain zero as in `nested_cv.train_outers`; energy-labelled
runs retain the original combined grids.

Trainer defaults to patience 100. Its global training RNGs and block shuffling
use train_seed; baseline nested CV derives this per inner fold from the base
study seed. Session chunks themselves remain deterministic. An explicit
trainer.shuffle_seed overrides only block order; None (default) follows the
training seed. Weight-initialization seeds remain a separate candidate grid.
In pooled CV the study seed controls block partitioning and supplies the base
training seed. Each fit derives its training seed using the compact inner-fold
position (0..8), matching the baseline/scaling convention. Neither seed changes
block boundaries. A pooled session limit is applied at both planning and runtime.

## AWS execution

```bash
.venv/bin/python code/servers/AWS/prepare.py --study baseline \
  --name baseline-s32-b8 --sequence-length 32 --batch-size 8

.venv/bin/python code/servers/AWS/prepare.py --study pooled_ws \
  --name pooled-s32-b8 --sequence-length 32 --batch-size 8
```

Use the existing build/smoke/full/reduce/download commands for either bundle;
see [AWS instructions](../servers/AWS/README.md). Packaging whitelists each
study's declared source dependencies and selected input trials. Server-specific
resource limits do not appear in the scientific experiment table.

Large task plans are split into AWS arrays of at most 10,000 tasks, with offsets
mapping each array child back to the global `task_df.index`. A singleton tail
uses an ordinary Batch job. Every part must succeed before reduction; retries
preserve successful parts and resubmit failed/missing parts explicitly.

## Using saved models in analyses

Result archives retain native exports within each task directory. The reduced
`experiment_df.json` includes completion flags, task IDs, and an artifact-index
reference; `analysis_df.json` contains the concrete per-task artifact paths.
These paths are relative to the run output root, not a container's `/app` path.

After downloading all results and unpacking task archives at the output root:

```python
from NM_TinyRNN.code.analyses.execution import read_analysis_df
from NM_TinyRNN.code.measures.performance import get_performance_df

analysis_df = read_analysis_df(output_root)
performance_df = get_performance_df(analysis_df)
```

Do not rediscover distributed task directories with the legacy native-directory
`get_analysis_df(experiment_df)` call; the emitted artifact index already does
that expansion. Ordinary local/SLURM baseline outputs continue to support the
existing discovery functions unchanged. Additional similarity analyses can use
these same saved-model paths and the existing reusable measures.

## Compatibility and deferred work

`models/pooled_ws_cv.py` remains an import/CLI alias for `analyses/pooled_ws.py`.
The pre-existing `session_learning_curve.py` prototype now lives in analyses too,
with its original models import/CLI alias and shell entry point preserved. It
has not been adopted as the new data-scaling design: its block-level choices are
not silently applied to that future study.

The reserved data-scaling study records the requested 2-unit GRU/tanh/decoder
bias on and 2-unit monoGRU/ReLU/decoder bias off models, but deliberately refuses
to build tasks until data budgets, evaluation sets and similarity comparisons
are agreed. No AWS bundle can be prepared for it yet.

Baseline session selection defaults to the latest 10 sessions per subject, using
timestamp folder order before blocking or constructing folds. The plan records
`selected_sessions`; `sessions_per_subject` can explicitly request a larger pool.

See [DATA_SCALING.md](DATA_SCALING.md) for the standalone scaling runner, local
WS20 smoke test, fixed holdouts and post-training similarity analysis.
