# Runtime of grid-initialized continuous ReLU alignment

## Measured results

Actual CPU `align_rnns` calls on 720 matched model pairs: 60 per variant (30 within-subject, 30 across-subject), across three architectures, two decoder-bias conditions and one/two hidden units. Four timed passes per method; method order alternated. Repeat timings were reduced to a median per pair and then averaged. A second complete timing pass reproduced the result; the first-pass summary is retained separately.

| Hidden units | Grid only | Grid + refinement | Runtime ratio | Added time / pair |
|---|---:|---:|---:|---:|
| 1 | 0.832 ms | 2.415 ms | 2.90× | 1.582 ms |
| 2 | 2.694 ms | 4.737 ms | 1.76× | 2.043 ms |

## Full comparison-set estimates

Each hidden size has six ReLU variants × 5,995 pairs = 35,970 comparisons. The following are ONE-CORE linear estimates, reweighting the deliberately balanced benchmark sample to the actual 495 within- and 5,500 across-subject pairs per variant. They are not observed full-run wall times and exclude model loading. Tanh alignment does not change.

| Hidden units | Grid only | Grid + refinement |
|---|---:|---:|
| 1 | 0.49 min | 1.63 min |
| 2 | 1.61 min | 3.03 min |

## What is timed

The same production alignment function is timed with `refine_scaling=False` and `refine_scaling=True`. Both include model deepcopy, extraction of parameters, exhaustive permutation/grid search, applying the transformation to main and gate weights, and computing final parameter distance. The refined path additionally runs one bounded positive-scale least-squares fit from the grid winner, using its exact Jacobian. The winning permutation is kept fixed. It does not refine every permutation or guarantee the global continuous optimum.
The alignment objective is unchanged: core Euclidean parameter distance, excluding gates and initial state. Full-parameter cosine is reported separately; reducing the objective need not raise that cosine. This is the original pairwise method, not independent minimum-norm balancing.
Loading saved fits, Python imports, warm-up, cosine reporting and postprocessing are excluded from per-call timings. One allowed CPU core was used, with single-threaded NumPy/BLAS and PyTorch. Hardware/background load affects absolute numbers. Multiple workers can reduce wall time, but no six-worker wall-time projection is claimed.
All 720 refinements converged. Function evaluations: median 8, 95th percentile 25, maximum 33. Final distances did not exceed grid distances beyond floating-point tolerance. Synthetic recovery/function-preservation tests and the existing scaling/pooled similarity tests passed (12 tests total).

## Usage and reproducibility

```python
aligned, info = align_rnns(model_a, model_b, refine_scaling=True)
```

The default remains discrete to preserve existing analyses/caches. This benchmark does not replace previously generated figures. Use a separate cache/output when switching the analysis to refinement.
```bash
.venv/bin/python code/measures/benchmark_grid_refinement.py
.venv/bin/python code/measures/report_grid_refinement_runtime.py
```

`timings.tsv` contains every timed call; `per_variant.tsv` contains variant summaries; `quality.tsv` and `improvements.tsv` contain alignment diagnostics.
