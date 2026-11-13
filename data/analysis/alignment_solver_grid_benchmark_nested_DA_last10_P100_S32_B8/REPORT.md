> Subsequent decision: retain TRF as the only alignment refinement solver. The L-BFGS-B option has been removed; the results below are historical benchmark measurements.

# Grid resolution/range and TRF versus L-BFGS-B

## Main findings

L-BFGS-B gives a speed improvement, not a materially different optimum, in this benchmark. The scalar objective is normalized as described below; the unnormalized version stopped prematurely on a synthetic small-weight test and was not used in these results.
With the original grid, mean runtime is 1.24 versus 2.27 ms per pair for one-unit models, and 3.14 versus 4.60 ms for two-unit models (L-BFGS-B versus TRF). These include the grid search and model transformation, not just refinement.
All 720 original-grid L-BFGS-B refinements converged and matched the TRF distances within numerical precision. Across all five grids (3,600 L-BFGS-B fits), one wider-grid one-unit case did not converge; the production routine retained its grid solution. A TRF fallback would be advisable before making L-BFGS-B the default.
None of the finer/wider grids materially improved refined distances relative to original-grid TRF at the stated threshold, and none changed the selected permutation. The original-grid refinement also matched the best observed result including the independent both-permutation diagnostic.
Recommendation: retain the original six-value grid for initialization. TRF remains the conservative default; normalized L-BFGS-B is a reasonable speed option if paired with a fallback and convergence checks. Larger grids add cost without a detected refinement-quality benefit in this sample.

Same 720 ReLU fit pairs across one/two-unit GRU, lightGRU and monoGRU models, with/without decoder bias; two timed repeats per condition. This is a controlled benchmark, not the complete analysis. One CPU core, single-thread BLAS/PyTorch; actual production alignment including model deepcopy, exhaustive grid search, transformations and final distance. Loading/warmup and score reporting excluded. The order of conditions was rotated/reversed.

## Grids

| Grid | Range | Values/unit | Candidates for two units | Spacing |
|---|---|---:|---:|---|
| original_6 | 0.5–2 | 6 | 72 | linear + unit scale |
| fine_12 | 0.5–2 | 12 | 288 | linear + unit scale |
| wide_9 | 0.25–4 | 9 | 162 | logarithmic |
| wide_fine_17 | 0.25–4 | 17 | 578 | logarithmic |
| very_wide_17 | 0.1–10 | 17 | 578 | logarithmic |

The alternative grids are not all nested supersets of the original values. Discrete-only scores need not improve monotonically; the main comparison is their final continuously refined solutions.

## Two-unit results

Quality is relative to the original six-value grid plus TRF, on 360 two-unit pairs. A meaningful distance difference exceeds 10⁻⁵ × max(reference distance, 1). “Miss best” compares against the best solution observed across all tested grids/solvers and an additional search refining both original-grid permutation branches; this is not a global-optimality certificate. Cosine need not improve when Euclidean distance improves.

| Grid | Solver | Mean ms/pair | Better / worse pairs | Permutation changes | Miss best | Mean cosine change | Failed |
|---|---|---:|---|---:|---:|---:|---:|
| original_6 | grid_only | 2.704 | 0 / 360 | 0 | 360 | -0.00464 | 0 |
| original_6 | trf | 4.597 | 0 / 0 | 0 | 0 | +0.00000 | 0 |
| original_6 | L-BFGS-B | 3.141 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| fine_12 | grid_only | 8.626 | 0 / 359 | 0 | 359 | -0.00258 | 0 |
| fine_12 | trf | 10.433 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| fine_12 | L-BFGS-B | 9.055 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| wide_9 | grid_only | 5.113 | 0 / 360 | 0 | 360 | -0.00097 | 0 |
| wide_9 | trf | 7.127 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| wide_9 | L-BFGS-B | 5.652 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| wide_fine_17 | grid_only | 16.574 | 0 / 360 | 0 | 360 | -0.00023 | 0 |
| wide_fine_17 | trf | 18.628 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| wide_fine_17 | L-BFGS-B | 17.113 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| very_wide_17 | grid_only | 16.485 | 0 / 360 | 0 | 360 | -0.00144 | 0 |
| very_wide_17 | trf | 18.587 | 0 / 0 | 0 | 0 | -0.00000 | 0 |
| very_wide_17 | L-BFGS-B | 17.155 | 0 / 0 | 0 | 0 | -0.00000 | 0 |

## Solver settings and interpretation

Both solvers start at the SAME winning grid point, with its permutation fixed. TRF uses analytic-Jacobian nonlinear least squares (ftol/xtol/gtol 10⁻¹⁰; max 300 residual evaluations). L-BFGS-B uses the analytical gradient of half the squared residual norm, divided by ||target||²+||unit-scale-permuted source||². This positive scalar normalization preserves the optimum and prevents premature absolute stopping on very small weight vectors. L-BFGS-B uses ftol 10⁻¹², gtol 10⁻¹⁰, maxiter/maxfun 300 and maxls 50. Both optimize log scales bounded to [-20,20]. Tolerance definitions differ, so final achieved distances—not tolerance labels—are compared.
If a refinement fails or does not improve the objective, the production function retains the grid candidate. Failure counts therefore matter in evaluating the solver. Successful convergence does not guarantee a global optimum. Neither solver changes the Euclidean alignment objective or introduces minimum-norm balancing.
A wider/finer grid can change the selected permutation and local starting basin, in addition to improving the discrete approximation. The extra quality diagnostic refines each permutation from its own best original-grid point; it is not included in the production timings. Its results are in `both_permutations_reference.tsv`.
The default remains TRF and the original grid pending evaluation; options are exposed as `scale_values=...` and `refinement_method="L-BFGS-B"`. Existing figures/caches are not replaced by this benchmark.

## Files and reproduction

- `summary.tsv`, `runtime_summary.tsv`, `quality_summary.tsv`: both hidden sizes, all grids and methods.
- `paired_solver_comparisons.tsv`: same-grid paired solver differences.
- `quality_comparisons.tsv`: pairwise quality relative to baseline and best observed solution.
- `timings.tsv`, `quality.tsv`: measured calls and alignment outcomes.
- `manifest.json`: exact grids, settings and source hashes.

```bash
.venv/bin/python code/measures/benchmark_alignment_solvers_grids.py
.venv/bin/python code/measures/report_alignment_solver_grids.py
```
