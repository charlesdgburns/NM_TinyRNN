"""Compare matched runtime/accuracy of ReLU grid designs and local solvers."""
from pathlib import Path
import sys,json
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
OUT=REPO/'data/analysis/alignment_solver_grid_benchmark_nested_DA_last10_P100_S32_B8'


def main():
    timing=pd.read_csv(OUT/'timings.tsv',sep='\t');q=pd.read_csv(OUT/'quality.tsv',sep='\t');reference=pd.read_csv(OUT/'both_permutations_reference.tsv',sep='\t')
    med=timing.groupby(['model_id','hidden_size','comparison','pair_number','grid','method'],as_index=False)[['wall_ms','cpu_ms']].median()
    runtime=med.groupby(['hidden_size','grid','method'],as_index=False).agg(mean_ms=('wall_ms','mean'),median_ms=('wall_ms','median'),p95_ms=('wall_ms',lambda x:x.quantile(.95)),mean_cpu_ms=('cpu_ms','mean'))
    runtime.to_csv(OUT/'runtime_summary.tsv',sep='\t',index=False)
    keys=['model_id','hidden_size','pair_number']
    baseline=q[q.grid.eq('original_6')&q.method.eq('trf')][keys+['distance','cosine','permutation']].rename(columns={'distance':'base_distance','cosine':'base_cosine','permutation':'base_permutation'})
    q=q.merge(baseline,on=keys,validate='many_to_one')
    q['distance_improvement']=q.base_distance-q.distance;q['cosine_change']=q.cosine-q.base_cosine
    q['permutation_changed']=q.permutation.ne(q.base_permutation)
    q['meaningful_improvement']=q.distance_improvement>1e-5*np.maximum(q.base_distance,1.)
    q['meaningful_worsening']=q.distance_improvement<-1e-5*np.maximum(q.base_distance,1.)
    best=pd.concat([q[keys+['distance']],reference[keys+['distance']]]).groupby(keys,as_index=False).distance.min().rename(columns={'distance':'best_observed_distance'})
    q=q.merge(best,on=keys,validate='many_to_one')
    q['misses_best']=q.distance-q.best_observed_distance>1e-5*np.maximum(q.best_observed_distance,1.)
    q.to_csv(OUT/'quality_comparisons.tsv',sep='\t',index=False)
    quality=q.groupby(['hidden_size','grid','method'],as_index=False).agg(pairs=('pair_number','size'),better=('meaningful_improvement','sum'),worse=('meaningful_worsening','sum'),
        different_permutation=('permutation_changed','sum'),misses_best=('misses_best','sum'),mean_distance_improvement=('distance_improvement','mean'),max_distance_improvement=('distance_improvement','max'),
        mean_cosine_change=('cosine_change','mean'),max_abs_cosine_change=('cosine_change',lambda x:x.abs().max()),
        failed=('success',lambda x:(x==False).sum()),mean_nfev=('nfev','mean'))
    quality.to_csv(OUT/'quality_summary.tsv',sep='\t',index=False)
    combined=runtime.merge(quality,on=['hidden_size','grid','method'],validate='one_to_one');combined.to_csv(OUT/'summary.tsv',sep='\t',index=False)
    # Same-grid comparison isolates the solvers from grid/permutation effects.
    solver=q[q.method.eq('trf')].merge(q[q.method.eq('L-BFGS-B')],on=keys+['grid'],suffixes=('_trf','_lbfgs'),validate='one_to_one')
    solver['distance_difference_lbfgs_minus_trf']=solver.distance_lbfgs-solver.distance_trf
    solver['cosine_difference_lbfgs_minus_trf']=solver.cosine_lbfgs-solver.cosine_trf
    solver.to_csv(OUT/'paired_solver_comparisons.tsv',sep='\t',index=False)
    specs=json.loads((OUT/'manifest.json').read_text())
    lines=['# Grid resolution/range and TRF versus L-BFGS-B','',
        'Same 720 ReLU fit pairs across one/two-unit GRU, lightGRU and monoGRU models, with/without decoder bias; two timed repeats per condition. This is a controlled benchmark, not the complete analysis. One CPU core, single-thread BLAS/PyTorch; actual production alignment including model deepcopy, exhaustive grid search, transformations and final distance. Loading/warmup and score reporting excluded. The order of conditions was rotated/reversed.','',
        '## Grids','',
        '| Grid | Range | Values/unit | Candidates for two units | Spacing |','|---|---|---:|---:|---|']
    for name,values in specs['grids'].items():
        n=len(np.unique(np.r_[values,1.]));lines.append(f'| {name} | {min(values):g}–{max(values):g} | {n} | {2*n*n} | '+('linear + unit scale' if name in ['original_6','fine_12'] else 'logarithmic')+' |')
    lines+=['','## Two-unit results','',
        'Quality is relative to the original six-value grid plus TRF, on 360 two-unit pairs. A meaningful distance difference exceeds 10⁻⁵ × max(reference distance, 1). “Miss best” compares against the best solution observed across all tested grids/solvers and an additional search refining both original-grid permutation branches; this is not a global-optimality certificate. Cosine need not improve when Euclidean distance improves.','',
        '| Grid | Solver | Mean ms/pair | Better / worse pairs | Permutation changes | Miss best | Mean cosine change | Failed |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    order=list(specs['grids'])
    for name in order:
        for method in ['grid_only','trf','L-BFGS-B']:
            r=combined[combined.hidden_size.eq(2)&combined.grid.eq(name)&combined.method.eq(method)].iloc[0]
            lines.append(f'| {name} | {method} | {r.mean_ms:.3f} | {int(r.better)} / {int(r.worse)} | {int(r.different_permutation)} | {int(r.misses_best)} | {r.mean_cosine_change:+.5f} | {int(r.failed)} |')
    lines+=['','## Solver settings and interpretation','',
        'Both solvers start at the SAME winning grid point, with its permutation fixed. TRF uses analytic-Jacobian nonlinear least squares (ftol/xtol/gtol 10⁻¹⁰; max 300 residual evaluations). L-BFGS-B uses the analytical gradient of half the squared residual norm, divided by ||target||²+||unit-scale-permuted source||². This positive scalar normalization preserves the optimum and prevents premature absolute stopping on very small weight vectors. L-BFGS-B uses ftol 10⁻¹², gtol 10⁻¹⁰, maxiter/maxfun 300 and maxls 50. Both optimize log scales bounded to [-20,20]. Tolerance definitions differ, so final achieved distances—not tolerance labels—are compared.',
        'If a refinement fails or does not improve the objective, the production function retains the grid candidate. Failure counts therefore matter in evaluating the solver. Successful convergence does not guarantee a global optimum. Neither solver changes the Euclidean alignment objective or introduces minimum-norm balancing.',
        'A wider/finer grid can change the selected permutation and local starting basin, in addition to improving the discrete approximation. The extra quality diagnostic refines each permutation from its own best original-grid point; it is not included in the production timings. Its results are in `both_permutations_reference.tsv`.',
        'TRF and the original grid are retained. TRF is now the only production refinement solver; custom grids remain available through `scale_values`. L-BFGS-B rows are historical benchmark results; the experimental production option has been removed.','',
        '## Files and reproduction','',
        '- `summary.tsv`, `runtime_summary.tsv`, `quality_summary.tsv`: both hidden sizes, all grids and methods.',
        '- `paired_solver_comparisons.tsv`: same-grid paired solver differences.',
        '- `quality_comparisons.tsv`: pairwise quality relative to baseline and best observed solution.',
        '- `timings.tsv`, `quality.tsv`: measured calls and alignment outcomes.',
        '- `manifest.json`: exact grids, settings and source hashes.','',
        '```bash','# Regenerate the historical summary from saved measurements:', '.venv/bin/python code/measures/report_alignment_solver_grids.py', '# Current least-squares-only grid benchmark (separate output directory):', '.venv/bin/python code/measures/benchmark_alignment_solvers_grids.py','```','']
    original=combined[combined.grid.eq('original_6')]
    conclusions=['## Main findings','',
        'L-BFGS-B gives a speed improvement, not a materially different optimum, in this benchmark. The scalar objective is normalized as described below; the unnormalized version stopped prematurely on a synthetic small-weight test and was not used in these results.',
        'With the original grid, mean runtime is 1.24 versus 2.27 ms per pair for one-unit models, and 3.14 versus 4.60 ms for two-unit models (L-BFGS-B versus TRF). These include the grid search and model transformation, not just refinement.',
        'All 720 original-grid L-BFGS-B refinements converged and matched the TRF distances within numerical precision. Across all five grids (3,600 L-BFGS-B fits), one wider-grid one-unit case did not converge; the production routine retained its grid solution. A TRF fallback would be advisable before making L-BFGS-B the default.',
        'None of the finer/wider grids materially improved refined distances relative to original-grid TRF at the stated threshold, and none changed the selected permutation. The original-grid refinement also matched the best observed result including the independent both-permutation diagnostic.',
        'Decision after this benchmark: retain the original six-value grid and TRF as the only production refinement solver. The L-BFGS-B implementation has been removed from alignment; its measured results are retained as historical evidence. Larger grids added cost without a detected refinement-quality benefit in this sample.','']
    lines[2:2]=conclusions
    lines.insert(lines.index('## Two-unit results'), 'The alternative grids are not all nested supersets of the original values. Discrete-only scores need not improve monotonically; the main comparison is their final continuously refined solutions.\n')
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(combined[combined.hidden_size.eq(2)].to_string(index=False))
    print('Solver distance disagreement max:',solver.distance_difference_lbfgs_minus_trf.abs().max())


if __name__=='__main__':main()
