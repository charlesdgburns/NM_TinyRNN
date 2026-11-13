"""Summarize paired alignment timings and extrapolate with actual within/across weights."""
from pathlib import Path
import sys,json,hashlib
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
OUT=REPO/'data/analysis/grid_refinement_runtime_nested_DA_last10_P100_S32_B8'


def main():
    timing=pd.read_csv(OUT/'timings.tsv',sep='\t');summary=pd.read_csv(OUT/'summary.tsv',sep='\t')
    medians=timing.groupby(['hidden_size','model_id','method','comparison','pair_number'],as_index=False)[['wall_ms','cpu_ms']].median()
    for index,row in summary.iterrows():
        strata=medians[medians.hidden_size.eq(row.hidden_size)&medians.method.eq(row.method)].groupby('comparison').wall_ms.mean()
        summary.loc[index,'full_set_weighted_wall_ms']=(495*strata['within_subject']+5500*strata['across_subject'])/5995
    summary['estimated_one_core_minutes_35970_pairs']=summary.full_set_weighted_wall_ms*35970/1000/60
    summary.to_csv(OUT/'summary.tsv',sep='\t',index=False)
    quality=pd.read_csv(OUT/'quality.tsv',sep='\t');improvements=pd.read_csv(OUT/'improvements.tsv',sep='\t')
    assert len(timing)==5760 and quality[quality.method.eq('grid_plus_refinement')].refinement_success.astype(bool).all()
    assert timing.groupby(['model_id','method','pair_number']).size().eq(4).all()
    lines=['# Runtime of grid-initialized continuous ReLU alignment','',
        '## Measured results','',
        'Actual CPU `align_rnns` calls on 720 matched model pairs: 60 per variant (30 within-subject, 30 across-subject), across three architectures, two decoder-bias conditions and one/two hidden units. Four timed passes per method; method order alternated. Repeat timings were reduced to a median per pair and then averaged. A second complete timing pass reproduced the result; the first-pass summary is retained separately.','',
        '| Hidden units | Grid only | Grid + refinement | Runtime ratio | Added time / pair |','|---|---:|---:|---:|---:|']
    for h in [1,2]:
        g=summary[summary.hidden_size.eq(h)].set_index('method');a=g.loc['grid_only','mean_wall_ms'];b=g.loc['grid_plus_refinement','mean_wall_ms']
        lines.append(f'| {h} | {a:.3f} ms | {b:.3f} ms | {b/a:.2f}× | {b-a:.3f} ms |')
    lines+=['','## Full comparison-set estimates','',
        'Each hidden size has six ReLU variants × 5,995 pairs = 35,970 comparisons. The following are ONE-CORE linear estimates, reweighting the deliberately balanced benchmark sample to the actual 495 within- and 5,500 across-subject pairs per variant. They are not observed full-run wall times and exclude model loading. Tanh alignment does not change.','',
        '| Hidden units | Grid only | Grid + refinement |','|---|---:|---:|']
    for h in [1,2]:
        g=summary[summary.hidden_size.eq(h)].set_index('method')
        lines.append(f'| {h} | {g.loc["grid_only","estimated_one_core_minutes_35970_pairs"]:.2f} min | {g.loc["grid_plus_refinement","estimated_one_core_minutes_35970_pairs"]:.2f} min |')
    nfev=quality[quality.method.eq('grid_plus_refinement')].nfev
    lines+=['','## What is timed','',
        'The same production alignment function is timed with `refine_scaling=False` and `refine_scaling=True`. Both include model deepcopy, extraction of parameters, exhaustive permutation/grid search, applying the transformation to main and gate weights, and computing final parameter distance. The refined path additionally runs one bounded positive-scale least-squares fit from the grid winner, using its exact Jacobian. The winning permutation is kept fixed. It does not refine every permutation or guarantee the global continuous optimum.',
        'The alignment objective is unchanged: core Euclidean parameter distance, excluding gates and initial state. Full-parameter cosine is reported separately; reducing the objective need not raise that cosine. This is the original pairwise method, not independent minimum-norm balancing.',
        'Loading saved fits, Python imports, warm-up, cosine reporting and postprocessing are excluded from per-call timings. One allowed CPU core was used, with single-threaded NumPy/BLAS and PyTorch. Hardware/background load affects absolute numbers. Multiple workers can reduce wall time, but no six-worker wall-time projection is claimed.',
        f'All 720 refinements converged. Function evaluations: median {nfev.median():.0f}, 95th percentile {nfev.quantile(.95):.0f}, maximum {nfev.max():.0f}. Final distances did not exceed grid distances beyond floating-point tolerance. Synthetic recovery/function-preservation tests and the existing scaling/pooled similarity tests passed (12 tests total).','',
        '## Usage and reproducibility','',
        '```python','aligned, info = align_rnns(model_a, model_b, refine_scaling=True)','```','',
        'The default remains discrete to preserve existing analyses/caches. This benchmark does not replace previously generated figures. Use a separate cache/output when switching the analysis to refinement.',
        '```bash','.venv/bin/python code/measures/benchmark_grid_refinement.py','.venv/bin/python code/measures/report_grid_refinement_runtime.py','```','',
        '`timings.tsv` contains every timed call; `per_variant.tsv` contains variant summaries; `quality.tsv` and `improvements.tsv` contain alignment diagnostics.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['extrapolation']='35970 pairs per hidden size; reweight sampled within/across timings to 495/5500 pairs per variant; one-core linear estimate, not observed full-run or six-worker time.'
    manifest['sha256'][str(REPO/'code/measures/benchmark_grid_refinement.py')]=hashlib.sha256((REPO/'code/measures/benchmark_grid_refinement.py').read_bytes()).hexdigest()
    manifest['sha256'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\n'.join(lines[:19]))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
