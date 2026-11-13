"""Paired production timings and quality for ReLU grids with least-squares refinement."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,gc,hashlib
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
import torch
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.mechanistic_variability import CPUUnpickler
from NM_TinyRNN.code.measures.scaling_sensitivity import parameters,fit_pair
BASE=REPO/'data/analysis/grid_refinement_runtime_nested_DA_last10_P100_S32_B8'
OUT=REPO/'data/analysis/least_squares_grid_benchmark_nested_DA_last10_P100_S32_B8'
GRIDS={'original_6':np.unique(np.r_[np.linspace(.5,2.,5),1.]),
       'fine_12':np.unique(np.r_[np.linspace(.5,2.,11),1.]),
       'wide_9':np.geomspace(.25,4.,9),
       'wide_fine_17':np.geomspace(.25,4.,17),
       'very_wide_17':np.geomspace(.1,10.,17)}


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    if affinity:os.sched_setaffinity(0,{affinity[0]})
    selected=pd.read_csv(BASE/'selected_models.tsv',sep='\t');selected.to_csv(OUT/'selected_models.tsv',sep='\t',index=False)
    previous=pd.read_csv(BASE/'timings.tsv',sep='\t');previous=previous[previous.repeat.eq(0)&previous.method.eq('grid_only')]
    configs=[(g,m) for g in GRIDS for m in ['grid_only','trf']]
    timing=[];quality=[];diagnostics=[]
    for gi,(key,rows) in enumerate(selected.groupby('model_id')):
        rows=rows.sort_values(['subject_id','outer_loop_n']).reset_index(drop=True)
        sample=previous[previous.model_id.eq(key)].sort_values('pair_number')
        assert len(sample)==60
        models={}
        for index in sorted(set(sample.index_A)|set(sample.index_B)):
            path=Path(rows.iloc[index].model_pickle_path)
            if not path.is_absolute():path=REPO.parent/path
            with path.open('rb') as f:models[index]=CPUUnpickler(f).load().cpu().eval()
        first=sample.iloc[0]
        for grid,method in configs:
            mv.align_rnns(models[first.index_A],models[first.index_B],scale_values=GRIDS[grid],refine_scaling=method!='grid_only')
        for repeat in range(2):
            order=configs[gi%len(configs):]+configs[:gi%len(configs)]
            if repeat:order=order[::-1]
            for grid,method in order:
                gc.collect()
                for row in sample.itertuples():
                    wall=time.perf_counter_ns();cpu=time.process_time_ns()
                    aligned,info=mv.align_rnns(models[row.index_A],models[row.index_B],scale_values=GRIDS[grid],
                        refine_scaling=method!='grid_only')
                    cpu_ms=(time.process_time_ns()-cpu)/1e6;wall_ms=(time.perf_counter_ns()-wall)/1e6
                    rec=dict(model_id=key,hidden_size=int(rows.hidden_size.iloc[0]),comparison=row.comparison,pair_number=row.pair_number,
                        index_A=row.index_A,index_B=row.index_B,grid=grid,method=method,repeat=repeat,wall_ms=wall_ms,cpu_ms=cpu_ms)
                    timing.append(rec)
                    if repeat==0:
                        quality.append({k:v for k,v in rec.items() if k not in ['wall_ms','cpu_ms','repeat']}|
                            dict(distance=info['aligned_distance'],grid_distance=info['grid_distance'],cosine=mv.compute_weight_similarity(models[row.index_A],aligned),
                                success=info['refinement_success'],nfev=info['refinement_nfev'],
                                permutation=json.dumps(info['perm'].tolist()),scales=json.dumps(info['transform_vec'].tolist())))
        # Additional quality reference: refine BOTH permutations, each from its own best original-grid point.
        for row in sample.itertuples():
            ref=fit_pair(parameters(models[row.index_A]),parameters(models[row.index_B]),'relu')
            diagnostics.append(dict(model_id=key,hidden_size=int(rows.hidden_size.iloc[0]),pair_number=row.pair_number,
                distance=ref['continuous_distance'],cosine=ref['continuous_similarity'],permutation=ref['continuous_perm'],
                failed_branches=ref['optimizer_failed_branches']))
        pd.DataFrame(timing).to_csv(OUT/'timings.tsv',sep='\t',index=False)
        pd.DataFrame(quality).to_csv(OUT/'quality.tsv',sep='\t',index=False)
        print(f'Finished {key}',flush=True)
    timing=pd.DataFrame(timing);quality=pd.DataFrame(quality);diagnostics=pd.DataFrame(diagnostics)
    timing.to_csv(OUT/'timings.tsv',sep='\t',index=False);quality.to_csv(OUT/'quality.tsv',sep='\t',index=False);diagnostics.to_csv(OUT/'both_permutations_reference.tsv',sep='\t',index=False)
    manifest=dict(grids={k:v.tolist() for k,v in GRIDS.items()},sample='Same 720 fit pairs as prior benchmark: 60 per variant, 30 within/30 across, twelve ReLU variants (one/two units; three architectures; two decoder bias conditions).',
        repeats=2,configs=len(configs),timed_calls=len(timing),benchmark='Actual production align_rnns; include deepcopy, parameter extraction, grid search, refinement and applying transformations; exclude loading, warm-up and quality reports. One CPU core and single-thread BLAS/PyTorch. Config order rotated/reversed.',
        solvers={'trf':'Analytic-Jacobian least squares; log bounds [-20,20], ftol/xtol/gtol 1e-10, max_nfev 300.'},
        refinement='Only winning grid permutation refined. Additional diagnostic refines both permutations from each branch best original-grid candidate; not a global-optimality proof.',
        cpu_affinity=affinity[0] if affinity else None,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(mv.__file__),BASE/'selected_models.tsv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('All least-squares grid benchmarks complete',flush=True)


if __name__=='__main__':main()
