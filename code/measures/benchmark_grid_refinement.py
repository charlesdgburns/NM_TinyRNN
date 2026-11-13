"""Repeated paired single-core timings: original grid vs grid-winner refinement."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,itertools,json,time,gc,platform,hashlib
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
import torch
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.mechanistic_variability import CPUUnpickler
OUT=REPO/'data/analysis/grid_refinement_runtime_nested_DA_last10_P100_S32_B8'


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    # Keep the timing process on one allowed core to limit migration noise.
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    if affinity:os.sched_setaffinity(0,{affinity[0]})
    frame=pd.read_csv(REPO/'data/analysis/performance_df_nested_DA_last10_P100_S32_B8_final.htsv',sep='\t')
    frame=frame[frame.nonlinearity.eq('relu')&frame.hidden_size.isin([1,2])]
    keys=['model_id','subject_id','outer_loop_n']
    selected=frame.sort_values(keys+['train_val_CE','inner_loop_idx']).groupby(keys,sort=False).head(1)
    selected.to_csv(OUT/'selected_models.tsv',sep='\t',index=False)
    rng=np.random.default_rng(20261116)
    within=[(i,j) for i,j in itertools.combinations(range(110),2) if i//10==j//10]
    across=[(i,j) for i,j in itertools.combinations(range(110),2) if i//10!=j//10]
    sample=[within[i] for i in rng.choice(len(within),30,replace=False)]+[across[i] for i in rng.choice(len(across),30,replace=False)]
    timing=[];quality=[];batch_times=[]
    for key,rows in selected.groupby('model_id'):
        rows=rows.sort_values(['subject_id','outer_loop_n']).reset_index(drop=True)
        assert len(rows)==110
        indices=sorted({i for pair in sample for i in pair});models={}
        for index in indices:
            path=Path(rows.iloc[index].model_pickle_path)
            if not path.is_absolute():path=REPO.parent/path
            with path.open('rb') as f:models[index]=CPUUnpickler(f).load().cpu().eval()
        # Warm both paths before starting measurements.
        for refined in [False,True]:mv.align_rnns(models[sample[0][0]],models[sample[0][1]],refine_scaling=refined)
        for repeat in range(4):
            for refined in ([False,True] if repeat%2==0 else [True,False]):
                gc.collect();batch_start=time.perf_counter()
                for pair_number,(ia,ib) in enumerate(sample):
                    wall=time.perf_counter_ns();cpu=time.process_time_ns()
                    aligned,info=mv.align_rnns(models[ia],models[ib],refine_scaling=refined)
                    cpu_ms=(time.process_time_ns()-cpu)/1e6;wall_ms=(time.perf_counter_ns()-wall)/1e6
                    timing.append(dict(model_id=key,hidden_size=int(rows.hidden_size.iloc[0]),architecture=rows.model_type.iloc[0],decoder_bias=bool(rows.decoder_bias.iloc[0]),
                        repeat=repeat,method='grid_plus_refinement' if refined else 'grid_only',pair_number=pair_number,index_A=ia,index_B=ib,
                        comparison='within_subject' if ia//10==ib//10 else 'across_subject',wall_ms=wall_ms,cpu_ms=cpu_ms,nfev=info['refinement_nfev']))
                    if repeat==0:
                        quality.append(dict(model_id=key,hidden_size=int(rows.hidden_size.iloc[0]),method='grid_plus_refinement' if refined else 'grid_only',
                            pair_number=pair_number,distance=info['aligned_distance'],cosine=mv.compute_weight_similarity(models[ia],aligned),
                            refinement_success=info['refinement_success'],nfev=info['refinement_nfev'],
                            scales=json.dumps(info['transform_vec'].tolist()),permutation=json.dumps(info['perm'].tolist())))
                batch_times.append(dict(model_id=key,hidden_size=int(rows.hidden_size.iloc[0]),repeat=repeat,method='grid_plus_refinement' if refined else 'grid_only',seconds=time.perf_counter()-batch_start))
        pd.DataFrame(timing).to_csv(OUT/'timings.tsv',sep='\t',index=False)
        print(f'Benchmarked {key}',flush=True)
    timing=pd.DataFrame(timing);quality=pd.DataFrame(quality);batch_times=pd.DataFrame(batch_times)
    timing.to_csv(OUT/'timings.tsv',sep='\t',index=False);quality.to_csv(OUT/'quality.tsv',sep='\t',index=False);batch_times.to_csv(OUT/'batch_times.tsv',sep='\t',index=False)
    # First average away repeated observations of the same sampled pair.
    medians=timing.groupby(['model_id','hidden_size','method','comparison','pair_number'],as_index=False)[['wall_ms','cpu_ms']].median()
    per_variant=medians.groupby(['model_id','hidden_size','method'],as_index=False).agg(mean_wall_ms=('wall_ms','mean'),median_wall_ms=('wall_ms','median'),p95_wall_ms=('wall_ms',lambda x:x.quantile(.95)),mean_cpu_ms=('cpu_ms','mean'))
    per_variant.to_csv(OUT/'per_variant.tsv',sep='\t',index=False)
    summary=medians.groupby(['hidden_size','method'],as_index=False).agg(pairs=('pair_number','size'),mean_wall_ms=('wall_ms','mean'),median_wall_ms=('wall_ms','median'),p95_wall_ms=('wall_ms',lambda x:x.quantile(.95)),mean_cpu_ms=('cpu_ms','mean'))
    summary['seconds_per_1000_pairs']=summary.mean_wall_ms
    for index,row in summary.iterrows():
        strata=medians[medians.hidden_size.eq(row.hidden_size)&medians.method.eq(row.method)].groupby('comparison').wall_ms.mean()
        summary.loc[index,'full_set_weighted_wall_ms']=(495*strata['within_subject']+5500*strata['across_subject'])/5995
    summary['estimated_one_core_minutes_35970_pairs']=summary.full_set_weighted_wall_ms*35970/1000/60
    summary.to_csv(OUT/'summary.tsv',sep='\t',index=False)
    q=quality.pivot(index=['model_id','hidden_size','pair_number'],columns='method',values=['distance','cosine'])
    assert (q['distance']['grid_plus_refinement']<=q['distance']['grid_only']+1e-5).all()
    improvements=pd.DataFrame(dict(model_id=q.index.get_level_values('model_id'),hidden_size=q.index.get_level_values('hidden_size'),
        distance_reduction=q['distance']['grid_only'].to_numpy()-q['distance']['grid_plus_refinement'].to_numpy(),
        cosine_change=q['cosine']['grid_plus_refinement'].to_numpy()-q['cosine']['grid_only'].to_numpy()))
    improvements.to_csv(OUT/'improvements.tsv',sep='\t',index=False)
    manifest=dict(pairs_per_variant=60,within_pairs=30,across_pairs=30,variants=12,unique_pairs=720,repeats=4,timed_calls=len(timing),
        selection='Minimum combined train+validation NLL, one inner fit per subject/variant/outer fold.',
        benchmark='Actual align_rnns CPU function, including deepcopy, parameter extraction, exhaustive grid/permutation search, applying transformations and calculating final distance. Model loading, imports, warm-up, cosine reporting and analysis excluded from per-call timings.',
        refined='Same grid; keep the winning permutation fixed; one positive-scale least-squares refinement initialized at winning scales, exact Jacobian, log bounds [-20,20], max 300 function evaluations. Not a refinement of every permutation.',
        order='Alternating method order across four passes. One allowed CPU core; NumPy/BLAS and PyTorch single-threaded. Medians over repeats per pair, then summarize.',
        cpu_affinity=affinity[0] if affinity else None,platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        extrapolation='35970 pairs = all 5995 pairs × six ReLU variants for one hidden size. One-core linear estimate, reweighting the 30/30 sample to 495 within and 5500 across pairs per variant; not an observed full-run or six-worker wall time.',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(mv.__file__)]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print('Refinement success:',quality[quality.method.eq('grid_plus_refinement')].refinement_success.value_counts().to_dict())


if __name__=='__main__':main()
