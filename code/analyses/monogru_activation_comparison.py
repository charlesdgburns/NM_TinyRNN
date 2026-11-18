"""Compare permutation-only and medoid-fitted monoGRU activations on identical inputs.

.venv/bin/python code/analyses/monogru_activation_comparison.py --jobs -1
No training or overwriting of the existing aligned examples.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
import numpy as np
import pandas as pd
import torch
from joblib import Parallel,delayed,parallel_config
from NM_TinyRNN.code.analyses import monoGRU_alignment as ma
from NM_TinyRNN.code.analyses import export_aligned_monoGRU as export
from NM_TinyRNN.code.measures import mechanistic_variability as mv,performance

COMPONENTS=['h1','h2','sum']
CONDITIONS=['Relabeled only','Medoid aligned']


def components(z):
    return np.concatenate([z,z.sum(-1,keepdims=True)],axis=-1)


def pearson(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x.mean();y=y-y.mean();denom=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(x@y/denom,-1,1)) if denom>1e-20 else np.nan


def paired_metrics(a,b):
    """Matched time samples; sum is across units at each trial, not across time."""
    a=components(np.asarray(a,dtype=float));b=components(np.asarray(b,dtype=float))
    records=[]
    for j,name in enumerate(COMPONENTS):
        x,y=a[:,j],b[:,j]
        records.append(dict(component=name,n_trials=len(x),mean_relabelled=float(x.mean()),mean_aligned=float(y.mean()),
            mean_change=float((y-x).mean()),mean_ratio=float(y.mean()/x.mean()) if x.mean()!=0 else np.nan,
            temporal_sd_relabelled=float(x.std(ddof=1)),temporal_sd_aligned=float(y.std(ddof=1)),
            integrated_relabelled=float(x.sum()),integrated_aligned=float(y.sum()),
            trajectory_r=pearson(x,y),rmse=float(np.sqrt(np.mean((y-x)**2)))))
    return records


def run_subject(subject,rows,reference_path,expected_hashes,out):
    torch.set_num_threads(1)
    rows=rows.sort_values('outer_loop_n')
    reference=ma.load_model(reference_path)
    raw_data=export.load_subject_data(subject,rows.iloc[0])
    a_all=[];b_all=[];splits=[];logits=[];gates=[];audits=[];metrics=[]
    for row in rows.itertuples():
        source=ma.resolve(row.model_pickle_path)
        if mv.sha256(source)!=expected_hashes[str(source)]:raise ValueError('Source checkpoint changed since medoid selection')
        model=ma.load_model(source)
        saved=pd.read_csv(ma.resolve(row.trials_data_path),sep='\t')
        # Condition 1: only the exact semantic permutation; scales stay 1.
        relabeled,label_info=ma.align_balanced_model(model,reference)
        np.testing.assert_array_equal(label_info['transform_vec'],np.ones(2))
        # Condition 2: reproduce the maintained two-stage fit, retaining its fixed medoid.
        balanced,balance_info=ma.balance_model(model)
        labeled_balanced,balance_labels=ma.align_balanced_model(balanced,balanced)
        fitted,fit_info=ma.fit_to_reference(labeled_balanced,reference)
        assert list(fit_info['perm'])==[0,1]
        perm=np.array(label_info['perm'])
        np.testing.assert_array_equal(perm,balance_labels['perm'])
        scales=np.exp([balance_info['log_scale_1'],balance_info['log_scale_2']])[perm]*fit_info['transform_vec']
        # Check this regenerated fit against the previously published aligned model.
        published=ma.load_model(row.aligned_model_path)
        for key,value in published.state_dict().items():torch.testing.assert_close(value,fitted.state_dict()[key],atol=2e-5,rtol=2e-5)
        df_a=export.regenerate_trials(relabeled,raw_data,saved)
        df_b=export.regenerate_trials(fitted,raw_data,saved)
        a=df_a[['hidden_1','hidden_2']].to_numpy();b=df_b[['hidden_1','hidden_2']].to_numpy()
        native=saved[['hidden_1','hidden_2']].to_numpy()
        np.testing.assert_allclose(a,native[:,perm],atol=5e-5,rtol=2e-4)
        np.testing.assert_allclose(a.sum(1),native.sum(1),atol=5e-5,rtol=2e-4)
        np.testing.assert_allclose(b,a*scales,atol=5e-5,rtol=2e-4)
        for col in ['logit_value','prob_A','prob_B','gate_update_1']:
            np.testing.assert_allclose(df_a[col],df_b[col],atol=5e-5,rtol=2e-4)
        for m in [relabeled,fitted]:
            contrast=(m.decoder.weight[0]-m.decoder.weight[1]).detach().numpy()
            assert contrast[0]>0 and contrast[1]<0
        a_all.append(a);b_all.append(b);splits.append(saved.split.to_numpy(dtype='U6'))
        logits.append(df_a.logit_value.to_numpy());gates.append(df_a.gate_update_1.to_numpy())
        nll_a,n_a=performance.cross_entropy_from_trials_df(df_a,'eval')
        nll_b,n_b=performance.cross_entropy_from_trials_df(df_b,'eval');assert n_a==n_b
        audits.append(dict(subject_id=subject,outer_loop_n=row.outer_loop_n,inner_loop_idx=row.inner_loop_idx,
            scale_h1=float(scales[0]),scale_h2=float(scales[1]),perm=json.dumps(perm.tolist()),
            label_only_sum_error=float(np.abs(a.sum(1)-native.sum(1)).max()),
            max_scale_identity_error=float(np.abs(b-a*scales).max()),
            max_logit_error=float(np.abs(df_a.logit_value-df_b.logit_value).max()),
            max_gate_error=float(np.abs(df_a.gate_update_1-df_b.gate_update_1).max()),
            eval_nll_relabelled=nll_a,eval_nll_aligned=nll_b,n_eval_free=n_a,
            distance_relabelled=float(np.linalg.norm(ma.core(ma.parameters(relabeled),2)-ma.core(ma.parameters(reference),2))),
            distance_aligned=fit_info['aligned_distance']))
        for split in ['all','train','val','eval']:
            keep=np.ones(len(a),dtype=bool) if split=='all' else saved.split.eq(split).to_numpy()
            if keep.sum()<2:raise ValueError('Insufficient matched trials')
            for record in paired_metrics(a[keep],b[keep]):
                metrics.append(dict(subject_id=subject,outer_loop_n=row.outer_loop_n,inner_loop_idx=row.inner_loop_idx,split=split,**record))
    a_all=np.stack(a_all);b_all=np.stack(b_all)
    subject_rows=[]
    for name,z in zip(CONDITIONS,[a_all,b_all]):
        c=components(z.astype(float))
        for j,component in enumerate(COMPONENTS):
            subject_rows.append(dict(subject_id=subject,condition=name,component=component,n_folds=len(rows),n_trials=len(raw_data),
                mean_activity=float(c[:,:,j].mean()),mean_pointwise_fold_sd=float(c[:,:,j].std(axis=0,ddof=1).mean())))
    arrays=out/'trajectories';arrays.mkdir(exist_ok=True)
    np.savez_compressed(arrays/f'{subject}.npz',hidden_relabelled=a_all,hidden_aligned=b_all,
        split=np.stack(splits),outer_folds=rows.outer_loop_n.to_numpy(),logit=np.stack(logits),gate=np.stack(gates),
        session=raw_data.session_folder_name.to_numpy(dtype=str),session_trial_idx=raw_data.session_trial_idx.to_numpy(),
        forced_choice=raw_data.forced_choice.to_numpy(),choice=raw_data.choice.to_numpy(),outcome=raw_data.outcome.to_numpy())
    raw_hashes={str((ROOT/'data/AB_behaviour'/session/'trials.htsv').relative_to(ROOT)):mv.sha256(ROOT/'data/AB_behaviour'/session/'trials.htsv') for session in raw_data.session_folder_name.unique()}
    print('Compared',subject,': ten folds,',len(raw_data),'trials per condition',flush=True)
    return dict(metrics=metrics,audits=audits,subjects=subject_rows,data_sha256=raw_hashes)


def plots(out,figures,subject,outer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subjects=pd.read_csv(out/'subject_summary.htsv',sep='\t')
    subject_ids=sorted(subjects.subject_id.unique());colors=dict(zip(subject_ids,plt.cm.tab20(np.linspace(0,1,len(subject_ids)))))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for j,component in enumerate(COMPONENTS):
        for r,(metric,ylabel) in enumerate([('mean_activity','Mean activity'),('mean_pointwise_fold_sd','Mean pointwise SD across ten folds')]):
            ax=axes[r,j];p=subjects[subjects.component.eq(component)].pivot(index='subject_id',columns='condition',values=metric)
            for sub,row in p.iterrows():ax.plot([0,1],[row[CONDITIONS[0]],row[CONDITIONS[1]]],'o-',color=colors[sub],alpha=.75,lw=1,ms=4,label=sub if (r==0 and j==0) else None)
            ax.plot([0,1],p[CONDITIONS].mean().to_numpy(),'kD-',lw=2.8,ms=6)
            ax.set(xticks=[0,1],xticklabels=['Relabeled\nonly','Medoid\naligned'],ylabel=ylabel,title=['h1 (A1)','h2 (A2)','h1 + h2'][j] if r==0 else '')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Permutation-only versus medoid-fitted activations\nLines: subjects; black: equal-subject mean · Same trials in both conditions')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=11,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig,figures/'subject_comparison');plt.close(fig)
    with np.load(out/'trajectories'/f'{subject}.npz') as data:
        a=data['hidden_relabelled'];b=data['hidden_aligned'];folds=data['outer_folds']
        index=int(np.flatnonzero(folds==outer)[0]);session=data['session'];last=session[-1]
        sample=np.flatnonzero(session==last)[:200];time=data['session_trial_idx'][sample]
        x=components(a[index]);y=components(b[index]);logit=data['logit'][index]
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for j,component in enumerate(COMPONENTS):
            axes[0,j].plot(time,x[sample,j],label=CONDITIONS[0],color='#4078b7',lw=1.3)
            axes[0,j].plot(time,y[sample,j],label=CONDITIONS[1],color='#df7934',lw=1.3)
            axes[0,j].set(title=['h1 (A1)','h2 (A2)','h1 + h2'][j],xlabel='Trial within session',ylabel='Activity')
            axes[0,j].grid(alpha=.15)
            sc=axes[1,j].scatter(x[:,j],y[:,j],c=logit,cmap='coolwarm',s=5,alpha=.35,edgecolors='none')
            limit=max(float(x[:,j].max()),float(y[:,j].max()))
            axes[1,j].plot([0,limit],[0,limit],'k--',lw=1)
            axes[1,j].set(xlabel='Relabeled-only activity',ylabel='Medoid-aligned activity',title=f'All trials: r = {pearson(x[:,j],y[:,j]):.4f}')
            axes[1,j].set_aspect('equal',adjustable='box');axes[1,j].grid(alpha=.15)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{subject}, outer fold {outer} · identical complete histories\nTop: last session, first 200 trials; bottom: full subject history · Dashed: identity')
        fig.tight_layout();save(fig,figures/'example_activations');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True,sharey='col')
        for r,(name,z) in enumerate(zip(CONDITIONS,[a,b])):
            c=components(z)
            for j in range(3):
                for fold in range(len(folds)):axes[r,j].plot(time,c[fold,sample,j],color=plt.cm.tab10(fold),lw=.65,alpha=.55)
                axes[r,j].plot(time,c[:,sample,j].mean(0),'k-',lw=1.8)
                axes[r,j].set(xlabel='Trial within session',ylabel=name,title=['h1 (A1)','h2 (A2)','h1 + h2'][j] if r==0 else '')
                axes[r,j].grid(alpha=.15)
        fig.suptitle(f'{subject} · ten outer-fold responses to the same inputs\nThin colored lines: folds; black: fold mean · Matched y limits within columns')
        fig.tight_layout();save(fig,figures/'fold_trajectories');plt.close(fig)


def save(fig,path):
    for ext in ['png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')


def write_report(out,figures):
    subjects=pd.read_csv(out/'subject_summary.htsv',sep='\t')
    fits=pd.read_csv(out/'fit_metrics.htsv',sep='\t')
    checks=pd.read_csv(out/'fit_checks.htsv',sep='\t')
    lines=['# Relabeling versus medoid alignment: hidden activations','',
        '110 selected two-unit monoGRU +BC-DB ReLU models from nested_DA_S32_B8: 11 subjects × ten outer folds. Both conditions were freshly replayed on identical raw subject histories.', '',
        'Condition 1 applies only the h1 → A1, h2 → A2 permutation using mechanistic_variability.align_rnns with scales fixed at one. It does not independently balance weights. Condition 2 reproduces the maintained minimum-norm initialization followed by positive-scale Euclidean/Frobenius core-parameter fitting to the fixed balanced medoid (WS14, outer 3), preserving the action labels. Every refit was checked against its previously published aligned model.', '',
        'Primary summaries include all trials (forced and free, all original splits). State runs continuously across the complete subject history. train/val/eval summaries apply masks afterward and are provided as sensitivities, not reset-based hidden-state evaluations.', '',
        '| Component | Mean, relabeled | Mean, aligned | Mean change | Pointwise fold SD, relabeled | Pointwise fold SD, aligned | SD change |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for component in COMPONENTS:
        p=subjects[subjects.component.eq(component)].pivot(index='subject_id',columns='condition',values=['mean_activity','mean_pointwise_fold_sd'])
        a=p['mean_activity'][CONDITIONS[0]].mean();b=p['mean_activity'][CONDITIONS[1]].mean()
        x=p['mean_pointwise_fold_sd'][CONDITIONS[0]].mean();y=p['mean_pointwise_fold_sd'][CONDITIONS[1]].mean()
        lines.append(f'| {component} | {a:.4f} | {b:.4f} | {100*(b/a-1):+.1f}% | {x:.4f} | {y:.4f} | {100*(y/x-1):+.1f}% |')
    sums=fits.query('split=="all" and component=="sum"')
    worst=sums.loc[sums.trajectory_r.idxmin()]
    p=subjects[subjects.component.eq('sum')].pivot(index='subject_id',columns='condition',values='mean_pointwise_fold_sd')
    lines += ['',
        'Means first average trials/folds within each subject, then subjects equally. Pointwise fold SD means: at each common input trial, take the sample SD across ten outer-fold responses, then average over trials and subjects. This quantifies across-fit amplitude variability; it is not uncertainty of the group mean. Folds share training data and no independent-fold significance test is used.', '',
        f'Summed activity (h1+h2 at each trial) has a median between-condition trajectory correlation of {sums.trajectory_r.median():.6f}, with range [{sums.trajectory_r.min():.6f}, {sums.trajectory_r.max():.6f}]. The smallest correlation occurs for {worst.subject_id}, outer fold {int(worst.outer_loop_n)}; this is an exploratory extreme, not the plotted example. Summed pointwise fold SD decreases in {int((p[CONDITIONS[1]]<p[CONDITIONS[0]]).sum())}/11 subjects.', '',
        f'Permutation alone preserves the original sum (maximum numerical discrepancy {checks.label_only_sum_error.max():.3g}). For every model, h_aligned,j(t) = d_j h_relabelled,j(t), checked by direct replay. Thus individual-unit Pearson correlations are one up to numerical precision, while the unweighted sum becomes d1*h1+d2*h2 and can change its time course when d1 differs from d2.', '',
        f'Behavior is preserved: maximum logit difference {checks.max_logit_error.max():.3g}, update-gate difference {checks.max_gate_error.max():.3g}, and outer continuous NLL difference {abs(checks.eval_nll_aligned-checks.eval_nll_relabelled).max():.3g} nats/free choice.', '',
        'The sum is dependent on the hidden-coordinate scale convention. A change in it under this exact symmetry is not a change in predictive computation, and the parameter-distance objective does not guarantee reduced activation variability for every subject.', '',
        '## Files', '',
        '- fit_metrics.htsv: matched all/train/val/eval unit and sum metrics for every model.',
        '- fit_checks.htsv: permutations, scale gains, prediction/gate preservation and NLL checks.',
        '- subject_summary.htsv: equal-subject aggregation and pointwise across-fold SD.',
        '- subject_fit_metrics.htsv: subject averages of fit metrics.',
        '- trajectories/<subject>.npz: both freshly recomputed hidden trajectories, outer-fold IDs, original split masks and shared trial identifiers. Array shape is (outer fold, trial, hidden unit).',
        '- manifest.json: source model, raw input and reference hashes.',
        f'- Figures: {figures}', '',
        'The time-series example is the predefined notebook example WS20, outer fold 10; the last session is cropped to its first 200 trials only for display, after full-history inference. The across-fold display includes all ten fits on those same inputs. All data and inference checks use complete histories.', '',
        'Reproduce: `.venv/bin/python code/analyses/monogru_activation_comparison.py --jobs -1`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=-1)
    parser.add_argument('--example-subject',default='WS20');parser.add_argument('--example-outer',type=int,default=10)
    parser.add_argument('--alignment-dir',type=Path,default=ROOT/'data/analyses/monoGRU_alignment/nested_DA_S32_B8/monoGRU_relu_ndb')
    parser.add_argument('--output',type=Path,default=ROOT/'data/analyses/monogru_activation_comparison')
    parser.add_argument('--figures',type=Path,default=ROOT/'results/appendix/monogru_activation_comparison')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.alignment_dir/'manifest.json').read_text())
    assert manifest['alignment_protocol']=='balanced_medoid_reference_fit_v3_choice0_first'
    rows=pd.read_csv(args.alignment_dir/'aligned_models.htsv',sep='\t');assert len(rows)==110
    reference=args.alignment_dir/'reference_model.pickle'
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=args.jobs)(delayed(run_subject)(sub,group,reference,manifest['source_model_sha256'],args.output) for sub,group in rows.groupby('subject_id',sort=True))
    fits=pd.DataFrame([r for result in results for r in result['metrics']]);audits=pd.DataFrame([r for result in results for r in result['audits']]);subjects=pd.DataFrame([r for result in results for r in result['subjects']])
    fits.to_csv(args.output/'fit_metrics.htsv',sep='\t',index=False);audits.to_csv(args.output/'fit_checks.htsv',sep='\t',index=False);subjects.to_csv(args.output/'subject_summary.htsv',sep='\t',index=False)
    subject_metrics=fits.groupby(['subject_id','split','component'],as_index=False).mean(numeric_only=True)
    subject_metrics.to_csv(args.output/'subject_fit_metrics.htsv',sep='\t',index=False)
    summary=subjects.groupby(['component','condition'],as_index=False)[['mean_activity','mean_pointwise_fold_sd']].mean()
    summary.to_csv(args.output/'summary.htsv',sep='\t',index=False)
    plots(args.output,args.figures,args.example_subject,args.example_outer)
    write_report(args.output,args.figures)
    definition=dict(source_run=manifest['folder'],models=len(rows),subjects=rows.subject_id.nunique(),jobs=args.jobs,
        condition_1='Only semantic permutation via mechanistic_variability.align_rnns; scaling fixed at 1; NO norm balancing',
        condition_2='Independent minimum-norm balancing then positive-scale Euclidean/Frobenius fit to the frozen balanced medoid, with semantic roles fixed; freshly replayed',
        activity='Continuous whole-subject histories on identical raw data, including forced trials and original split labels',
        sum='h1+h2 at each trial; integrated sums also recorded; activity scales are coordinate conventions, not physical units',
        aggregation='Ten matched folds averaged within subject, then 11 subjects equally; pointwise across-fold SD uses same full input history',
        example=dict(subject=args.example_subject,outer=args.example_outer,selection='Predefined notebook example, not chosen by effect'),
        reference_source=manifest['reference_source'],reference_sha256=mv.sha256(reference),selected_sha256=mv.sha256(args.alignment_dir/'aligned_models.htsv'),
        source_model_sha256=manifest['source_model_sha256'],data_sha256={k:v for result in results for k,v in result['data_sha256'].items()},
        code_sha256={str(Path(p).relative_to(ROOT)):mv.sha256(Path(p)) for p in [__file__,ma.__file__,mv.__file__,export.__file__]},
        max_logit_error=float(audits.max_logit_error.max()),max_label_only_sum_error=float(audits.label_only_sum_error.max()))
    (args.output/'manifest.json').write_text(json.dumps(definition,indent=2,default=ma.json_value)+'\n')
    print(summary.to_string(index=False),flush=True)
    print('Sum trajectory correlations:',fits.query('split=="all" and component=="sum"').trajectory_r.describe().to_dict(),flush=True)


if __name__=='__main__':main()
