"""Maintained conservative GRU/monoGRU report for the current selected-model cache."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT.parent))
import numpy as np,pandas as pd
from scipy.stats import binomtest,wilcoxon,false_discovery_control
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.figures import gru_monogru_report as plotting

METRICS=[('logit_value_similarity','Logit'),('weight_similarity','Weight'),('hidden_similarity','Hidden state'),('gate_update_similarity','Update gate')]
MODELS=[mv.STANDARD_GRU_REFERENCE,'2_unit_monoGRU_relu_unipolar_forced_ndb']


def generate_report(pairs,root=ROOT,*,include_main=True,include_appendix=False):
    root=Path(root);tables=root/'data/analyses/mechanistic_variability/main';tables.mkdir(parents=True,exist_ok=True)
    main=root/'results/main';appendix=root/'results/appendix/mechanistic_variability/gru_monogru'
    if include_main:main.mkdir(parents=True,exist_ok=True)
    if include_appendix:appendix.mkdir(parents=True,exist_ok=True)
    subjects,_=mv.summarize_reference_pairs(pairs[pairs.model_id.isin(MODELS)])
    rows=[];paired=[]
    for metric,label in METRICS:
        for scope in [True,False]:
            f=subjects[subjects.metric.eq(metric)&subjects.within_subject.eq(scope)]
            a=f[f.model_id.eq(MODELS[0])][['subject_A','subject_B','mean_similarity']].rename(columns={'mean_similarity':'gru'})
            b=f[f.model_id.eq(MODELS[1])][['subject_A','subject_B','mean_similarity']].rename(columns={'mean_similarity':'mono'})
            d=a.merge(b,on=['subject_A','subject_B'],validate='one_to_one');d['difference']=d.mono-d.gru;d['metric']=metric;d['component']=label;d['within_subject']=scope;paired.append(d)
            names=sorted(set(d.subject_A)|set(d.subject_B));assert len(names)==11 and len(d)==(11 if scope else 55)
            c=mv.contrast(d.rename(columns={'gru':'reference','mono':'model'}))
            median=float(d.difference.median());mean=float(d.difference.mean());fn=np.median if scope else np.mean
            loo=np.array([fn(d.loc[~(d.subject_A.eq(s)|d.subject_B.eq(s)),'difference']) for s in names])
            median_loo=np.array([d.loc[~(d.subject_A.eq(s)|d.subject_B.eq(s)),'difference'].median() for s in names])
            positive=int((d.difference>0).sum());nonzero=int((d.difference!=0).sum())
            rows.append(dict(metric=metric,component=label,within_subject=scope,n_units=len(d),gru_mean=d.gru.mean(),gru_sd=d.gru.std(ddof=1),mono_mean=d.mono.mean(),mono_sd=d.mono.std(ddof=1),
                median_difference=median,median_deletion_min=median_loo.min(),median_deletion_max=median_loo.max(),mean_difference=mean,deletion_min=loo.min(),deletion_max=loo.max(),deletion_estimator='median' if scope else 'mean',cohens_dz=c['cohens_dz'],
                delta_ci95_low=c['delta_ci95_low'],delta_ci95_high=c['delta_ci95_high'],
                p_jackknife=c['p_raw'] if not scope else np.nan,p_wilcoxon=c['p_wilcoxon'] if scope else np.nan,
                n_positive=positive,n_nonzero=nonzero,p_sign=binomtest(positive,nonzero,.5).pvalue if scope else np.nan))
    summary=pd.DataFrame(rows);mask=summary.within_subject;summary['q_sign_bh4']=np.nan
    summary.loc[mask,'q_sign_bh4']=false_discovery_control(summary.loc[mask,'p_sign'].to_numpy(),method='bh')
    summary['p_main']=np.where(summary.within_subject,summary.p_wilcoxon,summary.p_jackknife)
    summary['q_main_bh8']=false_discovery_control(summary.p_main.to_numpy(),method='bh')
    paired=pd.concat(paired,ignore_index=True);summary.to_csv(tables/'summary.csv',index=False);paired.to_csv(tables/'paired_subject_data.csv',index=False)
    if include_main:plotting.table_one(summary,main/'table_1.svg')
    main_rows=[]
    for metric,label in METRICS:
        record=dict(component=label)
        for scope,name in [(True,'within'),(False,'across')]:
            r=summary[summary.metric.eq(metric)&summary.within_subject.eq(scope)].iloc[0]
            record.update({name+'_mean_difference':r.mean_difference,name+'_ci95_low':r.delta_ci95_low,name+'_ci95_high':r.delta_ci95_high,name+'_cohens_dz':r.cohens_dz})
            record.update({name+'_GRU':r.gru_mean,name+'_monoGRU':r.mono_mean,name+'_median_difference':r.median_difference,name+'_median_deletion_min':r.median_deletion_min,name+'_median_deletion_max':r.median_deletion_max})
            record[name+'_p_main']=r.p_main;record[name+'_q_main_bh8']=r.q_main_bh8
            if scope:record['within_q_sign_bh4']=r.q_sign_bh4
        main_rows.append(record)
    if include_main:pd.DataFrame(main_rows).to_csv(main/'table_1.csv',index=False)
    if include_appendix:
        paired.to_csv(appendix/'paired_subject_data.csv',index=False)
        for scope,name in [(True,'within'),(False,'across')]:
            f=summary[summary.within_subject.eq(scope)];cells=[]
            for r in f.itertuples():
                row=[r.component,f'{r.gru_mean:.3f} ± {r.gru_sd:.3f}',f'{r.mono_mean:.3f} ± {r.mono_sd:.3f}',f'{r.median_difference:+.4f}',f'{r.mean_difference:+.4f}',f'[{r.delta_ci95_low:+.4f}, {r.delta_ci95_high:+.4f}]',f'{r.cohens_dz:+.2f}']
                if scope:row.extend([f'{r.n_positive}/{r.n_nonzero}',f'{r.p_wilcoxon:.4f}',f'{r.q_main_bh8:.4f}'])
                else:row.extend([f'[{r.median_deletion_min:+.4f}, {r.median_deletion_max:+.4f}]',f'{r.p_jackknife:.4g}',f'{r.q_main_bh8:.4f}'])
                cells.append(row)
            headers=['Component','GRU mean ± SD','monoGRU mean ± SD','Median Δ','Mean Δ','95% CI (mean Δ)',r'Cohen’s $d_z$']+(['Positive Δ','Wilcoxon p','BH q (8)'] if scope else ['Median deletion range','Jackknife p','BH q (8)'])
            foot=('Within: 11 independent paired subject summaries. 95% CI: t(10) interval for the mean difference; d_z = mean/SD of subject differences; p: two-sided Wilcoxon signed-rank test; BH over eight main-table tests.' if scope else 'Across: 55 dependent dyads sharing 11 subjects. 95% CI and p: subject-deletion jackknife, t(10); d_z = mean/(jackknife SE x sqrt(11)), the jackknife-consistent subject-level effect; BH over eight main-table tests.')
            plotting.table_figure(cells,headers,appendix/f'table_{name}.svg',('Within-subject' if scope else 'Across-subject')+' similarity: supplementary statistics',foot+'\nDeletion ranges describe influence, not confidence intervals. Block-reset outer selection; nested_DA_S32_B8.',figsize=(19,4))
            f.to_csv(appendix/f'table_{name}.csv',index=False)
        for metric,label in METRICS:plotting.paired_plot(paired[paired.metric.eq(metric)],summary[summary.metric.eq(metric)],appendix/f'figure_paired_{metric}.svg',label)
    files=([main/'table_1.svg',main/'table_1.csv'] if include_main else [])+(list(appendix.glob('*')) if include_appendix else [])
    (tables/'manifest.json').write_text(json.dumps(dict(source_run='nested_DA_S32_B8',selection='block_reset',within='Paired subject differences; two-sided Wilcoxon signed-rank (exact sign test kept as p_sign sensitivity)',across='Mean paired dyad differences; subject-deletion jackknife t; approximate',main_correction='BH over eight Wilcoxon-within/jackknife-across comparisons',effect_size='d_z = mean/(jackknife SE*sqrt(n)) at both levels',script_sha256=mv.sha256(__file__),plotting_sha256=mv.sha256(plotting.__file__),pairwise_source_sha256=mv.sha256(root/'data/analyses/mechanistic_variability/pairwise_similarities.htsv'),outputs_sha256={str(p.relative_to(root)):mv.sha256(p) for p in files if p.is_file()}),indent=2)+'\n')
    return summary


if __name__=='__main__':
    selected=pd.read_csv(ROOT/'data/analyses/mechanistic_variability/selected_models.htsv',sep='\t')
    if not selected.selection_method.eq('block_reset').all():raise ValueError('Report requires current block-reset winners')
    pairs=pd.read_csv(ROOT/'data/analyses/mechanistic_variability/pairwise_similarities.htsv',sep='\t');mv.validate_similarity_cache_models(pairs,selected)
    print(generate_report(pairs).round(5).to_string(index=False))
