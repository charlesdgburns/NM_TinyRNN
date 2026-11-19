"""Figures for the mechanistic-similarity variance decomposition."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_family_colors import COLOR_BY_FAMILY

METRICS = ['weight_similarity', 'logit_value_similarity', 'hidden_similarity', 'gate_update_similarity']
LABELS = ['Weights', 'Logits', 'Hidden state', 'Update gate']


def bayesian_summary(data, out):
    """Combine fitted-model components draw by draw; present estimates and updating."""
    import json
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    manifest = json.loads((data / 'manifest.json').read_text())
    fingerprint = manifest['fingerprint'][:12]
    evidence = pd.read_csv(data / 'bayes_factors.csv')
    evidence = evidence[evidence.prior_scale.eq(1) & evidence.hypothesis.eq('subject_fraction > fitted_solution_fraction')]
    summaries = []
    absolute_rows = []
    for model in ['GRU', 'monoGRU']:
        for within in [True, False]:
            for metric in METRICS:
                path = data / 'posteriors' / f'{model}_{metric}_{"within" if within else "across"}_{fingerprint}.npz'
                with np.load(path) as saved:
                    sd = saved['sd']
                variance = sd**2 * np.array([1 if within else 2, 0 if within else 1, 2, 2, 1])
                fraction = variance / variance.sum(axis=1, keepdims=True)
                values = {'subject': fraction[:, 0], 'fitted_solution': fraction[:, 2:4].sum(axis=1),
                          'subject_pair': fraction[:, 1], 'pair_residual': fraction[:, 4]}
                absolute_values = {'subject': variance[:, 0], 'fitted_solution': variance[:, 2:4].sum(axis=1),
                                   'subject_pair': variance[:, 1], 'pair_residual': variance[:, 4]}
                row = dict(model=model, metric=metric, component=LABELS[METRICS.index(metric)], within_subject=within)
                for key, draws in values.items():
                    row[key+'_mean']=float(np.mean(draws))
                    row[key+'_lo'], row[key+'_median'], row[key+'_hi'] = np.quantile(draws, [.025, .5, .975])
                    lo,median,hi = np.quantile(absolute_values[key],[.025,.5,.975])
                    row[key+'_variance_lo'],row[key+'_variance_median'],row[key+'_variance_hi']=lo,median,hi
                    if key!='subject_pair' or not within:
                        absolute_rows.append(dict(model=model,metric=metric,within_subject=within,component=key,
                                                  variance_median=median,variance_lo=lo,variance_hi=hi,
                                                  scale='atanh(similarity)^2'))
                e = evidence[evidence.model.eq(model) & evidence.metric.eq(metric) & evidence.within_subject.eq(within)]
                assert len(e) == 1
                e = e.iloc[0]
                probability = np.mean(values['subject'] > values['fitted_solution'])
                np.testing.assert_allclose(probability, e.posterior_probability, atol=1e-12)
                for key in ['posterior_probability', 'prior_probability', 'bayes_factor', 'bf_resolved',
                            'predictive_checks_outside_95', 'sampling_ok']:
                    row[key] = e[key]
                summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(data / 'bayesian_summary.csv', index=False)
    pd.DataFrame(absolute_rows).to_csv(data/'hierarchical_absolute_variance.csv',index=False)
    colors = {family: COLOR_BY_FAMILY[family] for family in ('GRU', 'monoGRU')}
    symbols = {'GRU': 'o', 'monoGRU': 's'}
    legend = [Line2D([], [], color=colors[m], marker=symbols[m], ls='', label=m) for m in colors]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, within in zip(axes, [True, False]):
        subset = summary[summary.within_subject.eq(within)]
        prior = subset.prior_probability.iloc[0]
        ax.axvline(prior,color='.4',ls='--',lw=1.4)
        ax.text(prior,.99,f'Prior {prior:.1%}',transform=ax.get_xaxis_transform(),ha='center',va='top',
                fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':2})
        for j, model in enumerate(colors):
            rows = subset[subset.model.eq(model)].set_index('metric').loc[METRICS]
            for k, (_, row) in enumerate(rows.iterrows()):
                y = k + (j-.5)*.24
                p = row.posterior_probability
                ax.plot([prior,p],[y,y],color=colors[model],alpha=.4,lw=2)
                ax.plot(p,y,marker=symbols[model],color=colors[model],ms=7,ls='')
                label = f'BF {row.bayes_factor:.1f}' if row.bf_resolved else 'BF unresolved'
                right = p < .72
                ax.text(p+(.025 if right else -.025),y,label,ha='left' if right else 'right',va='center',fontsize=10)
        ax.set_title('Within subject' if within else 'Across subjects',pad=15)
        ax.set_yticks(range(4),LABELS)
        ax.set_ylim(3.5,-.6)
        ax.set_xlim(0,1.04)
        ax.set_xticks(np.linspace(0,1,5))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Posterior probability: subject variance > fitted-solution variance')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Does subject variation exceed fitted-solution variation?',fontsize=18,y=.99)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False)
    fig.text(.5,.025,'Dashed line: prior probability. BF = posterior odds / prior odds. BF > 1 favours the stated inequality; BF < 1 favours its complement.\n'
             'Unresolved: too few posterior draws in the opposite region to estimate finite tail odds reliably.\n'
             'EXPLORATORY: these are conditional on a Gaussian model with predictive discrepancies; they are not p-values.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.23,wspace=.24)
    fig.savefig(out / 'figure_bayesian_evidence.svg',bbox_inches='tight')
    plt.close(fig)
    hierarchical_compact(summary,out)
    hierarchical_compact(summary,out,filename='figure_hierarchical_variance_clean.svg',show_title=False,show_description=False)
    bayesian_question1(summary,data,out)
    return summary


def hierarchical_compact(summary,out,absolute=False,filename=None,*,figsize=None,panel_wspace=.30,panel_hspace=1.0,section_hspace=.65,show_title=True,show_description=True):
    """All sources, with fitted common/role contributions combined draw by draw."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    if absolute:
        fig,axes=plt.subplots(2,4,figsize=figsize or (18,9),sharey=True)
    else:
        fig=plt.figure(figsize=figsize or (18,20))
        grid=fig.add_gridspec(2,1,hspace=section_hspace)
        sections=[grid[i].subgridspec(2,4,hspace=panel_hspace,wspace=panel_wspace) for i in range(2)]
        axes=np.array([[fig.add_subplot(sections[r//2][r%2,c]) for c in range(4)] for r in range(4)])
        for r in range(4):
            anchor=axes[0 if r<2 else 2,0]
            for ax in axes[r]:
                if ax is not anchor:ax.sharey(anchor)
    model_colors={family: COLOR_BY_FAMILY[family] for family in ('GRU', 'monoGRU')}
    labels={'subject':'Subject','subject_pair':'Subject\npair',
            'fitted_solution':'Outer-fold\nfit','pair_residual':'Model-fit\npair\n(residual)'}
    absolute_upper=max(summary[k+'_variance_hi'].max() for k in labels)*1.05
    for r in range(len(axes)):
        within=r%2==0
        is_absolute=absolute or r>=2
        suffix='_variance' if is_absolute else ''
        components=['subject','fitted_solution','pair_residual'] if within else ['subject','subject_pair','fitted_solution','pair_residual']
        for c,metric in enumerate(METRICS):
            ax=axes[r,c]
            for j,model in enumerate(['GRU','monoGRU']):
                row=summary[summary.model.eq(model)&summary.within_subject.eq(within)&summary.metric.eq(metric)].iloc[0]
                for k,component in enumerate(components):
                    median=row[component+suffix+'_median']
                    ax.errorbar(k+(j-.5)*.17,median,
                                yerr=[[median-row[component+suffix+'_lo']],[row[component+suffix+'_hi']-median]],
                                fmt='o' if j==0 else 's',color=model_colors[model],
                                markerfacecolor=model_colors[model],
                                capsize=5,capthick=2,ms=8.5,lw=2.4)
            if r%2==0:ax.set_title(LABELS[c],fontsize=20,pad=15)
            tick_labels=[('\n' if not within and i%2 else '')+labels[k] for i,k in enumerate(components)]
            ax.set_xticks(range(len(components)),tick_labels,rotation=0,ha='center',fontsize=16)
            ax.set_xlim(-.25,len(components)-.75);ax.set_ylim(0,absolute_upper if is_absolute else 1)
            if not is_absolute:ax.yaxis.set_major_formatter(PercentFormatter(1))
            if c>0:ax.tick_params(axis='y',labelleft=False)
            ax.tick_params(axis='both',labelsize=16,width=1.2,length=5,pad=5)
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        axes[r,0].set_ylabel(('Within subject' if within else 'Across subjects')+('\nVariance (z²)' if is_absolute else '\nVariance share'),fontsize=17,labelpad=14)
    if show_title:
        fig.suptitle('Hierarchical absolute variance contributions' if absolute else 'Hierarchical variance contributions',fontsize=26,y=.99)
    fig.legend(handles=[Line2D([],[],marker='o',color=model_colors['GRU'],markersize=9,ls='',label='GRU'),
                        Line2D([],[],marker='s',color=model_colors['monoGRU'],markersize=9,ls='',label='monoGRU')],
               loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False,fontsize=19,handletextpad=.5,columnspacing=2)
    footnote=('Medians and 95% credible intervals; z = atanh(similarity). All panels share one absolute y-axis; these are not raw-similarity variances.\n'
              if absolute else 'Points: posterior medians; lines: 95% credible intervals.\nTop: variance shares. Bottom: absolute variance in z², where z = atanh(similarity).\n')
    if show_description:
        fig.text(.5,.018,footnote+'Model-fit pair (residual): comparison-specific deviations after shared effects.\nExploratory: predictive discrepancies remain.',ha='center',fontsize=14)
    if not absolute:
        fig.text(.025,.90,'a   Proportional variance contributions',fontsize=22,fontweight='bold')
        fig.text(.025,.46,'b   Absolute variance contributions',fontsize=22,fontweight='bold')
        fig.subplots_adjust(left=.08,right=.99,top=.855,bottom=.12)
    else:
        fig.subplots_adjust(left=.08,right=.99,top=.80,bottom=.23,hspace=panel_hspace,wspace=panel_wspace)
    fig.savefig(out/(filename or ('figure_hierarchical_absolute_variance.svg' if absolute else 'figure_hierarchical_variance.svg')),bbox_inches='tight');plt.close(fig)


def bayesian_question1(summary,data,out):
    """All component shares and absolute variances, with credible intervals."""
    target=out/'bayesian';target.mkdir(exist_ok=True)
    hierarchical_compact(summary,target,filename='figure_question1_variance_contributions.svg')
    components=['subject','subject_pair','fitted_solution','pair_residual']
    columns=['model','metric','within_subject']+[component+suffix for component in components
              for suffix in ['_mean','_median','_lo','_hi','_variance_median','_variance_lo','_variance_hi']]
    summary[columns].to_csv(data/'question1_variance_contributions.csv',index=False)


def descriptive_variance_plots(d,out):
    """Percentage above absolute variance; every absolute axis uses one scale."""
    from matplotlib.ticker import PercentFormatter
    from matplotlib.colors import to_rgb
    names=['Between subjects','Outer fold effects','Pair-specific effects']
    model_colors={family: COLOR_BY_FAMILY[family] for family in ('GRU', 'monoGRU')}
    opacity=[1.,.65,.30]
    # Blend onto white explicitly, keeping grid lines from showing through fills.
    shades={model:[tuple(1-a+a*np.array(to_rgb(color))) for a in opacity] for model,color in model_colors.items()}
    def grid_legend(fig,top,height):
        from matplotlib.patches import Rectangle
        centers=[.35,.55,.75]
        for x,name in zip(centers,names):
            fig.text(x,top,name,ha='center',va='center',fontsize=10,fontweight='bold')
        for r,model in enumerate(model_colors):
            y=top-height*(.4+.4*r)
            fig.text(.235,y,model,ha='right',va='center',fontsize=10)
            for k,x in enumerate(centers):
                fig.add_artist(Rectangle((x-.0875,y-height*.135),.175,height*.27,
                                         transform=fig.transFigure,facecolor=shades[model][k],
                                         edgecolor='white',linewidth=.7,clip_on=False))
    limit=np.ceil(d.total_variance.max()*1.08/.02)*.02
    note=('Orange: GRU; teal: monoGRU. Dark to light: between subjects, outer fold, pair-specific effects. Across-subject groups are subject pairs.\n'
          'Outer fold effects describe variation among selected fits, including split, selection and initialisation differences.')

    def bars(ax,within,percentage):
        columns=['between_group_fraction','endpoint_fraction','interaction_fraction'] if percentage else ['between_group_variance','endpoint_variance','interaction_variance']
        for j,model in enumerate(['GRU','monoGRU']):
            rows=d[d.model.eq(model)&d.within_subject.eq(within)].set_index('metric').loc[METRICS]
            x=np.arange(4)+(j-.5)*.36
            bottom=np.zeros(4)
            for k,column in enumerate(columns):
                color=shades[model][k]
                heights=rows[column].to_numpy()
                ax.bar(x,heights,.33,bottom=bottom,color=color,edgecolor='white',linewidth=.6)
                if percentage:
                    for xx,base,height in zip(x,bottom,heights):
                        if height>.055:ax.text(xx,base+height/2,f'{height:.0%}',ha='center',va='center',color='white' if k==0 else '#222222',fontsize=10)
                bottom+=heights
        ax.set_xticks(range(4),LABELS,rotation=15,ha='right')
        ax.set_title('Within subject' if within else 'Across subjects',pad=12)
        ax.set_ylim(0,1 if percentage else limit)
        if percentage:ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)

    fig,axes=plt.subplots(2,2,figsize=(13.5,8),sharex='col')
    axes[0,1].sharey(axes[0,0]);axes[1,1].sharey(axes[1,0])
    for r in range(2):
        for c,within in enumerate([True,False]):bars(axes[r,c],within,r==0)
    for ax in axes.flat:ax.tick_params(axis='x',labelbottom=True)
    axes[0,0].set_ylabel('Percentage of total variance')
    axes[1,0].set_ylabel('Absolute similarity variance')
    fig.suptitle('Similarity variance breakdown',fontsize=19,y=1.015)
    grid_legend(fig,top=.96,height=.07)
    fig.text(.025,.85,'a   Percentage breakdown',fontsize=14,fontweight='bold')
    fig.text(.025,.465,'b   Absolute variances',fontsize=14,fontweight='bold')
    fig.text(.5,.025,note,ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,top=.80,bottom=.13,hspace=.65,wspace=.15)
    fig.savefig(out/'figure_variance_breakdown.svg',bbox_inches='tight');plt.close(fig)


def plot_results(data,out):
    """Publish maintained variance figures and the clean hierarchical variant."""
    plt.rcParams.update({'svg.fonttype':'none','font.size':12})
    out.mkdir(parents=True,exist_ok=True)
    descriptive_variance_plots(pd.read_csv(data/'descriptive_variance.csv'),out)
    if (data/'posterior_components.csv').exists():
        bayesian_summary(data,out)
    (out/'README.md').write_text("""# Mechanistic similarity variance

Dataset: nested_DA_S32_B8; block-reset-selected outer winners.
Standard GRU (tanh, decoder bias) versus monoGRU (ReLU, no decoder bias).

- figure_variance_breakdown.svg: exact descriptive raw-similarity variance;
  percentage breakdown above absolute variance, with shared absolute axes.
- figure_hierarchical_variance_clean.svg: matching version without an overall title or bottom description.
- figure_hierarchical_variance.svg: proportional scatterplots above absolute
  scatterplots, with posterior medians and 95% credible intervals. All absolute
  panels share a y-axis; no tables are embedded.
- figure_bayesian_evidence.svg: prior-to-posterior updating for subject variance
  exceeding fitted-solution variance. Bayes factors concern this inequality,
  not exact-zero component inclusion.
- bayesian/figure_question1_variance_contributions.svg: proportional and absolute
  component estimates with 95% credible intervals for each similarity measure.

Hierarchical absolute units are squared atanh(similarity), not raw-score
variance. Outer-fold fit combines common and role-specific fitted effects;
model-fit pair is the residual level, not a demonstrated causal mechanism.
These exploratory Gaussian models leave posterior predictive discrepancies.
Similarity-score variance is not a direct measure of parameter reproducibility.

Numerical summaries, posterior draws, diagnostics and provenance remain in
`data/analyses/mechanistic_variability/appendix/variance/`. See
`code/analyses/MECHANISTIC_VARIABILITY_VARIANCE.md` for the model specification.
Generate these figures using:
`.venv/bin/python code/analyses/mechanistic_variability_appendix.py --sections variance`
""")
