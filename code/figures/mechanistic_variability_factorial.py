"""Exploratory factorial cell means, contrasts and observed-path attribution."""
import numpy as np
from scipy.stats import t

from .model_family_colors import COLOR_BY_FAMILY


def plot_factorial(cells,contrasts,attribution,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[('logit_value_similarity','Logit'),('weight_similarity','Weights'),('hidden_similarity','Hidden state'),('gate_update_similarity','Update gate')]
    effects=['BC','DB_removed','BC:DB_removed','remove_reset_light_minus_GRU','remove_reset_light_minus_GRU:BC','remove_reset_light_minus_GRU:DB_removed','remove_reset_light_minus_GRU:BC:DB_removed','scalar_mono_minus_light','scalar_mono_minus_light:BC','scalar_mono_minus_light:DB_removed','scalar_mono_minus_light:BC:DB_removed']
    labels=['+BC','−DB','BC × DB','−reset','−reset × BC','−reset × DB','−reset × BC × DB','scalar','scalar × BC','scalar × DB','scalar × BC × DB']
    architectures=['GRU','lightGRU','monoGRU']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        for within,scope in [(True,'within'),(False,'across')]:
            fig,axes=plt.subplots(4,3,figsize=(14,14),gridspec_kw={'width_ratios':[1,1.45,.95]},layout='constrained')
            for row,(metric,label) in enumerate(metrics):
                ax=axes[row,0]
                cell=cells[cells.metric.eq(metric)&cells.within_subject.eq(within)]
                for bc,db,color,style in [(0,0,'#555555','-'),(1,0,'#1b6ca8','-'),(0,1,'#d1495b','--'),(1,1,'#6c3d91','--')]:
                    g=cell[cell.BC.eq(bc)&cell.DB_removed.eq(db)].set_index('model_type').reindex(architectures)
                    ax.plot(range(3),g.mean_similarity,color=color,ls=style,marker='o',label=('BC' if bc else 'tanh')+(' −DB' if db else ' +DB'))
                ax.set_xticks(range(3),architectures,rotation=20);ax.set_ylabel(label+' similarity')
                ax=axes[row,1]
                s=contrasts[contrasts.metric.eq(metric)&contrasts.within_subject.eq(within)].set_index('effect').loc[effects]
                y=np.arange(len(effects))[::-1];significant=s.q_BH_contrasts94.lt(.05).to_numpy()
                half=t.ppf(.975,10)*s.standard_error.to_numpy()
                ax.axvline(0,color='.7',lw=.7)
                ax.errorbar(s.mean_contrast,y,xerr=half,fmt='none',color='.5',capsize=2)
                for mask,face in [(significant,'#222222'),(~significant,'white')]:
                    ax.scatter(s.mean_contrast.to_numpy()[mask],y[mask],facecolor=face,edgecolor='#333333',s=25,zorder=3)
                ax.set_yticks(y,labels);ax.set_ylim(-.7,len(effects)-.3)
                ax=axes[row,2]
                a=attribution[attribution.metric.eq(metric)&attribution.within_subject.eq(within)]
                v=a[a.intervention.ne('TOTAL')];total=a[a.intervention.eq('TOTAL')].estimate.iloc[0]
                ax.axhline(0,color='.7',lw=.7)
                ax.bar(range(4),v.estimate,color=['#1b6ca8','#999999','#666666','#d1495b'],width=.65)
                ax.errorbar(range(4),v.estimate,yerr=t.ppf(.975,10)*v.standard_error,fmt='none',ecolor='#222222',capsize=2)
                ax.set_xticks(range(4),['+BC','−reset','scalar','−DB'],rotation=25)
                ax.set_ylabel('Contribution to similarity change');ax.set_title(f'Total change = {total:+.3f}',fontsize=10)
                if row==0:
                    axes[row,0].set_title('Observed configuration means')
                    axes[row,1].set_title('Factorial contrasts, 95% CI\nFilled: BH q < .05 (94 tests)')
            handles,legend=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
            fig.get_layout_engine().set(rect=(0,0,1,.92))
            fig.suptitle(('Within subjects' if within else 'Across subjects')+' · exploratory factorial analysis\nAttribution averages 12 valid paths; reset removal precedes scalar gating',fontsize=13)
            fig.savefig(output/f'figure_factorial_{scope}.svg',facecolor='white')
            plt.close(fig)


def plot_interactions(values,out):
    ARCH=['GRU','lightGRU','monoGRU']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[('weight_similarity','Weights'),('logit_value_similarity','Logit'),('hidden_similarity','Hidden state'),('gate_update_similarity','Update gate')]
    fig,axes=plt.subplots(2,4,figsize=(17,9))
    for r,scope in enumerate([True,False]):
        for col,(metric,label) in enumerate(metrics):
            ax=axes[r,col];data=values[values.within_subject.eq(scope)&values.metric.eq(metric)]
            for arch in ARCH:
                for db in [0,1]:
                    g=data[data.model_type.eq(arch)&data.DB_removed.eq(db)].groupby('BC').mean_similarity.mean()
                    ax.plot([0,1],g.reindex([0,1]),color=COLOR_BY_FAMILY[arch],ls='--' if db else '-',marker='o',label=arch+(' −DB' if db else ' +DB'))
            ax.set_xticks([0,1],['tanh / −BC','ReLU / +BC']);ax.set_title(('Within' if scope else 'Across')+' · '+label);ax.set_ylabel('Mean similarity');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=6,bbox_to_anchor=(.5,.98))
    fig.text(.5,.02,'Cell means over 11 subjects (within) or 55 dependent dyads (across). No independence-based error bars. BC also bundles activation/regularization changes.\n'
             'Reset removal: GRU→lightGRU; scalar update gate: lightGRU→monoGRU. Scalar-with-reset cell is absent; a separate gate×gate interaction is not estimable.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.12,hspace=.32,wspace=.30);fig.savefig(out/'figure_factorial_interactions.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
