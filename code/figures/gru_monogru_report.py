"""Compact main table, supplementary tables, and paired similarity plots."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .model_family_colors import COLOR_BY_FAMILY


def table_figure(cells,headers,path,title,footnote,widths=None,figsize=(15,4)):
    with plt.rc_context({'font.family':'serif','svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=figsize);ax.axis('off')
        table=ax.table(cellText=cells,colLabels=headers,cellLoc='center',loc='center',colWidths=widths)
        table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2.0)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd');cell.set_linewidth(.5)
            if r==0:cell.set_text_props(weight='bold');cell.set_facecolor('#f0f0f0')
        ax.set_title(title,fontsize=15,pad=12)
        fig.text(.5,.04,footnote,ha='center',fontsize=9)
        fig.subplots_adjust(left=.045,right=.955,top=.84,bottom=.23)
        fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)


def paired_plot(data,summary,path,label):
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'width_ratios':[1,1.5]})
    for r,scope in enumerate([True,False]):
        d=data[data.within_subject.eq(scope)].sort_values(['subject_A','subject_B']);s=summary[summary.within_subject.eq(scope)].iloc[0]
        ax=axes[r,0]
        for row in d.itertuples():
            ax.plot([0,1],[row.gru,row.mono],color='#888888',alpha=.65 if scope else .25,lw=1)
            ax.scatter([0,1],[row.gru,row.mono],c=[COLOR_BY_FAMILY['GRU'],COLOR_BY_FAMILY['monoGRU']],
                       alpha=.65 if scope else .25,s=9,zorder=3)
        ax.set_xticks([0,1],['GRU','monoGRU']);ax.set_ylabel('Similarity');ax.set_ylim(-1,1);ax.set_xlim(-.2,1.2)
        ax.set_title(('Within: 11 subjects' if scope else 'Across: 55 subject pairs'))
        ax=axes[r,1];y=np.arange(len(d));ax.scatter(d.difference,y,s=27 if scope else 12,color='#377eb8',alpha=.9 if scope else .6)
        ax.axvline(0,color='.5',lw=1);ax.axvline(s.median_difference,color='#e58130',ls='--',lw=1.5,label='Median difference')
        ax.set_yticks(y,d.subject_A if scope else ['']*len(d));ax.invert_yaxis();ax.set_xlabel('Paired difference: monoGRU − GRU')
        ax.set_title(f'Median Δ = {s.median_difference:+.4f}'+(f' · sign BH q = {s.q_main_bh8:.4f}' if scope else f' · mean-test jackknife BH q = {s.q_main_bh8:.4f}'))
        ax.grid(axis='x',alpha=.2)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle(label+' similarity: paired model comparisons',fontsize=16)
    fig.text(.5,.025,'Each within-subject point/line summarizes 45 outer-fold model pairs; each across-subject point/line summarizes 100.\n'
             'BH spans eight focused comparisons. Across jackknife tests the mean, not the median; inference is approximate. Orange dashed line: descriptive median.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.12,hspace=.35,wspace=.32);fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)


def table_one(summary,path):
    """Minimal table with independent within tests and dyadic across inference."""
    with plt.rc_context({'font.family':'serif','mathtext.fontset':'dejavuserif','svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(13,3.8));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        x=[.025,.235,.355,.475,.675,.795,.915]
        ax.text(.5,.97,'Table 1. Similarity of GRU and monoGRU models',ha='center',va='top',fontsize=16)
        ax.text(.355,.78,'Within subject',ha='center',fontweight='bold',fontsize=14)
        ax.text(.795,.78,'Across subjects',ha='center',fontweight='bold',fontsize=14)
        for i,(xx,label) in enumerate(zip(x,['Component','GRU','monoGRU','Median Δ','GRU','monoGRU','Median Δ'])):
            ax.text(xx,.68,label,ha='left' if i==0 else 'center',fontweight='bold',fontsize=12)
        ax.plot([.015,.99],[.63,.63],color='black',lw=.8)
        def stars(q):return '***' if q<.001 else '**' if q<.01 else '*' if q<.05 else ''
        for i,(_,group) in enumerate(summary.groupby('metric',sort=False)):
            w=group[group.within_subject].iloc[0];a=group[~group.within_subject].iloc[0];y=.55-i*.10
            ws,acs=stars(w.q_main_bh8),stars(a.q_main_bh8)
            cells=[w.component,f'{w.gru_mean:.3f}',f'{w.mono_mean:.3f}',f'{w.median_difference:+.4f}'+(r'$^{'+ws+'}$' if ws else ''),f'{a.gru_mean:.3f}',f'{a.mono_mean:.3f}'+(r'$^{'+acs+'}$' if acs else ''),f'{a.median_difference:+.4f}']
            for j,(xx,text) in enumerate(zip(x,cells)):
                bold=(j==3 and bool(ws)) or (j==5 and bool(acs))
                ax.text(xx,y,text,ha='left' if j==0 else 'center',va='center',fontsize=12,fontweight='bold' if bold else 'normal')
        ax.plot([.015,.99],[.18,.18],color='black',lw=.8)
        ax.text(.015,.135,'GRU/monoGRU columns: mean similarity over 11 subject summaries (within) or 55 subject-pair summaries (across). Median Δ = median paired monoGRU − GRU difference.\n'
            'Within stars: exact two-sided sign test on 11 independent subject differences. Across stars: mean-difference test using subject-deletion jackknife SE and approximate t(10).\n'
            'BH across all eight comparisons (* q < .05, ** < .01, *** < .001). Across stars annotate the mean, not the median. Block-reset winners; nested_DA_S32_B8.',ha='left',va='top',fontsize=8.2,linespacing=1.4)
        fig.subplots_adjust(left=.025,right=.98,top=.98,bottom=.02);fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)
