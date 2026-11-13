"""Plot functions for the mechanistic-variability study; no data loading or CLI."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from NM_TinyRNN.code.measures.mechanistic_variability import (
    STANDARD_GRU_REFERENCE as REFERENCE, SIMILARITY_METRICS as METRICS)


def save_figure(fig, output, stem):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(Path(output)/f'appendix_{stem}.{extension}', dpi=200,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)

ARCH=['GRU','lightGRU','monoGRU']
CONFIG=[('tanh',True),('tanh',False),('relu',True),('relu',False)]
VIOLIN_METRICS=[('weight_similarity','Parameter cosine'),('hidden_similarity','Hidden-state Pearson r'),
         ('gate_update_similarity','Update-gate Pearson r'),('gate_reset_similarity','Reset-gate Pearson r'),
         ('logit_value_similarity','Readout Pearson r')]


def variant_order(frame):
    return [frame.loc[frame.model_type.eq(a)&frame.nonlinearity.eq(act)&frame.decoder_bias.eq(b),'model_id'].iloc[0]
            for a in ARCH for act,b in CONFIG]


def label(a,act,b):return f'{a}\n{act}\n{ "+DB" if b else "−DB"}'


def violin(ax,values,x,color,width=.7,linewidth=1.2):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values):return
    if np.ptp(values)>1e-10:
        v=ax.violinplot(values,positions=[x],widths=width,showextrema=False,bw_method=.22,points=150)
        body=v['bodies'][0];body.set_facecolor('none');body.set_edgecolor(color);body.set_linewidth(linewidth);body.set_alpha(1)
    q1,med,q3=np.quantile(values,[.25,.5,.75])
    ax.plot([x,x],[q1,q3],color=color,lw=4)
    ax.scatter(x,med,s=16,color='white',edgecolors=color,zorder=4)


def similarity_figures(pairs,selected,subject,out,selection_label="Validation-selected"):
    order=variant_order(selected);meta=selected.drop_duplicates('model_id').set_index('model_id')
    colors=plt.get_cmap('tab20').colors
    summaries=[]
    def draw(ax,metric,scope):
        for i,key in enumerate(order):
            values=pairs.loc[pairs.model_id.eq(key)&pairs.within_subject.eq(scope),metric].dropna().to_numpy()
            c=colors[(i//4)*6+(i%4)]
            if len(values):
                violin(ax,values,i,c)
                means=subject.loc[subject.model_id.eq(key)&subject.within_subject.eq(scope)&subject.metric.eq(metric),'mean_similarity'].dropna().to_numpy()
                ax.scatter(i+np.linspace(-.12,.12,len(means)),means,s=7,c='#555555',alpha=.35,linewidths=0,zorder=5)
            else:ax.text(i,0,'No reset gate',rotation=90,ha='center',va='center',fontsize=7,color='#888888')
        ax.set_xticks(range(12),[label(meta.loc[k].model_type,meta.loc[k].nonlinearity,meta.loc[k].decoder_bias) for k in order],fontsize=7)
        ax.set_xlim(-.6,11.6);ax.set_ylim(-1,1);ax.axhline(1,color='#555555',ls='--',lw=1.5,clip_on=False,zorder=6);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for x in [3.5,7.5]:ax.axvline(x,color='#dddddd',lw=.7)
    fig,axes=plt.subplots(5,2,figsize=(17,16))
    fig.subplots_adjust(left=.065,right=.99,top=.91,bottom=.115,hspace=.75,wspace=.15)
    for r,(metric,title) in enumerate(VIOLIN_METRICS):
        for col,scope in enumerate([False,True]):
            ax=axes[r,col];draw(ax,metric,scope)
            ax.set_title(f'{title} · {"within subject" if scope else "across subjects"}',fontsize=12)
            ax.set_ylabel('Cosine similarity' if metric=='weight_similarity' else 'Pearson r')
            for key in order:
                v=pairs.loc[pairs.model_id.eq(key)&pairs.within_subject.eq(scope),metric].dropna()
                summaries.append(dict(model_id=key,within_subject=scope,metric=metric,n_pairs=len(v),
                                      q25=v.quantile(.25),median=v.median(),q75=v.quantile(.75)))
        # Also export each metric as a separate two-panel figure.
        individual,aa=plt.subplots(1,2,figsize=(16,5))
        individual.subplots_adjust(left=.06,right=.99,top=.79,bottom=.28,wspace=.14)
        for ax,scope in zip(aa,[False,True]):
            draw(ax,metric,scope);ax.set_title('Within subject' if scope else 'Across subjects')
            ax.set_ylabel('Similarity')
        individual.suptitle(f'Two-unit models: {title}\n{selection_label} · all 12 configurations',fontsize=14)
        individual.text(.5,.03,'Outline = fold-pair distribution; bar = IQR; white dot = median. Grey dots: 11 subject means (within), 55 subject-pair means (across).\n'
            '5500 across-subject or 495 within-subject pairs per configuration; overlapping pairs, descriptive distributions.',ha='center',fontsize=9)
        save_figure(individual,out,f'baseline_two_unit_{metric}_violins')
        if metric=='logit_value_similarity':
            zoom,zz=plt.subplots(1,2,figsize=(16,5))
            zoom.subplots_adjust(left=.06,right=.99,top=.79,bottom=.28,wspace=.14)
            low=max(-1.05,float(np.floor((pairs[metric].min()-.01)*20)/20))
            for ax,scope in zip(zz,[False,True]):
                draw(ax,metric,scope);ax.set_ylim(low,1.005)
                ax.set_title('Within subject' if scope else 'Across subjects');ax.set_ylabel('Pearson r (zoomed)')
            zoom.suptitle(f'Two-unit readout similarity: zoomed scale · {selection_label}',fontsize=14)
            zoom.text(.5,.03,'Same fold-pair distributions and subject summaries; shared zoomed y-axis. Flattened readout correlations.',ha='center',fontsize=9)
            save_figure(zoom,out,'baseline_two_unit_readout_violins_zoom')

    fig.suptitle('Two-unit mechanistic similarity across every model configuration\n'
        f'{selection_label} outer winners · 11 subjects · subject-specific trials (reference = subject B)',fontsize=16,y=.98)
    fig.text(.5,.025,'Outline = fold-pair distribution; bar = IQR; white dot = median. Grey dots: 11 subject means (within), 55 subject-pair means (across).\n'
        'Across: 5500 pairs per configuration; within: 495 pairs. Pairs overlap; no independence-based confidence intervals.\n'
        'Hidden scores average aligned unit correlations; gate/readout scores use flattened correlations. +DB / −DB = decoder bias present / absent.',ha='center',fontsize=10)
    save_figure(fig,out,'baseline_two_unit_similarity_violins_all_configs')
    pd.DataFrame(summaries).to_csv(out/'appendix_baseline_two_unit_violin_summary.csv',index=False)


def reference_tables(frame, out, selection_label='Minimum train + validation NLL'):
    """Separate and side-by-side tables using already computed reference contrasts."""
    frame = frame.fillna({'significance': ''})
    order=[f'2_unit_{a}_{act}_unipolar_forced'+('' if b else '_ndb')
           for a in ['GRU','lightGRU','monoGRU'] for act,b in [('tanh',True),('tanh',False),('relu',True),('relu',False)]]
    labels=[f'{a} · {act} · bias {"on" if b else "off"}' for a in ['GRU','lightGRU','monoGRU']
            for act,b in [('tanh',True),('tanh',False),('relu',True),('relu',False)]]
    headers=['Parameters','Hidden states','Update gate','Reset gate','Readout']
    cmap=plt.get_cmap('RdBu_r');norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=3)
    panels={}
    for scope in [True,False]:
        title='Within subject' if scope else 'Across subjects'
        subset=frame[frame.within_subject.eq(scope)].set_index(['model_id','metric'])
        fig,ax=plt.subplots(figsize=(14,11));fig.subplots_adjust(left=.23,right=.97,top=.85,bottom=.18)
        ax.set(xlim=(-.5,4.5),ylim=(11.5,-.5))
        ax.set_xticks(range(5),headers,fontsize=11);ax.xaxis.tick_top()
        ax.set_yticks(range(12),labels,fontsize=10)
        panel_cells=[]
        markdown=[f'# Two-unit similarity: {title}', '',f'Selection: {selection_label}. One model per subject/configuration/outer fold.', '', '| Configuration | '+' | '.join(headers)+' |','|---|'+'---|'*5]
        for i,key in enumerate(order):
            cells=[]
            for j,metric in enumerate(METRICS):
                row=subset.loc[(key,metric)]
                if not row.available:text='N/A';color='#eeeeee';cell='N/A'
                elif key==REFERENCE:
                    text=f'{row.mean_similarity:.3f}\nreference';color='#e5e7eb';cell=f'{row.mean_similarity:.3f} (reference)'
                else:
                    text=f'{row.mean_similarity:.3f}\nΔ {row.delta_similarity:+.3f}\ndz {row.cohens_dz:+.2f}{row.significance}'
                    rgba=cmap(norm(np.clip(row.cohens_dz,-3,3)));color=tuple(.55+.45*np.array(rgba[:3]))
                    cell=f'{row.mean_similarity:.3f}; Δ {row.delta_similarity:+.3f}; dz {row.cohens_dz:+.2f}{row.significance}'
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=1.5))
                if row.available and key!=REFERENCE and row.p_holm<.05:
                    ax.add_patch(plt.Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor='#222222',linewidth=2))
                ax.text(j,i,text,ha='center',va='center',fontsize=9,fontweight='bold' if row.available and key!=REFERENCE and row.p_holm<.05 else 'normal')
                panel_cells.append(dict(i=i,j=j,text=text,color=color,significant=bool(row.available and key!=REFERENCE and row.p_holm<.05)))
                cells.append(cell)
            markdown.append('| '+labels[i]+' | '+' | '.join(cells)+' |')
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.suptitle(f'Two-unit similarity relative to standard GRU · {title}\n{selection_label} · Reference: GRU, tanh, decoder bias on',fontsize=15,y=.97)
        fig.text(.5,.115,'Cell: mean similarity / raw difference Δ / descriptive paired Cohen dz. Blue = lower; red = higher similarity.\n'
            'Black outline and stars: Holm-adjusted p < .05 (*), .01 (**), .001 (***) across all 94 comparisons.',ha='center',fontsize=10)
        fig.text(.5,.055,'Subjects = 11. Within: paired subject means. Across: descriptive dz uses incident-dyad subject means;\n'
            'significance uses leave-one-subject-out jackknife (removes all incident dyads), approximate t(10).\n'
            'Exploratory inference; fold pairs are not independent. Higher similarity does not imply better predictive performance.',ha='center',fontsize=9,color='#555555')
        panels[scope]=panel_cells
        suffix='within' if scope else 'across'
        save_figure(fig,out,f'baseline_similarity_vs_standard_gru_{suffix}')
        markdown+=['','Means are raw correlations/cosines. Δ = model − reference. dz = mean paired subject-level difference / its SD.',
                   'Stars use Holm correction over all 94 non-reference comparisons. Across-subject inference is approximate node-jackknife t(10); descriptive dz does not supply its p-value.',
                   'Only 11 subjects; treat significance as exploratory. Reset gate is absent in lightGRU/monoGRU.']
        (out/f'appendix_baseline_similarity_vs_standard_gru_{suffix}.md').write_text('\n'.join(markdown)+'\n')
    combined,axes=plt.subplots(1,2,figsize=(23,11))
    combined.subplots_adjust(left=.155,right=.985,top=.84,bottom=.19,wspace=.07)
    for ax,scope in zip(axes,[False,True]):
        ax.set(xlim=(-.5,4.5),ylim=(11.5,-.5))
        ax.set_xticks(range(5),headers,fontsize=10);ax.xaxis.tick_top()
        ax.set_yticks(range(12),labels if not scope else ['']*12,fontsize=10)
        ax.set_title('Across subjects' if not scope else 'Within subject',fontsize=15,fontweight='bold',pad=36)
        for item in panels[scope]:
            i,j=item['i'],item['j']
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=item['color'],edgecolor='white',linewidth=1.5))
            if item['significant']:
                ax.add_patch(plt.Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor='#222222',linewidth=2))
            ax.text(j,i,item['text'],ha='center',va='center',fontsize=9,fontweight='bold' if item['significant'] else 'normal')
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    combined.suptitle(f'Two-unit similarity relative to standard GRU\n{selection_label} · Reference: GRU, tanh, decoder bias on',fontsize=17,y=.98)
    combined.text(.5,.115,'Cell: mean similarity / raw difference Δ / descriptive paired Cohen dz. Identical row order and colour scale in both panels.\n'
        'Blue = lower similarity; red = higher. Black outlines/stars: Holm-adjusted p < .05 (*), .01 (**), .001 (***) across all 94 tests.',ha='center',fontsize=11)
    combined.text(.5,.055,'11 subjects. Within: paired subject means. Across: descriptive dz uses incident-dyad subject means; inference uses subject-deletion jackknife, approximate t(10).\n'
        'Exploratory inference; fold pairs are not independent. Higher similarity does not imply better predictive performance.',ha='center',fontsize=10,color='#555555')
    save_figure(combined,out,'baseline_similarity_vs_standard_gru_side_by_side')
    frame.to_csv(out/'appendix_baseline_similarity_vs_standard_gru.csv',index=False)



def main_gru_monogru(pairs, selected, contrasts, output):
    """Notebook-style three-component violins and a matching numerical summary.

    Each violin contains individual outer-winner model pairs. Inferential columns
    retain the study's subject-level contrasts and global Holm correction.
    """
    from matplotlib.patches import Patch
    import seaborn as sns
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = selected.drop_duplicates('model_id')
    chosen = metadata.loc[
        metadata.hidden_size.eq(2) & (
            (metadata.model_type.eq('GRU') & metadata.nonlinearity.eq('tanh') & metadata.decoder_bias.eq(True)) |
            (metadata.model_type.eq('monoGRU') & metadata.nonlinearity.eq('relu') & metadata.decoder_bias.eq(False)))]
    if len(chosen) != 2:
        raise ValueError('Expected exactly the standard two-unit GRU and monoGRU configurations')
    labels = dict(zip(chosen.model_id, chosen.model_type.map({'GRU':'GRU', 'monoGRU':'monoGRU+BC-DB'})))
    order = ['GRU', 'monoGRU+BC-DB']
    components = {'weight_similarity':'Weights', 'logit_value_similarity':'Logit',
                  'hidden_similarity':'Hidden state'}
    frame = pairs[pairs.model_id.isin(labels)].copy()
    frame['model_label'] = frame.model_id.map(labels)
    colors = ['#a1c9f4', '#ffb482']
    rows = []
    with plt.rc_context():
        sns.set_context('poster')
        fig, axes = plt.subplots(3, 2, figsize=(14, 20), sharey='row')
        for r, (metric, component) in enumerate(components.items()):
            for c, scope in enumerate([False, True]):
                ax = axes[r, c]
                for x, model in enumerate(order):
                    group = frame[frame.model_label.eq(model) & frame.within_subject.eq(scope)]
                    values = group[metric].dropna().to_numpy()
                    violin(ax, values, x, colors[x], linewidth=4.0)
                    rows.append(dict(model_id=group.model_id.iloc[0], model=model,
                        within_subject=scope, component=component, metric=metric,
                        n_model_pairs=len(values), mean=np.mean(values), sd=np.std(values, ddof=1),
                        median=np.median(values), q25=np.quantile(values,.25), q75=np.quantile(values,.75)))
                ax.set_xticks([0,1], order, rotation=25, ha='right', rotation_mode='anchor')
                ax.set_xlim(-.6,1.6)
                ax.set_ylim(-1, 1)
                ax.set_yticks([-1, -.5, 0, .5, 1])
                ax.axhline(1, color='#555555', linestyle='--', linewidth=2,
                           clip_on=False, zorder=6)
                ax.set_xlabel('Model ID' if r == 2 else ''); ax.set_ylabel('Similarity')
                title = (('Across Subjects' if not scope else 'Within Subject')+'\n\n' if r == 0 else '')+component
                ax.set_title(title, pad=15)
                ax.spines[['top','right']].set_visible(False)
        fig.legend(handles=[Patch(facecolor='none',edgecolor=color,label=model) for model,color in zip(order,colors)],
                   loc='upper center', bbox_to_anchor=(.5,.99), ncol=2, title='Model ID', frameon=False)
        fig.subplots_adjust(top=.85,bottom=.12,hspace=.90,wspace=.30)
        fig.savefig(output/'figure_GRU_monoGRU_mech_var.svg',
                    facecolor='white', bbox_inches='tight')
        for extension in ['png', 'pdf']:
            (output/f'figure_GRU_monoGRU_mech_var.{extension}').unlink(missing_ok=True)
        plt.close(fig)
    table = pd.DataFrame(rows).merge(contrasts, on=['model_id','within_subject','metric'],
                                      how='left', validate='one_to_one')
    table.to_csv(output/'table_GRU_monoGRU_mech_var.csv',index=False)
    return table


def appendix_all_models(pairs, selected, output):
    """A4 portrait page: all twelve two-unit configurations and five scores.

    Violin observations are outer-winner model pairs, matching the main figure.
    Architectures have consistent colours; activation and decoder bias appear
    on the shared bottom axis. Missing gates are explicitly marked as absent.
    """
    import seaborn as sns
    from matplotlib.patches import Patch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = selected.loc[selected.hidden_size.eq(2)]
    order = variant_order(selected)
    meta = selected.drop_duplicates('model_id').set_index('model_id')
    colors = {'GRU':'#85b5eb', 'lightGRU':'#87bd9b', 'monoGRU':'#f6a36c'}
    metrics = [('weight_similarity','Weights'), ('logit_value_similarity','Logit'),
               ('hidden_similarity','Hidden state'), ('gate_update_similarity','Update gate'),
               ('gate_reset_similarity','Reset gate')]
    ticks = [('tanh' if act == 'tanh' else 'ReLU')+' '+('+DB' if bias else '−DB')
             for _ in ARCH for act, bias in CONFIG]
    with plt.rc_context():
        sns.set_context('paper', font_scale=1.1)
        fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69), sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.98, top=.86, bottom=.135, hspace=.49, wspace=.15)
        for r, (metric, title) in enumerate(metrics):
            for c, scope in enumerate([False, True]):
                ax = axes[r, c]
                for i, key in enumerate(order):
                    values = pairs.loc[pairs.model_id.eq(key) & pairs.within_subject.eq(scope),metric].dropna().to_numpy()
                    if len(values):
                        violin(ax, values, i, colors[meta.loc[key,'model_type']], width=.79, linewidth=1.6)
                for x in [3.5, 7.5]:
                    ax.axvline(x, color='#d8d8d8', linewidth=.6, zorder=0)
                if metric == 'gate_reset_similarity':
                    for x in [5.5, 9.5]:
                        ax.text(x, 0, 'No reset gate', ha='center', va='center', fontsize=7, color='#666666')
                ax.axhline(1, color='#555555', linestyle='--', linewidth=1, clip_on=False, zorder=6)
                ax.axhline(0, color='#ededed', linewidth=.6, zorder=0)
                ax.set_ylim(-1, 1)
                ax.set_yticks([-1, 0, 1])
                ax.set_xlim(-.6, 11.6)
                ax.set_title(f'{chr(65+r*2+c)}  {title}', loc='left', fontsize=10, pad=7)
                ax.set_ylabel('Similarity' if c == 0 else '', fontsize=9)
                ax.set_xticks(range(12), ticks, rotation=65, ha='right', rotation_mode='anchor', fontsize=6.5)
                ax.tick_params(axis='both', width=.9, length=3, labelsize=8 if r != 4 else 6.5)
                ax.spines[['top', 'right']].set_visible(False)
                if r == 0:
                    ax.text(.5, 1.36, 'Within subject' if scope else 'Across subjects',
                            transform=ax.transAxes, ha='center', va='bottom', fontsize=11, fontweight='bold')
                if r == 4:
                    for x, arch in zip([1.5, 5.5, 9.5], ARCH):
                        ax.text(x, -.53, arch, transform=ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=8, fontweight='bold', color=colors[arch])
        fig.suptitle('Mechanistic variability across model configurations', fontsize=13, fontweight='bold', y=.985)
        fig.text(.5, .958, 'Two-unit models · all ten outer-fold winners per subject', ha='center', fontsize=9)
        fig.legend(handles=[Patch(facecolor='none', edgecolor=colors[arch], linewidth=1.6, label=arch) for arch in ARCH],
                   loc='upper center', bbox_to_anchor=(.5,.947), ncol=3, frameon=False, fontsize=9)
        fig.text(.5, .025,
                 'Outline: fold-pair distribution; bar: interquartile range; white dot: median. Dashed line: similarity ceiling.\n'
                 'tanh / ReLU: activation; +DB / −DB: decoder bias present / absent. Models selected by train + validation NLL.\n'
                 'Across: 5,500 model pairs per configuration; within: 495. Pairs share models and subjects.\n'
                 'Hidden: mean aligned unit correlations; logit and gates: flattened correlations; weights: parameter cosine.',
                 ha='center', va='bottom', fontsize=7, linespacing=1.5)
        path = output/'figure_all_models_mech_var.svg'
        fig.savefig(path, facecolor='white')
        plt.close(fig)
    return path


def appendix_cohens_d(contrasts, selected, output):
    """Matched subject-level dz panels using the study's existing contrasts.

    Across-subject dz is descriptive: its subject summaries share dyads. Stars
    retain the existing subject-deletion inference and global Holm correction.
    """
    from matplotlib.cm import ScalarMappable
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = selected.loc[selected.hidden_size.eq(2)].drop_duplicates('model_id').set_index('model_id')
    order = variant_order(selected.loc[selected.hidden_size.eq(2)])
    metrics = ['weight_similarity', 'logit_value_similarity', 'hidden_similarity',
               'gate_update_similarity', 'gate_reset_similarity']
    names = ['Weights', 'Logit', 'Hidden\nstate', 'Update\ngate', 'Reset\ngate']
    frame = contrasts.loc[contrasts.model_id.isin(order)].copy().fillna({'significance':''})
    finite = frame.cohens_dz[np.isfinite(frame.cohens_dz)]
    limit = max(1., float(np.ceil(finite.abs().max())))
    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    labels = [f'{meta.loc[key,"model_type"]} · {meta.loc[key,"nonlinearity"]} · '
              f'{"+DB" if meta.loc[key,"decoder_bias"] else "−DB"}' for key in order]
    with plt.rc_context({'font.size':9}):
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(left=.205, right=.92, top=.79, bottom=.24, wspace=.08)
        for ax, scope in zip(axes, [False, True]):
            subset = frame.loc[frame.within_subject.eq(scope)].set_index(['model_id','metric'])
            for i, key in enumerate(order):
                for j, metric in enumerate(metrics):
                    row = subset.loc[(key, metric)]
                    significant = bool(row.available and key != REFERENCE and row.p_holm < .05)
                    if not row.available:
                        text, color, text_color = 'N/A', '#eeeeee', '#777777'
                    elif key == REFERENCE:
                        text, color, text_color = 'ref.', '#dfe3e8', '#444444'
                    elif not np.isfinite(row.cohens_dz):
                        text, color, text_color = 'N/A', '#eeeeee', '#777777'
                    else:
                        text = f'{row.cohens_dz:+.2f}{row.significance}'
                        color = cmap(norm(row.cohens_dz))
                        text_color = 'white' if abs(row.cohens_dz) > limit*.55 else '#111111'
                    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=color,
                                              edgecolor='white',linewidth=1))
                    if significant:
                        ax.add_patch(plt.Rectangle((j-.48,i-.48),.96,.96,fill=False,
                                                  edgecolor='#111111',linewidth=1.6))
                    ax.text(j,i,text,ha='center',va='center',fontsize=8.5,color=text_color,
                            fontweight='bold' if significant else 'normal')
            for y in [3.5, 7.5]:
                ax.axhline(y, color='#666666', linewidth=1)
            ax.set(xlim=(-.5,4.5),ylim=(11.5,-.5))
            ax.set_xticks(range(5),names,fontsize=9)
            ax.xaxis.tick_top()
            ax.set_yticks(range(12),labels if not scope else ['']*12,fontsize=8.5)
            if not scope:
                ax.get_yticklabels()[order.index(REFERENCE)].set_fontweight('bold')
            ax.tick_params(length=0, pad=7)
            ax.set_title('Within subject' if scope else 'Across subjects',pad=38,
                         fontsize=12,fontweight='bold')
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle('Mechanistic similarity relative to baseline GRU',fontsize=15,fontweight='bold',y=.975)
        fig.text(.56,.93,'Two-unit models · reference: GRU, tanh, decoder bias on',ha='center',fontsize=10)
        color_axis = fig.add_axes([.94,.25,.015,.53])
        colorbar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=color_axis)
        colorbar.set_label('Paired Cohen’s $d_z$',fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        n_tests = int(frame.loc[frame.model_id.ne(REFERENCE),'p_holm'].notna().sum())
        fig.text(.56,.145,'Blue: lower similarity than baseline; red: higher similarity. +DB / −DB: decoder bias present / absent.\n'
                 f'Black outlines and stars: Holm-adjusted p < .05 (*), .01 (**), .001 (***) across {n_tests} comparisons. N/A: no gate.',
                 ha='center',va='center',fontsize=8,linespacing=1.5)
        fig.text(.56,.06,'Within: paired differences between 11 subject means, divided by their SD. Across: descriptive $d_z$ uses\n'
                 '11 incident-dyad subject means; those means share dyads. P-values use subject-deletion jackknife, approximate t(10).\n'
                 'Models selected by train + validation NLL. Fold pairs are not independent; inference is exploratory.',
                 ha='center',va='center',fontsize=8,color='#555555',linespacing=1.5)
        path = output/'figure_mech_var_cohens_d.svg'
        fig.savefig(path,facecolor='white')
        plt.close(fig)
    frame['model_label'] = frame.model_id.map(dict(zip(order,labels)))
    frame['baseline_model_id'] = REFERENCE
    frame.to_csv(output/'table_mech_var_cohens_d.csv',index=False)
    return path
