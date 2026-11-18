"""Export aligned monoGRU +BC-DB winners, regenerated subject trials and parameter mean.

Run after monoGRU_alignment.py. Sources remain unchanged; exported histories are
continuous whole-subject rollouts, as in the original examples.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from NM_TinyRNN.code.analyses import monoGRU_alignment as alignment
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.models import datasets as ds
from NM_TinyRNN.code.models.save_data import get_split_labels


def load_subject_data(subject, row):
    """Read raw session trial files through the native dataset encoder."""
    previous=ds.DATA_PATH
    ds.DATA_PATH=ROOT/'data'
    try:
        dataset=ds.AB_Dataset(ROOT/'data/AB_behaviour'/subject, subject_ids=[subject],
            sequence_length=int(row.sequence_length), input_encoding=row.input_encoding,
            input_forced_choice=bool(row.input_forced_choice),device=torch.device('cpu'))
    finally:
        ds.DATA_PATH=previous
    return dataset.subject_df.reset_index(drop=True)


def regenerate_trials(model, subject_data, saved):
    """Re-run the shared mechanistic rollout; retain exact original split masks."""
    identity=['session_folder_name','session_trial_idx','forced_choice','choice','outcome',
              'good_poke','sequence_block_idx','block_trial_idx']
    pd.testing.assert_frame_equal(subject_data[identity],saved[identity],check_dtype=False)
    splits={}
    for label in ('train','val','eval'):
        block_ids=saved.loc[saved.split.eq(label),'sequence_block_idx']
        if block_ids.isna().any():raise ValueError('A split labels a trial outside a complete block')
        splits[label]=sorted(block_ids.astype(int).unique().tolist())
    labels=get_split_labels(subject_data,splits)
    np.testing.assert_array_equal(labels,saved.split.to_numpy())
    raw=torch.tensor(subject_data[['forced_choice','choice','outcome']].to_numpy(),dtype=torch.float32)[None]
    inputs=ds.input_encoder(raw,model.input_encoding,model.input_forced_choice)
    activations=mv._run_model_on_trials(model,subject_data,inputs)
    out=subject_data.copy()
    for col in activations:out[col]=activations[col].to_numpy()
    logits=out.logit_value.to_numpy()
    # logit_value is log(p(choice 0)/p(choice 1)); stable two-class softmax.
    out['prob_A']=np.exp(-np.logaddexp(0.,-logits))
    out['prob_B']=np.exp(-np.logaddexp(0.,logits))
    out['logit_past']=np.r_[np.nan,logits[:-1]]
    out['logit_change']=np.r_[np.nan,np.diff(logits)]
    names=np.array(['A1,R=0','A1,R=1','A2,R=0','A2,R=1'])
    out['trial_type']=names[2*out.choice.to_numpy().astype(int)+out.outcome.to_numpy().astype(int)]
    out['split']=labels
    # Native examples keep activation columns first, then trial columns.
    columns=[c for c in saved if c in out]+[c for c in out if c not in saved]
    return out[columns]


def average_parameters(models, rows):
    """Equal subject means after averaging ten outer folds within each subject."""
    states=[m.state_dict() for m in models]
    subject_states=[]
    for subject in sorted(rows.subject_id.unique()):
        indices=np.flatnonzero(rows.subject_id.eq(subject).to_numpy())
        subject_states.append({k:torch.stack([states[i][k].double() for i in indices]).mean(0) for k in states[0]})
    mean={k:torch.stack([s[k] for s in subject_states]).mean(0) for k in states[0]}
    averaged=deepcopy(models[0]);averaged.load_state_dict(mean)
    records=[]
    for name,parameter in averaged.named_parameters():
        values=torch.stack([s[name] for s in subject_states]).numpy()
        fit_values=torch.stack([s[name].double() for s in states]).numpy()
        for index in np.ndindex(parameter.shape):
            x=values[(slice(None),)+index]
            records.append(dict(parameter=name,index=json.dumps(index),mean=float(x.mean()),
                subject_sd=float(x.std(ddof=1)),subject_sem=float(x.std(ddof=1)/np.sqrt(len(x))),
                fit_sd=float(fit_values[(slice(None),)+index].std(ddof=1)),
                n_subjects=len(subject_states),n_fits=len(states)))
    return averaged,pd.DataFrame(records)


def parameter_latex(table):
    """Column-vector equations from stored parameter means, two significant figures."""
    arrays={}
    for name,group in table.groupby('parameter'):
        entries=[(tuple(json.loads(r.index)),r.mean) for r in group.itertuples()]
        ndim=len(entries[0][0])
        shape=tuple(max(index[j] for index,value in entries)+1 for j in range(ndim))
        array=np.zeros(shape)
        for index,value in entries:array[index]=value
        arrays[name]=array
    def matrix(array):
        a=np.atleast_2d(array)
        return r'\begin{bmatrix}'+r' \\ '.join(' & '.join(f'{x:.2g}' for x in row) for row in a)+r'\end{bmatrix}'
    x=r'\begin{bmatrix}c_t \\ a_t \\ o_t\end{bmatrix}'
    hp=r'\begin{bmatrix}h_{1,t-1} \\ h_{2,t-1}\end{bmatrix}'
    ht=r'\begin{bmatrix}h_{1,t} \\ h_{2,t}\end{bmatrix}'
    return '\n'.join([
        '% Mean of 110 reference-fitted monoGRU +BC-DB models, equally weighted by subject.',
        r'\begin{align*}',
        r'\mathbf{n}_t &= \operatorname{ReLU}\!\left('+matrix(arrays['rnn.W_ih'].T)+x+' + '+matrix(arrays['rnn.W_hh'].T)+hp+' + '+matrix(arrays['rnn.bias_h'][:,None])+r'\right), \\',
        r'z_t &= \sigma\!\left('+matrix(arrays['rnn.W_iz'].T)+x+' + '+matrix(arrays['rnn.W_hz'].T)+hp+' + '+f"{float(arrays['rnn.bias_z']):.2g}"+r'\right), \\',
        r'\mathbf{h}_t &= (1-z_t)\mathbf{n}_t + z_t\mathbf{h}_{t-1}, \\',
        r'\hat{\mathbf{a}}_{t+1} &= \operatorname{Softmax}\!\left('+matrix(arrays['decoder.weight'])+ht+r'\right).',
        r'\end{align*}', ''])


def equations():
    return [
        dict(lhs='n_t',activation=r'\mathrm{ReLU}',terms=[
            dict(param='rnn.W_ih',transpose=True,vector_labels=['c_t','a_t','o_t']),
            dict(param='rnn.W_hh',transpose=True,vector_labels=['h_{1,t-1}','h_{2,t-1}']),dict(param='rnn.bias_h')]),
        dict(lhs='z_t',activation=r'\sigma',terms=[
            dict(param='rnn.W_iz',transpose=True,vector_labels=['c_t','a_t','o_t']),
            dict(param='rnn.W_hz',transpose=True,vector_labels=['h_{1,t-1}','h_{2,t-1}']),dict(param='rnn.bias_z')]),
        dict(lhs=r'\hat{a}_{t+1}',activation='Softmax',terms=[
            dict(param='decoder.weight',vector_labels=['h_{1,t}','h_{2,t}'])])]


def export_subject(subject, group, out, source_manifest):
    """One process per subject; all ten folds reuse the same raw subject data."""
    torch.set_num_threads(1)
    artifacts=[];audits=[];data_sources={};exports=[]
    subject_data=load_subject_data(subject,group.iloc[0])
    session_names=subject_data.session_folder_name.unique()
    for session in session_names:
        file=ROOT/'data/AB_behaviour'/session/'trials.htsv'
        data_sources[str(file.relative_to(ROOT))]=mv.sha256(file)
    for r in group.itertuples():
        model=alignment.load_model(r.aligned_model_path)
        saved=pd.read_csv(alignment.resolve(r.trials_data_path),sep='\t')
        regenerated=regenerate_trials(model,subject_data,saved)
        perm=np.array(json.loads(r.perm));scales=np.array(json.loads(r.scales))
        expected=saved[['hidden_1','hidden_2']].to_numpy()[:,perm]*scales
        np.testing.assert_allclose(regenerated[['hidden_1','hidden_2']],expected,atol=5e-5,rtol=2e-4)
        for col in ['logit_value','prob_A','prob_B','gate_update_1']:
            np.testing.assert_allclose(regenerated[col],saved[col],atol=5e-5,rtol=2e-4)
        info=json.loads(alignment.resolve(r.info_path).read_text())
        info['alignment']=dict(reference_model='balanced_population_medoid_model.pickle',
            source_run=source_manifest['folder'],source_model=str(alignment.resolve(r.model_pickle_path)),
            perm=perm.tolist(),scales=scales.tolist(),label_rule=source_manifest['label_rule'],
            original_training_metrics='Retained from source; no retraining. Scale-dependent penalties have not been recomputed.')
        folder=out/subject/'monoGRU'/f'outer_fold_{r.outer_loop_n}';folder.mkdir(parents=True,exist_ok=True)
        info_path=folder/f'{r.model_id}_info.json';model_path=folder/f'{r.model_id}_model.pickle'
        trial_path=folder/f'{r.model_id}_trials_data.htsv'
        info_path.write_text(json.dumps(info,indent=2)+'\n')
        with model_path.open('wb') as f:pickle.dump(model,f)
        regenerated.to_csv(trial_path,sep='\t',index=False)
        # Verify serialization, not only in-memory objects.
        reread=pd.read_csv(trial_path,sep='\t')
        np.testing.assert_array_equal(reread[['hidden_1','hidden_2','prob_A','prob_B']].to_numpy(dtype=np.float32),regenerated[['hidden_1','hidden_2','prob_A','prob_B']].to_numpy(dtype=np.float32))
        for path in [info_path,model_path,trial_path]:artifacts.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=mv.sha256(path)))
        entry=group.loc[r.Index].to_dict()
        entry.update(example_info_path=str(info_path),example_model_pickle_path=str(model_path),example_trials_data_path=str(trial_path))
        exports.append(entry)
        audits.append(dict(subject_id=subject,outer_loop_n=r.outer_loop_n,n_trials=len(saved),
            max_hidden_error=float(np.abs(regenerated[['hidden_1','hidden_2']].to_numpy()-expected).max()),
            max_logit_error=float(np.abs(regenerated.logit_value-saved.logit_value).max()),
            max_probability_error=float(np.abs(regenerated.prob_A-saved.prob_A).max()),
            max_gate_error=float(np.abs(regenerated.gate_update_1-saved.gate_update_1).max())))
    print(f'Exported {subject}: ten aligned outer winners and regenerated trial tables',flush=True)
    return dict(artifacts=artifacts,audits=audits,data_sources=data_sources,exports=exports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-dir',type=Path,default=ROOT/'data/analyses/monoGRU_alignment/nested_DA_S32_B8/monoGRU_relu_ndb')
    parser.add_argument('--jobs',type=int,default=-1,help='Joblib process workers; -1 uses all available CPUs, 1 runs serially')
    parser.add_argument('--output',type=Path,default=ROOT/'data/rnns/subject_examples_aligned')
    args=parser.parse_args();torch.set_num_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((args.alignment_dir/'manifest.json').read_text())
    if source_manifest.get('alignment_protocol') != 'balanced_medoid_reference_fit_v3_choice0_first':
        raise ValueError('Re-run monoGRU_alignment.py for the final choice-0-first convention')
    if source_manifest.get('decoder_bias') is not False:raise ValueError('Expected monoGRU +BC-DB reference')
    rows=pd.read_csv(args.alignment_dir/'aligned_models.htsv',sep='\t')
    if rows.decoder_bias.any() or rows.duplicated(['subject_id','outer_loop_n']).any():raise ValueError('Wrong aligned cohort')
    if not rows.groupby('subject_id').outer_loop_n.apply(lambda x: sorted(x)==list(range(1,11))).all():raise ValueError('Missing outer folds')
    hashes=source_manifest['source_model_sha256']
    if any(mv.sha256(alignment.resolve(p))!=hashes[str(alignment.resolve(p))] for p in rows.model_pickle_path):
        raise ValueError('A source checkpoint changed since alignment')
    groups=list(rows.groupby('subject_id',sort=True))
    started=time.perf_counter()
    # Processes avoid the Python recurrence's GIL; native CPU work stays one-threaded.
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=args.jobs)(delayed(export_subject)(subject,group,out,source_manifest)
                                          for subject,group in groups)
    rollout_seconds=time.perf_counter()-started
    artifacts=[item for result in results for item in result['artifacts']]
    audits=[item for result in results for item in result['audits']]
    exports=[item for result in results for item in result['exports']]
    data_sources={k:v for result in results for k,v in result['data_sources'].items()}
    models=[alignment.load_model(item['example_model_pickle_path']) for item in exports]
    exported=pd.DataFrame(exports).reset_index(drop=True)
    exported.to_csv(out/'outer_fold_winners.htsv',sep='\t',index=False)
    pd.DataFrame(audits).to_csv(out/'regeneration_checks.htsv',sep='\t',index=False)
    reference=alignment.load_model(args.alignment_dir/'reference_model.pickle')
    with (out/'balanced_population_medoid_model.pickle').open('wb') as f:pickle.dump(reference,f)
    averaged,table=average_parameters(models,exported)
    with (out/'average_aligned_model.pickle').open('wb') as f:pickle.dump(averaged,f)
    table.to_csv(out/'average_parameters.htsv',sep='\t',index=False)
    (out/'average_parameters.tex').write_text(parameter_latex(table))
    from NM_TinyRNN.code.figures.monoGRU_appendix import write_appendix, equation_block
    write_appendix(table,out/'average_parameters_methods.tex')
    (out/'parameter_standard_deviations.tex').write_text(equation_block(table,'subject_sd'))
    lines=['# Average aligned monoGRU +BC-DB parameters', '',
        'Equal mean across 11 subject means, each averaging ten aligned outer folds.',
        'SD is across subject means; this is a descriptive parameter average, not a refitted network.', '',
        '| Parameter | Index (stored orientation) | Mean | Subject SD |',
        '|---|---|---:|---:|']
    for r in table.itertuples():lines.append(f'| {r.parameter} | {r.index} | {r.mean:.6f} | {r.subject_sd:.6f} |')
    (out/'average_parameters.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from NM_TinyRNN.code.figures.monoGRU_parameters import plot_model_equations
    fig=plot_model_equations(averaged,equations())
    for ext in ['png','svg','pdf']:fig.savefig(out/f'average_parameters_equations.{ext}',bbox_inches='tight',pad_inches=.15)
    import matplotlib.pyplot as plt
    plt.close(fig)
    metadata=dict(source_run=source_manifest['folder'],model_type='monoGRU +BC-DB',n_models=len(models),
        n_subjects=exported.subject_id.nunique(),reference=source_manifest['reference_source'],
        execution=dict(requested_jobs=args.jobs,subject_tasks=len(groups),backend='loky',worker_threads=1,rollout_seconds=rollout_seconds),
        alignment='Initial full-parameter norm balancing, followed by positive-scale distance minimization to the frozen balanced medoid with semantic unit labels preserved',
        alignment_protocol=source_manifest['alignment_protocol'],
        label_rule=source_manifest['label_rule'],
        trial_data='Regenerated by mechanistic_variability._run_model_on_trials on raw subject sessions; continuous full-subject history; original block split labels retained',
        parameter_average='Equal subject mean of ten aligned outer winners; all named parameters included; descriptive mean network, not a trained fit or predictive ensemble',
        uncertainty='subject_sd and subject_sem describe 11 subject parameter means; fit_sd describes dependent folds; no independent-fold inference',
        alignment_manifest_sha256=mv.sha256(args.alignment_dir/'manifest.json'),
        code_sha256={str(Path(p).relative_to(ROOT)):mv.sha256(Path(p)) for p in [__file__,alignment.__file__,mv.__file__,ROOT/'code/figures/monoGRU_parameters.py',ROOT/'code/measures/monogru_balancing.py',ROOT/'code/figures/monoGRU_appendix.py']},
        subject_data_sha256=data_sources,artifacts=artifacts,
        max_logit_error=max(a['max_logit_error'] for a in audits),max_probability_error=max(a['max_probability_error'] for a in audits))
    for path in out.iterdir():
        if (path.is_file() and path.name!='export_manifest.json'
                and not path.name.startswith('GRU_') and path.name!='README_GRU.md'):
            metadata['artifacts'].append(dict(path=path.name,bytes=path.stat().st_size,sha256=mv.sha256(path)))
    (out/'export_manifest.json').write_text(json.dumps(metadata,indent=2,default=alignment.json_value)+'\n')
    print(f'Exported {len(models)} models to {out}; parameter table has {len(table)} entries',flush=True)


if __name__=='__main__':main()
