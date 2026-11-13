"""Recency-based learning curves with fixed session holdouts.

Standalone: python code/analyses/data_scaling.py --subjects WS20 --output PATH --smoke
The newest session is an extra fixed holdout. Freeze session-level inner/outer
folds once, then take the newest eligible training blocks at each exact budget.
"""
from copy import copy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import torch
from NM_TinyRNN.code.models import datasets, experiments, training, nested_cv as cv
from NM_TinyRNN.code.measures.prediction import free_choice_nll
from NM_TinyRNN.code.analyses import baseline, common

BUDGETS = [512, 1024, 1536, 2048]
MODELS = ['2_unit_GRU_tanh_unipolar_forced', '2_unit_monoGRU_relu_unipolar_forced_ndb']
SOURCE_FILES = baseline.SOURCE_FILES + ['code/analyses/data_scaling.py']


def session_plan(dataset, max_blocks):
    """Find the smallest recent-session pool supporting every full nested split.

    Includes the fixed newest holdout in the pool size. Pool selection always
    checks all 90 splits, even if a local test executes fewer outer folds.
    """
    names = cv._get_sessions(dataset)
    if len(names) < 11:
        raise ValueError('Need at least 11 sessions: fixed holdout plus 10 CV sessions')
    for count in range(11, len(names)+1):
        selected = names[-count:]
        view = copy(dataset)
        view.subject_df = dataset.subject_df[
            dataset.subject_df.session_folder_name.isin(selected[:-1])]
        splits = [cv.nested_cv_splits(view, 10, outer) for outer in range(1, 11)]
        if all(len(inner['train']) >= max_blocks for split in splits for inner in split['inner_folds']):
            return selected.tolist(), splits
    raise ValueError(f'Cannot supply {max_blocks*dataset.sequence_length} training targets in every split')


def build_plan(config, data_root):
    config = common.normalize(config, 'data_scaling')
    budgets = config.setdefault('training_trials', BUDGETS.copy())
    length = config['sequence_length']
    if (not isinstance(budgets, list) or not budgets or
            any(isinstance(n, bool) or not isinstance(n, int) or n < length or n % length for n in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError('training_trials must be increasing unique positive multiples of sequence_length')
    if config.get('sessions_per_subject') is not None or config.get('blocks_per_subject') is not None:
        raise ValueError('Session pools are chosen automatically; use training_trials for budgets')
    subjects = config.get('subjects', experiments.DA_SUBJECT_LIST)
    if not subjects or set(subjects)-set(experiments.DA_SUBJECT_LIST):
        raise ValueError('Choose nonempty DA subjects')
    models = config.get('models', MODELS)
    if not models or set(models)-set(MODELS):
        raise ValueError(f'Choose models from {MODELS}')
    outers = config.setdefault('outer_folds', list(range(1,11)))
    if not outers or any(type(n) is not int or not 1 <= n <= 10 for n in outers) or len(set(outers))!=len(outers):
        raise ValueError('outer_folds must contain unique fold numbers from 1 to 10')
    base = experiments.get_DA_info_df(processed_data_path=Path(data_root), save_path=Path('results'),
        sequence_length=length, batch_size=config['trainer']['batch_size'])
    base = base[base.subject_id.isin(subjects) & base.model_id.isin(models) & base.outer_loop_n.isin(outers)]
    if set(base.subject_id)!=set(subjects) or set(base.model_id)!=set(models):
        raise ValueError('Requested subjects/models are absent from the experiment preset')
    files, details, rows, tasks = {}, {}, [], []
    for subject in sorted(subjects):
        row = base[base.subject_id.eq(subject)].iloc[0].to_dict()
        all_data = baseline.load_dataset(data_root,subject,row,length,latest_sessions=None)
        selected, _ = session_plan(all_data,max(budgets)//length)
        dataset = baseline.load_dataset(data_root,subject,row,length,latest_sessions=len(selected))
        selected, full_splits = session_plan(dataset,max(budgets)//length)
        files[subject] = sorted((Path(data_root)/subject).glob('*/trials.htsv'))[-len(selected):]
        details[subject] = dict(selected_sessions=selected, n_sessions=len(selected),
            fixed_holdout_session=selected[-1],fixed_holdout=cv._blocks_for_sessions(dataset,[selected[-1]]),
            splits={},n_blocks=len(dataset))
        block_sessions = dataset.subject_df.dropna(subset=['sequence_block_idx']).groupby('sequence_block_idx').session_folder_name.first()
        for budget in budgets:
            splits=[]
            for full in full_splits:
                split=dict(outer_eval=full['outer_eval'],inner_folds=[])
                for inner in full['inner_folds']:
                    # Block indices follow timestamp session order and within-session time.
                    chosen=sorted(inner['train'])[-budget//length:]
                    split['inner_folds'].append(dict(fold_idx=inner['fold_idx'],train=chosen,val=inner['val'],
                        training_sessions=sorted(set(block_sessions.loc[chosen].tolist()))))
                splits.append(split)
            details[subject]['splits'][str(budget)]=splits
    for source in base.sort_values(['subject_id','model_id','outer_loop_n']).to_dict('records'):
        for budget in budgets:
            row=dict(source,training_trials=budget,completed=False,experiment_id=f'data-scaling-{len(rows):06d}')
            row['save_path']=f"results/data_scaling/{row['subject_id']}/T{budget}/{row['model_id']}"
            row['data_path']=f"data/AB_behaviour/{row['subject_id']}"
            rows.append(row)
            for inner in range(9):
                tasks.append(dict(index=len(tasks),experiment_id=row['experiment_id'],subject_id=row['subject_id'],
                    model=row['model_id'],outer=row['outer_loop_n']-1,inner=inner,training_trials=budget))
    config.update(subjects=sorted(subjects),models=sorted(models),task_count=len(tasks),n_folds=10)
    return dict(config=config,experiment_df=pd.DataFrame(rows),task_df=pd.DataFrame(tasks),
        details=dict(subjects=details,n_blocks={s:d['n_blocks'] for s,d in details.items()}),data_files=files)


def init_context(config,payload,data_root,smoke=False):
    return dict(rows={r['experiment_id']:r for r in payload['experiment_df']},data={},
        subjects=payload['details']['subjects'],data_root=Path(data_root),smoke=smoke)


def dataset_for(row,config,context):
    subject=row['subject_id']
    if subject not in context['data']:
        info=context['subjects'][subject]
        data=baseline.load_dataset(context['data_root'],subject,row,config['sequence_length'],info['n_sessions'])
        if cv._get_sessions(data).tolist()!=info['selected_sessions']:
            raise ValueError('Runtime sessions differ from frozen scaling plan')
        context['data'][subject]=data
    return context['data'][subject]


def split_for(row,context):
    return context['subjects'][row['subject_id']]['splits'][str(row['training_trials'])][row['outer_loop_n']-1]


def run_task(task,config,output,context):
    row=context['rows'][task['experiment_id']]
    dataset=dataset_for(row,config,context)
    split=split_for(row,context)
    result=cv._run_inner_fold(split['inner_folds'][task['inner']],baseline.model_for(row,dataset),dataset,
        baseline.effective_trainer(row,config,context['smoke']),config['seed'],Path(output)/row['save_path'],
        row['outer_loop_n'],split['outer_eval'])
    checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in result['state_dict'].items()},
        config=list(result['config']),val_nll=result['val_loss'],inner_fold=task['inner'])
    common.check_result(checkpoint)
    torch.save(checkpoint,Path(output)/'result.pt')


def collect_results(config,payload,output,context):
    output=Path(output);groups={};analysis=[];results=[]
    for task in payload['task_df']:
        folder=common.task_folder(output,task['index'])
        checkpoint=torch.load(folder/'result.pt',map_location='cpu',weights_only=True)
        common.check_result(checkpoint)
        if checkpoint['inner_fold']!=task['inner']:raise ValueError('Checkpoint fold mismatch')
        groups.setdefault(task['experiment_id'],[]).append((task,checkpoint))
        row=dict(context['rows'][task['experiment_id']],task_id=task['index'],inner_loop_idx=task['inner'],completed=True)
        native=folder/row['save_path']/f"outer_fold_{row['outer_loop_n']}"/f"inner_fold_{task['inner']}"
        for field,suffix in [('info_path','info.json'),('model_pickle_path','model.pickle'),('trials_data_path','trials_data.htsv')]:
            row[field]=str((native/f"{row['model_id']}_{suffix}").relative_to(output))
        analysis.append(row)
    for experiment_id,candidates in groups.items():
        if len(candidates)!=9:raise ValueError('Missing inner-fold result')
        task,best=min(candidates,key=lambda pair:(pair[1]['val_nll'],pair[0]['inner']))
        row=context['rows'][experiment_id];data=dataset_for(row,config,context);split=split_for(row,context)
        model=baseline.model_for(row,data);model.load_state_dict(best['state_dict'])
        score=free_choice_nll(model,data,split['outer_eval'],config['trainer']['batch_size'])
        hold=free_choice_nll(model,data,context['subjects'][row['subject_id']]['fixed_holdout'],config['trainer']['batch_size'])
        inner=split['inner_folds'][task['inner']]
        results.append(dict(subject_id=row['subject_id'],model_id=row['model_id'],training_trials=row['training_trials'],
            outer_loop_n=row['outer_loop_n'],selected_inner_fold=task['inner'],selected_task_id=task['index'],
            selected_val_nll=best['val_nll'],winning_config=best['config'],n_training_sessions=len(inner['training_sessions']),
            training_sessions=inner['training_sessions'],**score,
            fixed_holdout_nll=hold['held_out_nll'],fixed_holdout_nll_sum=hold['nll_sum'],fixed_holdout_n_free=hold['n_free']))
    common.write_json(output/'reduced/analysis_df.json',analysis)
    frame=pd.DataFrame(results);frame.to_csv(output/'reduced/fold_results.csv',index=False)
    return frame


def analyse_results(results,output):
    summary=results.groupby(['subject_id','model_id','training_trials'],as_index=False).agg(
        mean_outer_nll=('held_out_nll','mean'),mean_fixed_holdout_nll=('fixed_holdout_nll','mean'),
        n_outer_folds=('outer_loop_n','count'))
    summary.to_csv(Path(output)/'summary.csv',index=False)
    return summary


def main(argv=None):
    import argparse,json,hashlib
    from NM_TinyRNN.code.analyses import execution
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects',nargs='+',default=experiments.DA_SUBJECT_LIST)
    parser.add_argument('--training-trials',type=int,nargs='+',default=BUDGETS)
    parser.add_argument('--outer-folds',type=int,nargs='+',help='1-based folds; default all 10, smoke default first two')
    parser.add_argument('--sequence-length',type=int,default=datasets.SEQUENCE_LENGTH)
    parser.add_argument('--batch-size',type=int,default=training.BATCH_SIZE)
    parser.add_argument('--early-stop',type=int,default=training.EARLY_STOP)
    parser.add_argument('--seed',type=int,default=42,help='Base seed for fold-derived training RNGs')
    parser.add_argument('--shuffle-seed',type=int,help='Optional loader-only override; default follows fold seed')
    parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[2]/'data/AB_behaviour')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='Two epochs, two initializations, reduced regularizer grid; diagnostic only')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.output.exists():parser.error('Output already exists; choose a new directory')
    config=dict(subjects=args.subjects,training_trials=args.training_trials,sequence_length=args.sequence_length,seed=args.seed,
        outer_folds=args.outer_folds or ([1,2] if args.smoke else list(range(1,11))),trainer=dict(batch_size=args.batch_size,early_stop=args.early_stop,shuffle_seed=args.shuffle_seed))
    if args.smoke:
        config['trainer']=common.smoke_trainer(common.normalize(config,'data_scaling')['trainer'],True)
        config['diagnostic_only']=True
    try:plan=build_plan(config,args.data_root)
    except ValueError as error:parser.error(str(error))
    payload=common.plan_payload(plan)
    common.write_json(args.output/'run-config.json',plan['config']);common.write_json(args.output/'plan.json',payload)
    sources=[Path(__file__).resolve().parents[2]/p for p in dict.fromkeys(common.SOURCE_FILES+SOURCE_FILES)]
    files=sources+[p for paths in plan['data_files'].values() for p in paths]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    common.write_json(args.output/'inputs.json',hashes)
    for subject,detail in plan['details']['subjects'].items():
        print(f"{subject}: {detail['n_sessions']} recent sessions including fixed holdout {detail['fixed_holdout_session']}",flush=True)
    print(f"{len(plan['task_df'])} fits; budgets {args.training_trials}; diagnostic={args.smoke}",flush=True)
    if args.plan_only:return
    identity=hashlib.sha256(json.dumps(dict(hashes=hashes,config=plan['config'],plan=payload),sort_keys=True).encode()).hexdigest()
    runtime=execution.make_runtime(plan['config'],payload,args.data_root,identity)
    for task in payload['task_df']:
        execution.run_task(runtime,task['index'],args.output)
        print(f"Completed {task['index']+1}/{len(payload['task_df'])}: {task['model']} T{task['training_trials']} outer{task['outer']+1} inner{task['inner']}",flush=True)
    execution.collect_results(runtime,args.output)
    print(args.output/'reduced/summary.csv',flush=True)


if __name__=='__main__':main()
