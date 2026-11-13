"""Standard per-subject session-level CV, defined by experiments.get_DA_info_df."""
from pathlib import Path

import pandas as pd
import torch
from NM_TinyRNN.code.models import datasets as ds, experiments, nested_cv as cv, rnns
from NM_TinyRNN.code.measures.prediction import free_choice_nll
from NM_TinyRNN.code.analyses import common

SOURCE_FILES = ['code/analyses/baseline.py', 'code/models/experiments.py', 'code/models/nested_cv.py']


def load_dataset(data_root, subject, row, length, latest_sessions=10):
    # AB_Dataset historically reads DATA_PATH, irrespective of the supplied path.
    original = ds.DATA_PATH
    ds.DATA_PATH = Path(data_root).parent
    try:
        return ds.AB_Dataset(Path(data_root)/subject, subject_ids=[subject], sequence_length=length,
                             input_encoding=row['input_encoding'],
                             input_forced_choice=row['input_forced_choice'], device=torch.device('cpu'),
                             latest_sessions=latest_sessions)
    finally:
        ds.DATA_PATH = original


def dataset_key(row):
    return f"{row['subject_id']}|{row['input_encoding']}|{int(row['input_forced_choice'])}"


def model_for(row, dataset):
    options = rnns.OPTIONS_DICT.copy()
    for key in ['hidden_size','nm_size','nm_dim','nm_mode','input_encoding',
                'input_forced_choice','nonlinearity','decoder_bias']:
        options[key] = row[key]
    options.update(rnn_type=row['model_type'],input_size=dataset.inputs.shape[-1])
    model = rnns.TinyRNN(**options)
    if model.get_model_id() != row['model_id']:
        raise ValueError('Model definition differs from experiment manifest')
    return model


def build_plan(config, data_root):
    config = common.normalize(config,'baseline')
    if config.get('blocks_per_subject') is not None:
        raise ValueError('Baseline uses complete sessions; block budgets belong to data_scaling')
    config.setdefault('sessions_per_subject', 10)
    limit = config['sessions_per_subject']
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < cv.N_OUTER_LOOPS:
        raise ValueError('Baseline sessions_per_subject must be an integer >= 10')
    frame = experiments.get_DA_info_df(processed_data_path=Path(data_root),save_path=Path('results'),
        sequence_length=config['sequence_length'],batch_size=config['trainer']['batch_size'],
        model_types=config.get('model_types'))
    if config.get('subjects') is not None:
        missing = set(config['subjects'])-set(frame.subject_id)
        if missing:
            raise ValueError(f'Unknown baseline subjects: {sorted(missing)}')
        frame = frame[frame.subject_id.isin(config['subjects'])]
    if config.get('models') is not None:
        missing = set(config['models'])-set(frame.model_id)
        if missing:
            raise ValueError(f'Unknown baseline model IDs: {sorted(missing)}')
        frame = frame[frame.model_id.isin(config['models'])]
    if config.get('outer_folds') is not None:
        outers = config['outer_folds']
        if not outers or len(set(outers)) != len(outers) or any(type(n) is not int or not 1 <= n <= cv.N_OUTER_LOOPS for n in outers):
            raise ValueError('outer_folds must be unique fold numbers from 1 to 10')
        frame = frame[frame.outer_loop_n.isin(outers)]
    if config.get('shuffle_seeds') is not None:
        seeds = config['shuffle_seeds']
        if not seeds or len(set(seeds)) != len(seeds) or any(type(n) is not int or n < 0 for n in seeds):
            raise ValueError('shuffle_seeds must be unique nonnegative integers')
        if config['trainer']['shuffle_seed'] is not None:
            raise ValueError('Use shuffle_seeds or trainer.shuffle_seed, not both')
        frame = pd.concat([frame.assign(shuffle_seed=seed) for seed in seeds],ignore_index=True)
    if frame.empty:
        raise ValueError('No baseline experiments selected')
    frame = frame.sort_values(['subject_id','model_id','outer_loop_n']).reset_index(drop=True)
    frame['experiment_id'] = [f'baseline-{i:06d}' for i in range(len(frame))]
    frame['data_path'] = frame.subject_id.map(lambda subject:f'data/AB_behaviour/{subject}')
    frame['completed'] = False # a fresh planned run, not implicit historical checkpoint reuse
    files, data, split_plans = {}, {}, {}
    tasks = []
    for row in frame.to_dict('records'):
        subject = row['subject_id']
        files.setdefault(subject, sorted((Path(data_root)/subject).glob('*/trials.htsv'))[-limit:])
        key = dataset_key(row)
        if key not in data:
            dataset = load_dataset(data_root,subject,row,config['sequence_length'],limit)
            if dataset.subject_df.session_folder_name.nunique() < cv.N_OUTER_LOOPS:
                raise ValueError(f'Baseline subject {subject} needs at least {cv.N_OUTER_LOOPS} sessions')
            data[key] = dataset
            split_plans[key] = [cv.nested_cv_splits(dataset,cv.N_OUTER_LOOPS,outer)
                                for outer in range(1,cv.N_OUTER_LOOPS+1)]
        plan = split_plans[key][row['outer_loop_n']-1]
        for inner in plan['inner_folds']:
            if not inner['train'] or not inner['val'] or not plan['outer_eval']:
                raise ValueError('Baseline contains an empty train/validation/evaluation split')
            tasks.append(dict(index=len(tasks),experiment_id=row['experiment_id'],
                subject_id=subject,model=row['model_id'],outer=row['outer_loop_n']-1,inner=inner['fold_idx'],
                **({'shuffle_seed':row['shuffle_seed']} if 'shuffle_seed' in row else {})))
    config.update(subjects=sorted(files),models=sorted(frame.model_id.unique()),
                  task_count=len(tasks),n_folds=cv.N_OUTER_LOOPS)
    details = dict(selected_sessions={subject:[p.parent.name for p in paths] for subject,paths in files.items()},splits=split_plans,n_blocks={key:len(dataset) for key,dataset in data.items()})
    return dict(config=config,experiment_df=frame,task_df=pd.DataFrame(tasks),details=details,data_files=files)


def init_context(config,payload,data_root,smoke=False):
    rows = {row['experiment_id']:row for row in payload['experiment_df']}
    return dict(rows=rows,data={},data_root=Path(data_root),splits=payload['details']['splits'],smoke=smoke)


def dataset_for(row,config,context):
    key = dataset_key(row)
    if key not in context['data']:
        dataset = load_dataset(context['data_root'],row['subject_id'],row,config['sequence_length'],config['sessions_per_subject'])
        actual = [cv.nested_cv_splits(dataset,cv.N_OUTER_LOOPS,outer)
                  for outer in range(1,cv.N_OUTER_LOOPS+1)]
        if actual != context['splits'][key]:
            raise ValueError('Packaged baseline session splits differ from runtime data')
        context['data'][key] = dataset
    return context['data'][key]


def effective_trainer(row,config,smoke=False):
    trainer = common.smoke_trainer(config['trainer'],smoke)
    if row.get('shuffle_seed') is not None:
        trainer['shuffle_seed'] = int(row['shuffle_seed'])
    # Preserve train_outers' existing constraint semantics, including an active
    # sparsity grid for energy-labelled runs and zero energy for sparsity runs.
    if row['constraint']=='sparsity':
        trainer['energy_lambdas'] = [0]
    return trainer


def run_task(task,config,output,context):
    row = context['rows'][task['experiment_id']]
    dataset = dataset_for(row,config,context)
    split = context['splits'][dataset_key(row)][task['outer']]
    native = Path(output)/row['save_path']
    result = cv._run_inner_fold(split['inner_folds'][task['inner']],model_for(row,dataset),dataset,
        effective_trainer(row,config,context['smoke']),config['seed'],native,
        row['outer_loop_n'],split['outer_eval'])
    checkpoint = dict(state_dict={k:v.detach().cpu() for k,v in result['state_dict'].items()},
        config=list(result['config']),val_nll=result['val_loss'],inner_fold=task['inner'])
    common.check_result(checkpoint)
    torch.save(checkpoint,Path(output)/'result.pt')


def collect_results(config,payload,output,context):
    output = Path(output)
    groups,analysis_rows = {},[]
    for task in payload['task_df']:
        folder = common.task_folder(output,task['index'])
        result = torch.load(folder/'result.pt',map_location='cpu',weights_only=True)
        common.check_result(result)
        if result['inner_fold']!=task['inner']:
            raise ValueError('Checkpoint inner fold differs from task plan')
        groups.setdefault(task['experiment_id'],[]).append((task,result))
        row = dict(context['rows'][task['experiment_id']],task_id=task['index'],inner_loop_idx=task['inner'])
        native = folder/row['save_path']/f"outer_fold_{row['outer_loop_n']}"/f"inner_fold_{task['inner']}"
        for field,suffix in [('info_path','info.json'),('model_pickle_path','model.pickle'),('trials_data_path','trials_data.htsv')]:
            row[field] = str((native/f"{row['model_id']}_{suffix}").relative_to(output))
        row['completed'] = True
        analysis_rows.append(row)
    rows = []
    for experiment_id,candidates in groups.items():
        if len(candidates)!=cv.N_OUTER_LOOPS-1:
            raise ValueError('Missing baseline inner-fold result')
        task,best = min(candidates,key=lambda pair:(pair[1]['val_nll'],pair[0]['inner']))
        row = context['rows'][experiment_id]
        dataset = dataset_for(row,config,context)
        model = model_for(row,dataset)
        model.load_state_dict(best['state_dict'])
        split = context['splits'][dataset_key(row)][task['outer']]
        score = free_choice_nll(model,dataset,split['outer_eval'],config['trainer']['batch_size'])
        rows.append(dict(subject_id=row['subject_id'],model_id=row['model_id'],outer_loop_n=row['outer_loop_n'],
                         selected_inner_fold=task['inner'],selected_task_id=task['index'],
                         selected_val_nll=best['val_nll'],winning_config=best['config'],**score,
                         **({'shuffle_seed':row['shuffle_seed']} if 'shuffle_seed' in row else {})))
    common.write_json(output/'reduced/analysis_df.json',analysis_rows)
    results = pd.DataFrame(rows)
    results.to_csv(output/'reduced/fold_results.csv',index=False)
    return results


def analyse_results(results,output):
    return common.analyse_baseline(results,output)
