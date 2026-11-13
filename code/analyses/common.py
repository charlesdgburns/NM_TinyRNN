"""Small shared helpers for serializable study plans and task execution."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pandas as pd
import torch
from NM_TinyRNN.code.models import datasets, training
from NM_TinyRNN.code.models.save_data import _json_safe

SOURCE_FILES = ['code/analyses/__init__.py', 'code/analyses/common.py', 'code/analyses/execution.py',
                'code/models/datasets.py', 'code/models/rnns.py',
                'code/models/training.py', 'code/models/save_data.py',
                'code/measures/prediction.py']


def normalize(config, study):
    config = deepcopy(config)
    config['study'] = config['workflow'] = study
    config['schema_version'] = 2
    config['sequence_length'] = datasets.validate_sequence_length(config.get('sequence_length', datasets.SEQUENCE_LENGTH))
    config.setdefault('seed', 42)
    if not isinstance(config['seed'], int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    trainer = training.Trainer()
    fields = ['weight_seeds','sparsity_lambdas','energy_lambdas','hebbian_lambdas',
              'covariance_lambdas','batch_size','max_epochs','early_stop','learning_rate','shuffle_seed']
    defaults = {name:deepcopy(getattr(trainer, name)) for name in fields}
    if study == 'pooled_ws':
        defaults['weight_seeds'] = list(range(1, 11))
    overrides = config.get('trainer', {})
    if set(overrides) - set(fields):
        raise ValueError(f'Unknown trainer settings: {set(overrides)-set(fields)}')
    defaults.update(overrides)
    config['trainer'] = defaults
    for name in ['batch_size','max_epochs','early_stop']:
        value = defaults[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if defaults['shuffle_seed'] is not None and (type(defaults['shuffle_seed']) is not int or defaults['shuffle_seed'] < 0):
        raise ValueError('shuffle_seed must be a nonnegative integer or None')
    if not math.isfinite(defaults['learning_rate']) or defaults['learning_rate'] <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not defaults['weight_seeds'] or any(not isinstance(v,int) or v < 0 for v in defaults['weight_seeds']):
        raise ValueError('weight_seeds must contain nonnegative integers')
    for name in ['sparsity_lambdas','energy_lambdas','hebbian_lambdas','covariance_lambdas']:
        if not defaults[name] or any(not math.isfinite(v) or v < 0 for v in defaults[name]):
            raise ValueError(f'{name} must contain finite nonnegative values')
    return config


def json_safe(value):
    return _json_safe(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def plan_payload(plan):
    return json_safe(dict(experiment_df=plan['experiment_df'].to_dict('records'),
                          task_df=plan['task_df'].to_dict('records'), details=plan['details']))


def frames(payload):
    return pd.DataFrame(payload['experiment_df']), pd.DataFrame(payload['task_df'])


def smoke_trainer(trainer, smoke):
    result = deepcopy(trainer)
    if smoke:
        result.update(weight_seeds=[1,2], sparsity_lambdas=[.001], energy_lambdas=[.01], max_epochs=2)
    return result


def task_folder(output, index):
    return Path(output)/f'task-{index:04d}'


def check_result(result):
    if not math.isfinite(result['val_nll']) or not all(torch.isfinite(p).all() for p in result['state_dict'].values()):
        raise ValueError('Task contains nonfinite validation loss or weights')


def analyse_baseline(results, output):
    rows = []
    keys = ['subject_id','model_id'] + (['shuffle_seed'] if 'shuffle_seed' in results else [])
    for identity, group in results.groupby(keys, sort=True):
        rows.append(dict(zip(keys, identity), n_folds=len(group),
                         mean_held_out_nll=float(group.held_out_nll.mean()),
                         pooled_nll=float(group.nll_sum.sum()/group.n_free.sum()),
                         fold_sd=float(group.held_out_nll.std(ddof=1)), n_free=int(group.n_free.sum())))
    summary = pd.DataFrame(rows)
    Path(output).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(output)/'summary.csv', index=False)
    return summary
