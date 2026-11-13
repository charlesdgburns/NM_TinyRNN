"""Local task execution and collection, with no server imports or services."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from NM_TinyRNN.code.analyses import get_study, common


def make_runtime(config,payload,data_root,base_identity,smoke=False):
    study = get_study(config['study'])
    if len(payload['task_df']) != config['task_count'] or [t['index'] for t in payload['task_df']] != list(range(config['task_count'])):
        raise ValueError('Task table must contain each planned index exactly once')
    versions = dict(torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
    identity = hashlib.sha256(json.dumps(dict(bundle=base_identity,smoke=smoke,versions=versions),sort_keys=True).encode()).hexdigest()
    return dict(study=study,config=config,payload=payload,tasks=payload['task_df'],identity=identity,
                versions=versions,smoke=smoke,context=study.init_context(config,payload,data_root,smoke))


def validate_metadata(runtime,meta,index):
    if not 0 <= index < len(runtime['tasks']):
        raise ValueError('Task index is outside the plan')
    task = runtime['tasks'][index]
    expected = dict(identity=runtime['identity'],study=runtime['config']['study'],**task)
    if any(meta.get(key) != value for key,value in expected.items()) or meta.get('versions') != runtime['versions']:
        raise ValueError(f'Task metadata/identity mismatch at index {index}')


def run_task(runtime,index,output):
    if not 0 <= index < len(runtime['tasks']):
        raise ValueError(f'Task index must be in [0, {len(runtime["tasks"])-1}]')
    task = runtime['tasks'][index]
    folder = common.task_folder(output,index)
    if folder.exists():
        raise FileExistsError(f'Task directory already exists: {folder}')
    folder.mkdir(parents=True)
    start = time.monotonic()
    runtime['study'].run_task(task,runtime['config'],folder,runtime['context'])
    common.write_json(folder/'task.json',dict(identity=runtime['identity'],study=runtime['config']['study'],
        versions=runtime['versions'],seconds=time.monotonic()-start,**task))
    return folder


def collect_results(runtime,output):
    if runtime['smoke']:
        raise ValueError('Smoke tasks cannot be collected as full results')
    output = Path(output)
    for index in range(len(runtime['tasks'])):
        meta = json.loads((common.task_folder(output,index)/'task.json').read_text())
        validate_metadata(runtime,meta,index)
    reduced = output/'reduced'
    reduced.mkdir(parents=True,exist_ok=True)
    results = runtime['study'].collect_results(runtime['config'],runtime['payload'],output,runtime['context'])
    runtime['study'].analyse_results(results,reduced)
    task_ids = {}
    for task in runtime['tasks']:
        task_ids.setdefault(task['experiment_id'],[]).append(task['index'])
    experiments = [dict(row,completed=True,artifact_index='reduced/analysis_df.json',
                        task_ids=task_ids[row['experiment_id']])
                   for row in runtime['payload']['experiment_df']]
    reports = {'run-config':runtime['config'],'experiment_df':experiments,
               'task_df':runtime['tasks'],
               'identity':dict(identity=runtime['identity'],versions=runtime['versions'])}
    for name,value in reports.items():
        common.write_json(reduced/f'{name}.json',value)
    return results


def read_analysis_df(output):
    """Resolve portable artifact paths after task archives are unpacked at output/."""
    output = Path(output).resolve()
    frame = pd.DataFrame(json.loads((output/'reduced/analysis_df.json').read_text()))
    for column in ['info_path','model_pickle_path','trials_data_path']:
        frame[column] = frame[column].map(lambda path:str(output/path))
    return frame
