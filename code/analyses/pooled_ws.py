#!/usr/bin/env python3
"""Pooled WS-subject nested 10-fold block CV, run locally without SLURM.

    .venv/bin/python code/analyses/pooled_ws.py --dry-run
    .venv/bin/python code/analyses/pooled_ws.py --jobs 12
    .venv/bin/python code/analyses/pooled_ws.py --smoke-test \
        --models monoGRU_relu_biasoff --output /tmp/pooled-ws-smoke --jobs 12

All WS* mouse directories and ALL datetime-named sessions are used by default.
Each subject's complete blocks are shuffled and partitioned into 10 folds
BEFORE pooling. Every outer test fold and inner validation fold consequently
contains every subject. Each block uses sequence_length + 1 source trials for sequence_length next-choice
pairs. Blocks never cross sessions. This is within-subject generalization,
not leave-subject-out CV: sessions may appear in both training and test.

For each of 8 two-unit models: 10 outer folds x 9 inner fits. Each fit searches
10 weight seeds x 5 sparsity x 3 energy values with Trainer's vmap; best
validation model across inner fits is evaluated without refitting. Defaults:
sequence length from datasets.SEQUENCE_LENGTH, batch size from training.BATCH_SIZE, lr=.01, patience 100, 10,000 maximum epochs.
Training RNGs use base seed * 10000 + compact inner-fold position (0..8);
the same policy is used by the standalone CLI and shared/AWS adapter. Initial
hidden states are shared learned vectors initialized at zero, reset per block.
Training pools blocks (subjects with more blocks contribute more). Evaluation
reports both pooled trial-weighted and equal-subject mean NLL.

Native-style artifacts: <output>/<kind>/<constraint>/outer_fold_N/inner_fold_M/
<model_id>_{model.pickle,info.json,trials_data.htsv,complete.json}.
Trial export resets hidden state per session, including subject boundaries.
Checkpoint, data/code manifest and output lock support safe local resume.
Smoke mode uses 10 random complete blocks PER SUBJECT, 2 seeds, one penalty
pair and 2 epochs; retains all subjects and all 10x9 folds. Not scientific results.
"""
from __future__ import annotations
import argparse
from copy import copy,deepcopy
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys
from contextlib import contextmanager

# Independent, one-thread CPU workers; no competing GPU contexts.
for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES'): os.environ[name]='-1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
import torch
from joblib import Parallel,delayed,parallel_config,effective_n_jobs
from torch.utils.data import Dataset,DataLoader,Subset
from NM_TinyRNN.code.models.training import Trainer, BATCH_SIZE, EARLY_STOP
from NM_TinyRNN.code.models.nested_cv import _fold_seed
from NM_TinyRNN.code.models import datasets as ds
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.models.save_data import get_model_trial_by_trial_df
from NM_TinyRNN.code.measures.prediction import free_choice_nll as nll

K=10
MODELS={f'{kind}_{act}_bias{"on" if bias else "off"}':(kind,act,bias)
        for kind,act,bias in itertools.product(['monoGRU','GRU'],['tanh','relu'],[False,True])}

@contextmanager
def atomic_file(path,mode='w'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open(mode) as stream:
            yield stream
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def write_json(path,value):
    with atomic_file(path) as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def discover(data_root,subjects=None,sessions_per_subject=None):
    dirs=sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('WS'))
    if subjects is not None:
        available={p.name:p for p in dirs}
        missing=set(subjects)-available.keys()
        if missing:raise ValueError(f'Unknown WS subjects: {sorted(missing)}')
        dirs=[available[s] for s in sorted(set(subjects))]
    if not dirs:raise ValueError(f'No WS* subjects found under {data_root}')
    files={}
    for subject in dirs:
        sessions=sorted((p for p in subject.iterdir() if p.is_dir() and (p/'trials.htsv').is_file()),
                        key=lambda p:datetime.strptime(p.name,'%Y-%m-%d-%H%M%S'))
        if not sessions:raise ValueError(f'No sessions for {subject.name}')
        if sessions_per_subject is not None:
            if len(sessions)<sessions_per_subject:raise ValueError(f'{subject.name} has only {len(sessions)} sessions')
            sessions=sessions[-sessions_per_subject:]
        files[subject.name]=[p/'trials.htsv' for p in sessions]
    return files

class PooledDataset(Dataset):
    def __init__(self,files,seed=42,blocks_per_subject=None,sequence_length=ds.SEQUENCE_LENGTH):
        self.sequence_length=ds.validate_sequence_length(sequence_length)
        S=self.sequence_length
        frames=[];offset=0
        for subject,paths in sorted(files.items()):
            for path in paths:
                df=pd.read_csv(path,sep='\t')
                required={'forced_choice','choice','outcome','good_poke'}
                if not required.issubset(df):raise ValueError(f'{path}: missing {required-set(df)}')
                if df[list(required)].isna().any().any():raise ValueError(f'{path}: missing behavioral values')
                n=len(df);complete=(n//(S+1))*(S+1)
                df['subject_id']=subject
                df['session_folder_name']=subject+'/'+path.parent.name
                df['session_trial_idx']=np.arange(n)
                df['sequence_block_idx']=np.r_[np.arange(complete)//(S+1)+offset,
                                                np.full(n-complete,np.nan)]
                df['block_trial_idx']=np.r_[np.arange(complete),np.full(n-complete,np.nan)]
                offset+=n//(S+1);frames.append(df)
        df=pd.concat(frames,ignore_index=True)
        df['forced_choice']=df.forced_choice.astype(int);df['outcome']=df.outcome.astype(int)
        if not df.forced_choice.isin([0,1]).all() or not df.outcome.isin([0,1]).all():
            raise ValueError('forced_choice and outcome must be binary')
        choices=pd.Categorical(df.choice)
        if len(choices.categories)!=2:raise ValueError(f'Expected exactly 2 global choice labels: {choices.categories}')
        self.choice_labels=[str(v) for v in choices.categories]
        df['choice']=choices.codes.astype(int)
        df['good_poke']=pd.Categorical(df.good_poke).codes.astype(int)
        # Optional reduced-data smoke/budget; sample separately inside each subject.
        if blocks_per_subject is not None:
            keep=[]
            for subject,group in df.groupby('subject_id',sort=True):
                blocks=group.sequence_block_idx.dropna().unique().astype(int)
                if len(blocks)<blocks_per_subject:raise ValueError(f'{subject}: fewer than {blocks_per_subject} blocks')
                salt=int(hashlib.sha256(subject.encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(np.random.SeedSequence([seed,salt]))
                keep.extend(rng.choice(blocks,size=blocks_per_subject,replace=False))
            df=df[df.sequence_block_idx.isin(keep)].copy()
        # Remap source block indices once, BEFORE subject-level splitting.
        ids=sorted(df.sequence_block_idx.dropna().unique())
        mapping={old:new for new,old in enumerate(ids)}
        df['sequence_block_idx']=df.sequence_block_idx.map(mapping)
        self.subject_df=df.reset_index(drop=True)
        valid=self.subject_df.dropna(subset=['sequence_block_idx'])
        raw=torch.tensor(valid[['forced_choice','choice','outcome']].to_numpy(),dtype=torch.float32)
        seq=raw.reshape(-1,S+1,3)
        self.inputs=seq[:,:-1,:].contiguous()
        self.targets=seq[:,1:,1].long().contiguous()
        self.forced_choice_mask=seq[:,1:,0].contiguous()
        self.block_subjects=valid.groupby('sequence_block_idx',sort=True).subject_id.first().to_numpy()
        self.splits=None
    def __len__(self):return len(self.inputs)
    def __getitem__(self,i):return self.inputs[i],self.targets[i],self.forced_choice_mask[i]
    def _session_split(self,**kwargs):
        if self.splits is None:raise RuntimeError('Explicit pooled CV splits must be supplied')
        return self.splits

def subject_folds(dataset,seed):
    folds=[[] for _ in range(K)];subject_plan={}
    for subject in sorted(set(dataset.block_subjects)):
        blocks=np.flatnonzero(dataset.block_subjects==subject)
        if len(blocks)<K:raise ValueError(f'{subject} has {len(blocks)} blocks; at least {K} required')
        salt=int(hashlib.sha256(subject.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(np.random.SeedSequence([seed,salt,1]))
        parts=[p.tolist() for p in np.array_split(rng.permutation(blocks),K)]
        for f,part in enumerate(parts):
            if not (dataset.forced_choice_mask[part]==0).any():
                raise ValueError(f'{subject}, fold {f+1}: no free-choice targets; choose another seed')
            folds[f].extend(part)
        subject_plan[subject]={'n_blocks':len(blocks),'fold_block_counts':list(map(len,parts))}
    flattened=[b for fold in folds for b in fold]
    if len(flattened)!=len(dataset) or set(flattened)!=set(range(len(dataset))):
        raise AssertionError('Folds must partition all blocks exactly once')
    return folds,subject_plan

def splits_for(folds,outer,inner):
    if outer==inner:raise ValueError('Outer and inner fold must differ')
    return {'train':[b for i,f in enumerate(folds) if i not in (outer,inner) for b in f],
            'val':folds[inner],'eval':folds[outer]}

def model_directory(output,key):
    kind,activation,bias=MODELS[key]
    return output/kind/('energy' if activation=='relu' else 'sparsity')

def make_model(key):
    kind,activation,bias=MODELS[key]
    return TinyRNN(rnn_type=kind,hidden_size=2,nonlinearity=activation,input_size=3,
                   input_encoding='unipolar',input_forced_choice=True,init_decoder=False,decoder_bias=bias)

def export_inner(model,result,dataset,splits,trainer_kwargs,folder,outer,inner_idx):
    winner=deepcopy(model);winner.load_state_dict(result['state_dict']);winner.eval()
    names=['sparsity_lambda','energy_lambda','hebbian_lambda','covariance_lambda','weight_seed']
    config=dict(zip(names,result['config']))
    for name,value in config.items():setattr(winner,name,value)
    prefix=folder/winner.get_model_id()
    with atomic_file(str(prefix)+'_model.pickle','wb') as f:pickle.dump(winner,f)
    eval_result=nll(winner,dataset,splits['eval'],trainer_kwargs['batch_size'])
    # Preserve native trial table columns, but reset hidden/logit history per session.
    with atomic_file(str(prefix)+'_trials_data.htsv') as stream:
        first=True
        for _,frame in dataset.subject_df.groupby('session_folder_name',sort=False):
            local=copy(dataset);local.subject_df=frame.reset_index(drop=True)
            table=get_model_trial_by_trial_df(winner,local,splits)
            table.to_csv(stream,sep='\t',index=False,header=first);first=False
    info={'outer_loop_number':outer+1,'inner_fold_idx':inner_idx,'winning_config':config,
          'val_loss':result['val_nll'],'eval_pred_loss':eval_result['held_out_nll'],
          'split_sizes':{k:len(v) for k,v in splits.items()},'n_outer_eval_blocks':len(splits['eval']),
          'hparam_ranges':trainer_kwargs,'model_options':winner.get_options_dict(),
          'trial_export_reset':'session','subject_split_sizes':{
              subject:{label:int(np.sum(dataset.block_subjects[ids]==subject)) for label,ids in splits.items()}
              for subject in sorted(set(dataset.block_subjects))}}
    # Convert any model-option tensors/NumPy values through the repository helper.
    from NM_TinyRNN.code.models.save_data import _json_safe
    write_json(str(prefix)+'_info.json',_json_safe(info))
    write_json(str(prefix)+'_complete.json',{'outer_fold':outer+1,'inner_fold_idx':inner_idx})

def fit_inner(key,dataset,folds,outer,inner,output,trainer_kwargs,threads,train_seed=42):
    position=inner if inner<outer else inner-1
    trainer_kwargs=dict(trainer_kwargs,train_seed=_fold_seed(train_seed,position))
    torch.set_num_threads(threads)
    from functools import partial
    from tqdm import tqdm
    from NM_TinyRNN.code.models import training
    training.tqdm=partial(tqdm,disable=True)
    model=make_model(key);model_id=model.get_model_id();directory=model_directory(output,key)
    checkpoint=directory/'.checkpoints'/model_id/f'outer_{outer+1:02d}_inner_{inner+1:02d}.pt'
    result=None
    if checkpoint.exists():
        try:
            result=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if not math.isfinite(result['val_nll']):raise ValueError('Nonfinite cached loss')
            model.load_state_dict(result['state_dict'])
        except (EOFError,RuntimeError,ValueError,KeyError,pickle.UnpicklingError):
            checkpoint.rename(checkpoint.with_suffix('.pt.interrupted'));result=None
    local=copy(dataset);local.splits=splits_for(folds,outer,inner)
    if result is None:
        trainer=Trainer(**trainer_kwargs);state,config,_=trainer.fit(model,local)
        loss=float(trainer._last_best_val_loss)
        if not math.isfinite(loss) or not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite fit')
        result={'state_dict':{k:v.detach().cpu() for k,v in state.items()},
                'config':list(config),'val_nll':loss,'inner_fold':inner+1}
        with atomic_file(checkpoint,'wb') as f:torch.save(result,f)
    idx=inner if inner<outer else inner-1
    folder=directory/f'outer_fold_{outer+1}'/f'inner_fold_{idx}'
    marker=folder/f'{model_id}_complete.json'
    complete=False
    if marker.exists():
        try:
            json.loads(marker.read_text())
            complete=all((folder/f'{model_id}_{s}').is_file() and (folder/f'{model_id}_{s}').stat().st_size>0
                         for s in ('model.pickle','info.json','trials_data.htsv'))
        except (OSError,ValueError):pass
    if not complete:export_inner(model,result,local,local.splits,trainer_kwargs,folder,outer,idx)
    print(f'Completed {key} outer={outer+1} inner={idx} val_nll={result["val_nll"]:.6f}',flush=True)
    return key,outer,result

def outer_path(output,key,outer):
    return model_directory(output,key)/f'outer_fold_{outer+1}'/f'{make_model(key).get_model_id()}_outer_result.json'

def report(output,rows):
    frame=pd.DataFrame(rows).sort_values(['model','outer_fold'])
    with atomic_file(output/'fold_results.csv') as f:frame.drop(columns='per_subject').to_csv(f,index=False)
    per_subject=[];summaries=[]
    for row in rows:
        for subject,metrics in row['per_subject'].items():
            per_subject.append(dict(model=row['model'],outer_fold=row['outer_fold'],subject_id=subject,**metrics))
    with atomic_file(output/'subject_fold_results.csv') as f:pd.DataFrame(per_subject).to_csv(f,index=False)
    for model,g in frame.groupby('model'):
        if len(g)!=K:continue
        summaries.append(dict(model=model,n_folds=K,mean_held_out_nll=float(g.held_out_nll.mean()),
                              pooled_nll=float(g.nll_sum.sum()/g.n_free.sum()),
                              mean_subject_nll=float(g.mean_subject_nll.mean()),fold_sd=float(g.held_out_nll.std(ddof=1)),
                              n_free=int(g.n_free.sum())))
    if summaries:
        with atomic_file(output/'summary.csv') as f:pd.DataFrame(summaries).to_csv(f,index=False)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root',type=Path,default=REPO/'data/AB_behaviour')
    p.add_argument('--output',type=Path,default=REPO/'data/rnns/nested_pooled_WS')
    p.add_argument('--subjects',nargs='+',help='Default: every WS* directory')
    p.add_argument('--sessions-per-subject',type=int,help='Optional latest N per subject; default: ALL')
    p.add_argument('--blocks-per-subject',type=int,help='Optional random block budget per subject; default: ALL')
    p.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--shuffle-seed',type=int,help='Loader-only override; default follows inner-fold training seed')
    p.add_argument('--jobs',type=int,default=-1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--sequence-length',type=int,default=ds.SEQUENCE_LENGTH,
                   help='Predictions per block (default: datasets.SEQUENCE_LENGTH)')
    p.add_argument('--max-epochs',type=int,default=10000)
    p.add_argument('--early-stop',type=int,default=EARLY_STOP)
    p.add_argument('--learning-rate',type=float,default=.01)
    p.add_argument('--weight-seeds',nargs='+',type=int,default=list(range(1,11)))
    p.add_argument('--sparsity-lambdas',nargs='+',type=float,default=[.1,.01,.001,.0001,.00001])
    p.add_argument('--energy-lambdas',nargs='+',type=float,default=[.1,.01,.001])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    for name in ('sequence_length','threads','batch_size','max_epochs','early_stop','sessions_per_subject','blocks_per_subject'):
        value=getattr(args,name)
        if value is not None and value<1:p.error(f'{name} must be positive')
    if args.jobs==0:p.error('--jobs cannot be zero')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:p.error('Invalid learning rate')
    if any(not math.isfinite(x) or x<0 for x in args.sparsity_lambdas+args.energy_lambdas):p.error('Invalid penalties')
    if args.seed<0:p.error('--seed must be nonnegative')
    if args.shuffle_seed is not None and args.shuffle_seed<0:p.error('--shuffle-seed must be nonnegative')
    if args.smoke_test:
        args.blocks_per_subject=10 if args.blocks_per_subject is None else args.blocks_per_subject
        args.weight_seeds=[1,2];args.sparsity_lambdas=[.001];args.energy_lambdas=[.01];args.max_epochs=2
    args.models=list(dict.fromkeys(args.models))
    return args


def main():
    args=parse_args();torch.set_num_threads(args.threads)
    files=discover(args.data_root,args.subjects,args.sessions_per_subject)
    dataset=PooledDataset(files,args.seed,args.blocks_per_subject,sequence_length=args.sequence_length)
    folds,subject_plan=subject_folds(dataset,args.seed)
    trainer=dict(weight_seeds=args.weight_seeds,sparsity_lambdas=args.sparsity_lambdas,
                 energy_lambdas=args.energy_lambdas,hebbian_lambdas=[0.],covariance_lambdas=[0.],
                 batch_size=args.batch_size,max_epochs=args.max_epochs,early_stop=args.early_stop,
                 learning_rate=args.learning_rate,shuffle_seed=args.shuffle_seed)
    paths=[Path(__file__),*(REPO/'code/models'/name for name in ('rnns.py','datasets.py','training.py','save_data.py'))]
    paths.extend([REPO/'code/measures/prediction.py',REPO/'code/models/nested_cv.py'])
    paths.extend(path for session_files in files.values() for path in session_files)
    manifest={'subjects':sorted(files),'sessions':{s:[p.parent.name for p in ps] for s,ps in files.items()},
              'sequence_length':args.sequence_length,'n_folds':K,'inner_folds':K-1,'models':args.models,'trainer':trainer,
              'seed':args.seed,'blocks_per_subject':args.blocks_per_subject,'smoke_test':args.smoke_test,
              'choice_labels':dataset.choice_labels,'folds':folds,'subject_plan':subject_plan,
              'versions':{'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__},
              'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.output.mkdir(parents=True,exist_ok=True)
    # Keep lock for the entire run; don't allow two processes to write the same fits.
    with (args.output/'.run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is using this output directory')
        manifest_path=args.output/'run.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
            raise ValueError('Output has different code/data/settings; choose a new --output')
        write_json(manifest_path,manifest)
        block_rows=dataset.subject_df.dropna(subset=['sequence_block_idx']).groupby('sequence_block_idx').agg(
            subject_id=('subject_id','first'),session_folder_name=('session_folder_name','first'))
        block_rows['fold']=pd.Series({block:f+1 for f,ids in enumerate(folds) for block in ids})
        with atomic_file(args.output/'block_assignments.csv') as f:block_rows.to_csv(f)
        for subject,plan in subject_plan.items():
            print(f'{subject}: {len(files[subject])} sessions, {plan["n_blocks"]} blocks; fold counts {plan["fold_block_counts"]}',flush=True)
        configs=len(args.weight_seeds)*len(args.sparsity_lambdas)*len(args.energy_lambdas)
        print(f'{len(files)} subjects, {len(dataset)} pooled blocks, {len(args.models)*K*(K-1)} inner fits x {configs} configurations',flush=True)
        if args.dry_run:return
        rows=[];pending={};tasks=[]
        for key in args.models:
            for outer in range(K):
                path=outer_path(args.output,key,outer)
                if path.exists():rows.append(json.loads(path.read_text()));continue
                pending[key,outer]=[]
                for inner in range(K):
                    if inner!=outer:tasks.append(delayed(fit_inner)(key,dataset,folds,outer,inner,args.output,trainer,args.threads,train_seed=args.seed))
        jobs=effective_n_jobs(args.jobs)
        print(f'Running {jobs} workers x {args.threads} threads; {len(tasks)} unfinished/cached inner tasks',flush=True)
        with parallel_config(backend='loky',inner_max_num_threads=args.threads):
            for key,outer,result in Parallel(n_jobs=jobs,return_as='generator_unordered',pre_dispatch=jobs)(tasks):
                results=pending[key,outer];results.append(result)
                if len(results)<K-1:continue
                best=min(results,key=lambda r:(r['val_nll'],r['inner_fold']))
                model=make_model(key);model.load_state_dict(best['state_dict'])
                pooled=nll(model,dataset,folds[outer],args.batch_size)
                per_subject={s:nll(model,dataset,[b for b in folds[outer] if dataset.block_subjects[b]==s],args.batch_size)
                             for s in sorted(files)}
                row=dict(model=key,outer_fold=outer+1,**pooled,per_subject=per_subject,
                         mean_subject_nll=float(np.mean([r['held_out_nll'] for r in per_subject.values()])),
                         selected_inner_fold=best['inner_fold'],selected_val_nll=best['val_nll'],winning_config=best['config'])
                write_json(outer_path(args.output,key,outer),row);rows.append(row);report(args.output,rows)
                del pending[key,outer]
        report(args.output,rows)
        print(f'Saved pooled and per-subject CV results in {args.output.resolve()}',flush=True)



# Server-independent study interface. The local CLI above remains available.
SOURCE_FILES = ['code/analyses/pooled_ws.py', 'code/models/pooled_ws_cv.py', 'code/models/nested_cv.py']


def build_plan(config, data_root):
    from NM_TinyRNN.code.analyses import common
    config = common.normalize(config, 'pooled_ws')
    if config.get('model_types') is not None:
        raise ValueError('pooled_ws selects --models keys; --model-types applies only to baseline')
    models = list(dict.fromkeys(MODELS if config.get('models') is None else config['models']))
    if not models or set(models)-MODELS.keys():
        raise ValueError(f'Choose pooled model keys from {list(MODELS)}')
    for name in ['sessions_per_subject','blocks_per_subject']:
        value = config.get(name)
        if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    files = discover(Path(data_root), config.get('subjects'), config.get('sessions_per_subject'))
    dataset = PooledDataset(files, config['seed'], config.get('blocks_per_subject'),
                            sequence_length=config['sequence_length'])
    folds, subjects = subject_folds(dataset, config['seed'])
    experiments = []
    tasks = []
    for key in models:
        kind, activation, bias = MODELS[key]
        model_id = make_model(key).get_model_id()
        for outer in range(K):
            experiment_id = f'{key}-outer-{outer+1:02d}'
            experiments.append(dict(experiment_id=experiment_id, subject_id='WS_all', model=key,
                model_id=model_id, model_type=kind, hidden_size=2, nonlinearity=activation,
                decoder_bias=bias, constraint='energy' if activation=='relu' else 'sparsity',
                outer_loop_n=outer+1, sequence_length=config['sequence_length'],
                batch_size=config['trainer']['batch_size'], data_path='data/AB_behaviour',
                save_path=str(model_directory(Path('results'), key)), completed=False))
            for inner in range(K):
                if inner != outer:
                    tasks.append(dict(index=len(tasks), experiment_id=experiment_id,
                        subject_id='WS_all', model=key, outer=outer, inner=inner))
    config.update(models=models, subjects=sorted(files), n_folds=K, task_count=len(tasks),
                  blocks_per_subject=config.get('blocks_per_subject'), folds=folds, subject_plan=subjects)
    details = dict(folds=folds, subject_plan=subjects, n_blocks=len(dataset))
    return dict(config=config, experiment_df=pd.DataFrame(experiments), task_df=pd.DataFrame(tasks),
                details=details, data_files=files)


def init_context(config, payload, data_root, smoke=False):
    from NM_TinyRNN.code.analyses import common
    torch.set_num_threads(1)
    files = discover(Path(data_root), config['subjects'], config.get('sessions_per_subject'))
    data = PooledDataset(files, config['seed'], 10 if smoke else config['blocks_per_subject'],
                         sequence_length=config['sequence_length'])
    folds, subjects = subject_folds(data, config['seed'])
    if not smoke and (folds != payload['details']['folds'] or subjects != payload['details']['subject_plan']):
        raise ValueError('Packaged pooled fold plan differs from runtime data partition')
    return dict(data=data, folds=folds, trainer=common.smoke_trainer(config['trainer'], smoke))


def run_task(task, config, output, context):
    fit_inner(task['model'], context['data'], context['folds'], task['outer'], task['inner'],
              Path(output), context['trainer'], 1, train_seed=config['seed'])


def collect_results(config, payload, output, context):
    from NM_TinyRNN.code.analyses import common
    output = Path(output)
    groups = {}
    analysis_rows = []
    experiments = {row['experiment_id']:row for row in payload['experiment_df']}
    for task in payload['task_df']:
        key, outer, inner = task['model'], task['outer'], task['inner']
        folder = common.task_folder(output, task['index'])
        model = make_model(key)
        checkpoint = model_directory(folder,key)/'.checkpoints'/model.get_model_id()/f'outer_{outer+1:02d}_inner_{inner+1:02d}.pt'
        result = torch.load(checkpoint, map_location='cpu', weights_only=True)
        common.check_result(result)
        if result['inner_fold'] != inner+1:
            raise ValueError('Checkpoint inner fold differs from task plan')
        groups.setdefault((key,outer), []).append(result)
        inner_idx = inner if inner < outer else inner-1
        native = model_directory(folder,key)/f'outer_fold_{outer+1}'/f'inner_fold_{inner_idx}'
        row = dict(experiments[task['experiment_id']], task_id=task['index'], inner_loop_idx=inner_idx, completed=True)
        for field, suffix in [('info_path','info.json'),('model_pickle_path','model.pickle'),('trials_data_path','trials_data.htsv')]:
            row[field] = str((native/f'{model.get_model_id()}_{suffix}').relative_to(output))
        analysis_rows.append(row)
    rows = []
    data, folds, trainer = context['data'], context['folds'], context['trainer']
    for (key,outer), results in groups.items():
        if len(results) != K-1:
            raise ValueError('Missing inner-fold result')
        best = min(results, key=lambda row:(row['val_nll'],row['inner_fold']))
        model = make_model(key)
        model.load_state_dict(best['state_dict'])
        pooled = nll(model,data,folds[outer],trainer['batch_size'])
        subjects = {s:nll(model,data,[b for b in folds[outer] if data.block_subjects[b]==s],trainer['batch_size'])
                    for s in sorted(set(data.block_subjects))}
        row = dict(model=key,outer_fold=outer+1,**pooled,per_subject=subjects,
            mean_subject_nll=float(np.mean([v['held_out_nll'] for v in subjects.values()])),
            selected_inner_fold=best['inner_fold'],selected_val_nll=best['val_nll'],winning_config=best['config'])
        write_json(outer_path(output/'reduced',key,outer),row)
        rows.append(row)
    common.write_json(output/'reduced/analysis_df.json',analysis_rows)
    return pd.DataFrame(rows)


def analyse_results(results, output):
    report(Path(output),results.to_dict('records'))
    return pd.read_csv(Path(output)/'summary.csv')


if __name__=='__main__':
    main()
