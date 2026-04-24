'''Code to run training jobs in parallel on a HPC cluster via SLURM'''

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from joblib import Parallel, delayed


from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import datasets
from NM_TinyRNN.code.models import rnns
from NM_TinyRNN.code.models import nested


JOBS_PATH  = Path("./Jobs/NM_TinyRNN")
for jobs_folder in ["slurm", "out", "err"]:
    (JOBS_PATH/jobs_folder).mkdir(parents=True, exist_ok=True)
SAVE_PATH = Path('./NM_TinyRNN/data/rnns') #path to save out trained rnns.  
PROCESSED_DATA_PATH = Path('./NM_TinyRNN/data/AB_behaviour/') #subfolders here are subjects


def run_training(overwrite=False):
    '''Submit jobs to HPC cluster via slurm to run training'''
    train_df = get_DA_info_df()
    if not overwrite:
        train_df = train_df[train_df.completed==False]
    if train_df.empty:
        print("All files have been registered. No new videos to track.")
        return
    for session_info in train_df.itertuples():
        print(f"Submitting model training for {session_info.subject_ID} to HPC")
        script_path = get_NM_TinyRNN_SLURM_script(session_info)
        os.system(f"sbatch {script_path}")
    print("All NM_TinyRNN jobs submitted to HPC. Check progress with 'squeue -u <username>'")

# Parallel wrapper
def train_parallel(data_path, 
                   save_path,
                   model_type:str, 
                   hidden_size:int=2, 
                   nm_size:str=1, 
                   nm_dim:str=1, 
                   nm_mode:str=1,
                   input_encoding:str='unipolar',
                   nonlinearity:str='relu',
                   constraint:str='energy',
                   train_seed:int=42,
                   weight_seeds=[i for i in range(1,21)], 
                   n_jobs=-1, **kwargs):
    '''Parallelizes training over weight_seeds. n_jobs=-1 uses all cores.'''
    Parallel(n_jobs=n_jobs)(
        delayed(train_model_AB)(data_path, Path(save_path)/f'weight_seed_{seed}', 
                                model_type,hidden_size,
                                nm_size,nm_dim,nm_mode,
                                input_encoding,nonlinearity, constraint,
                                train_seed = train_seed,
                                weight_seeds=[seed])
        for seed in weight_seeds)

def train_model_AB(data_path, 
                   save_path,
                   model_type:str, 
                   hidden_size:int=2, 
                   nm_size:str=1, 
                   nm_dim:str=1, 
                   nm_mode:str=1,
                   input_encoding:str='unipolar',
                   nonlinearity:str='relu',
                   constraint:str='energy',
                   train_seed:int=42,
                   weight_seeds:list=[1,2,3,4,5]):
    '''Minimal inputs required to test fit all model types.'''
    options = rnns.OPTIONS_DICT
    options['rnn_type'] = model_type
    options['hidden_size'] = hidden_size
    options['nm_size'] = nm_size
    options['nm_dim'] = nm_dim
    options['nm_mode'] = nm_mode
    options['input_encoding'] = input_encoding
    options['nonlinearity'] = nonlinearity
    
    dataset = datasets.AB_Dataset(data_path)
    model = rnns.TinyRNN(**options)
    if constraint == 'sparsity':
        trainer = training.Trainer(save_path, 
                                   weight_seeds = weight_seeds,
                                   train_seed=train_seed,
                                   energy_lambdas=[0]) #sparsity constraint only
    else: 
        trainer = training.Trainer(save_path,
                                   weight_seeds=weight_seeds,
                                   train_seed=train_seed) #default range specified in training.py
    trainer.fit(model,dataset)
    return None


def get_NM_TinyRNN_SLURM_script(train_info, RAM="64GB", time_limit="23:59:00"):
    """
    Writes a SLURM script to run sleep tracking on the video from a session specified in video_info.
    Input: train_info: pd.Series
    Output: script_path: str, path to the SLURM script (saved in /jobs/slurm/)
    """
    session_ID = f"{train_info.model_id}"
    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN
#SBATCH --output={JOBS_PATH}/out/{session_ID}.out
#SBATCH --error={JOBS_PATH}/err/{session_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}
set -euo pipefail

echo "Node: $SLURMD_NODENAME"

# Properly initialize conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Debug info
which python
python --version
conda info --envs

# Activate your env
conda deactivate
conda deactivate
conda deactivate
conda activate /nfs/nhome/live/cburns/miniconda3/envs/NM_TinyRNN

echo "Running training"
python -c "from NM_TinyRNN.code.models import parallelised_training as pat; \
pat.train_parallel('{train_info.data_path}','{train_info.save_path}','{train_info.model_type}',{int(train_info.hidden_size)},{int(train_info.nm_size)},{int(train_info.nm_dim)},'{train_info.nm_mode}', '{train_info.input_encoding}', '{train_info.nonlinearity}','{train_info.constraint}',{int(train_info.train_seed)})"
"""
    script_path = JOBS_PATH/'slurm'/f'{session_ID}.sh'  
    with open(script_path, "w") as f:
        f.write(script)

    return script_path

def get_var1_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    
    Saving out this function for later use. This is the first run we did with WS18.
    Here there's a ton of hyperparameters.'''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        print(subdir)
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in [1,2,3,4,5,6,7,8,9,10]: #later add more seeds
            for model_type in ['GRU','monoGRU','monoGRU2','stereoGRU']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                for constraint in ['energy','sparsity']:
                    for nonlinearity in ['tanh','relu']:
                        for input_encoding in ['bipolar','unipolar']:
                            for hidden_size in [1,2]:
                                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                                #nmrnns are tricky since we're testing this.
                                if not model_type == 'NMRNN':
                                    model_id =  f'{hidden_size}_unit_{model_type}'
                                    model_save_path = save_path/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                                    completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                                    for k,v in zip(df_dict.keys(),
                                                [subject_ID,train_seed,model_type,hidden_size,
                                                 nonlinearity, input_encoding,constraint,
                                                nm_size,nm_dim,nm_mode,
                                                model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                                df_dict[k].append(v)
                                elif model_type == 'NMRNN':
                                    for nm_size, nm_dim in [(1,1),(2,1),(1,2),(2,2)]:
                                        for nm_mode in ['low_rank']: #later have a look at 'row' and 'column'
                                            if nm_dim>hidden_size:
                                                continue
                                            model_id = f'{hidden_size}_unit_{model_type}_{nm_size}_subunits_{nm_dim}_{nm_mode}'
                                            model_save_path = save_path/subject_ID/f'train_seed_{train_seed}'/model_type
                                            completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                                            for k,v in zip(df_dict.keys(),
                                                        [subject_ID,str(train_seed),model_type,hidden_size, 
                                                        nonlinearity, input_encoding,constraint,
                                                        nm_size,nm_dim,nm_mode,
                                                        model_id,model_save_path,data_path,completed]):
                                                df_dict[k].append(v)
    return pd.DataFrame(df_dict)



def get_var2_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Second run with just monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-2], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        print(subdir)
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in [1,2,3,4,5,6,7,8,9,10]: #later add more seeds
            for model_type in ['monoGRU']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                for constraint in ['energy']:
                    for nonlinearity in ['relu']:
                        for input_encoding in ['unipolar']:
                            for hidden_size in [2]:
                                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                                #nmrnns are tricky since we're testing this.
                                model_id =  f'{hidden_size}_unit_{model_type}'
                                model_save_path = save_path/'run_2'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                                completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                            [subject_ID,train_seed,model_type,hidden_size,
                                                nonlinearity, input_encoding,constraint,
                                            nm_size,nm_dim,nm_mode,
                                            model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                            df_dict[k].append(v)
                               
    return pd.DataFrame(df_dict)
    

def get_var3_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Second run with just monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-2], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        print(subdir)
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in range(1,51): #later add more seeds
            for model_type in ['monoGRU']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                for constraint in ['energy']:
                    for nonlinearity in ['relu']:
                        for input_encoding in ['unipolar']:
                            for hidden_size in [2]:
                                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                                #nmrnns are tricky since we're testing this.
                                model_id =  f'{hidden_size}_unit_{model_type}'
                                model_save_path = save_path/'run_3'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                                completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                            [subject_ID,train_seed,model_type,hidden_size,
                                                nonlinearity, input_encoding,constraint,
                                            nm_size,nm_dim,nm_mode,
                                            model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                            df_dict[k].append(v)
                               
    return pd.DataFrame(df_dict)
    
                    
def get_var4_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Second run with just monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-2], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in range(1,21): #later add more seeds
            for model_type in ['monoGRU','GRU']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                for constraint in ['energy','sparsity']:
                    nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                    input_encoding = 'unipolar'
                    for hidden_size in [1,2]:
                        nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                        #nmrnns are tricky since we're testing this.
                        model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                        model_save_path = save_path/'run_4'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                        completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                        for k,v in zip(df_dict.keys(),
                                    [subject_ID,train_seed,model_type,hidden_size,
                                        nonlinearity, input_encoding,constraint,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)
                            
    return pd.DataFrame(df_dict)
    
               
                    
def get_ws02_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Comparing standard GRU with monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-2], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        if not 'WS02' in subdir.stem:
            continue
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in range(1,21): #later add more seeds
            for model_type in ['monoGRU','GRU']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                constraint = 'energy' if model_type == 'monoGRU' else'sparsity'
                nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encoding = 'unipolar'
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for hidden_size in [1,2]:
                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                    model_save_path = save_path/'run_ws02'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                    completed = 1
                    for weight_seed in range(1,21):
                        completed *= (model_save_path/f'weight_seed_{weight_seed}'/f'{model_id}_trials_data.htsv').exists()
                    for k,v in zip(df_dict.keys(),
                        [subject_ID,train_seed,model_type,hidden_size,
                        nonlinearity, input_encoding,constraint,
                        nm_size,nm_dim,nm_mode,
                        model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                        df_dict[k].append(v)
                        
    return pd.DataFrame(df_dict)
                     
                    
def get_DA_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Comparing standard GRU with monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-3], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if not "WS" in subject_ID:
            continue
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in range(1,11): #later add more seeds
            for model_type in ['vanilla','monoGRU','GRU', 'constGate']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                nonlinearities = ['relu','tanh']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encoding = 'unipolar'
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for nonlinearity in nonlinearities:
                    constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                    for hidden_size in [1,2]:
                        model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                        model_save_path = save_path/'run_DA_again'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                        completed = 1
                        for weight_seed in range(1,21):
                            completed *= (model_save_path/f'weight_seed_{weight_seed}'/f'{model_id}_trials_data.htsv').exists()
                        for k,v in zip(df_dict.keys(),
                            [subject_ID,train_seed,model_type,hidden_size,
                            nonlinearity, input_encoding,constraint,
                            nm_size,nm_dim,nm_mode,
                            model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                            df_dict[k].append(v)
                            
    return pd.DataFrame(df_dict)

                    
                    
def get_MA_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''
    RUNNING ON MOHAMMED'S DATA
    Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.
    Comparing standard GRU with monoGRU - select hyperparameters set in training.py script
    energy_lambda = [1e-2], sparsity_lambda = [1e-5]. 
    Only variability across weight and training seeds.
    '''
    #see each subject
    df_dict = {'subject_ID':[],'train_seed':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if 'MA' not in subject_ID:
            continue
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for train_seed in range(1,21): #later add more seeds
            for model_type in ['monoGRU','GRU']: #['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                constraint = 'energy' if model_type == 'monoGRU' else'sparsity'
                nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encoding = 'unipolar'
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for hidden_size in [1,2]:
                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                    model_save_path = save_path/'run_MA_fast'/subject_ID/f'random_seed_{train_seed}'/model_type/constraint
                    completed = 1
                    for weight_seed in range(1,6):
                        completed *= (model_save_path/f'weight_seed_{weight_seed}'/f'{model_id}_trials_data.htsv').exists()
                    for k,v in zip(df_dict.keys(),
                        [subject_ID,train_seed,model_type,hidden_size,
                        nonlinearity, input_encoding,constraint,
                        nm_size,nm_dim,nm_mode,
                        model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                        df_dict[k].append(v)
                        
    return pd.DataFrame(df_dict)

    
                    
def run_training_per_subject(overwrite=False):
    '''Submit one SLURM job per subject, each running all model combinations in parallel.'''
    train_df = get_DA_info_df()
    if not overwrite:
        train_df = train_df[train_df.completed == False]
    if train_df.empty:
        print("All training completed. Nothing to submit.")
        return

    for subject_ID, subject_df in train_df.groupby('subject_ID'):
        print(f"Submitting job for subject {subject_ID} ({len(subject_df)} model configs) to HPC")
        script_path = get_subject_SLURM_script(subject_ID, subject_df)
        os.system(f"sbatch {script_path}")

    print("All subject jobs submitted. Check progress with 'squeue -u <username>'")


def get_subject_SLURM_script(subject_ID, subject_df, RAM="64GB", time_limit="23:59:00"):
    '''
    Writes a SLURM script that runs all model configurations for a single subject in parallel.
    Each config is launched as a parallel train_parallel() call via joblib.
    '''
    # Build a list of train_parallel() calls, one per row in subject_df
    python_calls = []
    for row in subject_df.itertuples():
        call = (
            f"delayed(pat.train_parallel)("
            f"'{row.data_path}', '{row.save_path}', '{row.model_type}', "
            f"{int(row.hidden_size)}, {int(row.nm_size)}, {int(row.nm_dim)}, "
            f"'{row.nm_mode}', '{row.input_encoding}', '{row.nonlinearity}', "
            f"'{row.constraint}', {int(row.train_seed)})"
        )
        python_calls.append(call)

    parallel_block = "Parallel(n_jobs=-1)([" + ", ".join(python_calls) + "])"

    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN_{subject_ID}
#SBATCH --output={JOBS_PATH}/out/{subject_ID}.out
#SBATCH --error={JOBS_PATH}/err/{subject_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

echo "Node: $SLURMD_NODENAME"
source "$(conda info --base)/etc/profile.d/conda.sh"

which python
python --version
conda info --envs

conda deactivate
conda deactivate
conda deactivate
conda activate /nfs/nhome/live/cburns/miniconda3/envs/NM_TinyRNN

echo "Running training for subject {subject_ID}"
python -c "
from joblib import Parallel, delayed
from NM_TinyRNN.code.models import parallelised_training as pat
{parallel_block}
"
"""
    script_path = JOBS_PATH / 'slurm' / f'{subject_ID}.sh'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path

def run_training_per_subject_seed(overwrite=False):
    '''Submit one SLURM job per subject per train_seed, each running all model combinations in parallel.'''
    train_df = get_DA_info_df()
    if not overwrite:
        train_df = train_df[train_df.completed == False]
    if train_df.empty:
        print("All training completed. Nothing to submit.")
        return

    for (subject_ID, train_seed), group_df in train_df.groupby(['subject_ID', 'train_seed']):
        print(f"Submitting job for subject {subject_ID}, train_seed {train_seed} ({len(group_df)} model configs) to HPC")
        script_path = get_subject_seed_SLURM_script(subject_ID, train_seed, group_df)
        os.system(f"sbatch {script_path}")

    print("All subject/seed jobs submitted. Check progress with 'squeue -u <username>'")


def get_subject_seed_SLURM_script(subject_ID, train_seed, subject_df, RAM="32GB", time_limit="23:59:00"):
    '''
    Writes a SLURM script that runs all model configurations for a single subject/train_seed.
    Calls train_parallel sequentially; parallelism over weight_seeds happens inside each call.
    '''
    python_calls = []
    for row in subject_df.itertuples():
        call = (
            f"pat.train_parallel("
            f"'{row.data_path}', '{row.save_path}', '{row.model_type}', "
            f"{int(row.hidden_size)}, {int(row.nm_size)}, {int(row.nm_dim)}, "
            f"'{row.nm_mode}', '{row.input_encoding}', '{row.nonlinearity}', "
            f"'{row.constraint}', {int(row.train_seed)})"
        )
        python_calls.append(call)

    python_block = "\n".join(python_calls)

    job_id = f"{subject_ID}_seed{train_seed}"
    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN_{job_id}
#SBATCH --output={JOBS_PATH}/out/{job_id}.out
#SBATCH --error={JOBS_PATH}/err/{job_id}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

echo "Node: $SLURMD_NODENAME"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda deactivate
conda deactivate
conda deactivate
conda activate /nfs/nhome/live/cburns/miniconda3/envs/NM_TinyRNN

echo "Running training for subject {subject_ID}, train_seed {train_seed}"
python -c "
from NM_TinyRNN.code.models import parallelised_training as pat
{python_block}
"
"""
    script_path = JOBS_PATH / 'slurm' / f'{job_id}.sh'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path