'''Code to submit jobs for nested loops. 
Essentially parallelised training on steroids here.
Different architectures: parallelised on different nodes of HPC
For each architecture, the outer loop is also parallelised on different nodes.
Within each outer loop, the inner loops are parallelised across CPU cores
and within each inner loop, the hyperparameters are parallelised using vectors.

It's fast. 
'''
# imports
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import datasets
from NM_TinyRNN.code.models import rnns

from NM_TinyRNN.code.models import nested_cv
from NM_TinyRNN.code.models import training_fast

# global variables
N_OUTER_LOOPS = 10

JOBS_PATH  = Path("./Jobs/NM_TinyRNN")
for jobs_folder in ["slurm", "out", "err"]:
    (JOBS_PATH/jobs_folder).mkdir(parents=True, exist_ok=True)
SAVE_PATH = Path('./NM_TinyRNN/data/rnns') #path to save out trained rnns.  
PROCESSED_DATA_PATH = Path('./NM_TinyRNN/data/AB_behaviour/') #subfolders here are subjects


# functions

### top level ###


def run_training(overwrite=False, test = True):
    '''Submit jobs to HPC cluster via slurm to run training'''
    train_df = get_job_info_df()
    if not overwrite:
        train_df = train_df[train_df.completed==False]
    if train_df.empty:
        print("All files have been registered. No new videos to track.")
        return
    if test == True:
       train_df = train_df.query('subject_ID=="WS16" and model_type == "monoGRU"')
    #Computing outer loops sequentially:
    train_df = train_df.drop_duplicates(['subject_ID','model_id'])
    for session_info in train_df.itertuples():
        print(f"Submitting model training for {session_info.subject_ID} to HPC")
        script_path = get_NM_TinyRNN_SLURM_script(session_info)
        os.system(f"sbatch {script_path}")
    print("All NM_TinyRNN jobs submitted to HPC. Check progress with 'squeue -u <username>'")
    

def get_job_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    #see each subject
    df_dict = {'subject_ID':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
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
        
        for outer_loop_n in range(1,N_OUTER_LOOPS+1): #10 loops is recommended
            for model_type in ['vanilla','monoGRU','GRU', 'constGate']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                nonlinearities = ['relu','tanh']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encoding = 'unipolar'
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for nonlinearity in nonlinearities:
                    constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                    for hidden_size in [1,2]:
                        model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                        model_save_path = save_path/'nested_DA'/subject_ID/model_type/constraint
                        completed = 1
                        for inner_loop_n in range(0,N_OUTER_LOOPS-1):
                            completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                        for k,v in zip(df_dict.keys(),
                            [subject_ID,outer_loop_n,model_type,hidden_size,
                            nonlinearity, input_encoding,constraint,
                            nm_size,nm_dim,nm_mode,
                            model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                            df_dict[k].append(v)
                            
    return pd.DataFrame(df_dict)


# top level training function # 


# SLURM functions to call the server # 

def train_outers(data_path, 
                   save_path,
                   model_type:str, 
                   hidden_size:int=2, 
                   nm_size:str=1, 
                   nm_dim:str=1, 
                   nm_mode:str=1,
                   input_encoding:str='unipolar',
                   nonlinearity:str='relu',
                   constraint:str='energy',):
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
        trainer_kwargs ={'energy_lambdas':[0]}
    else: 
        trainer_kwargs = {}
    for outer_loop_n in range(1, N_OUTER_LOOPS+1):
        nested_cv.run_outer_fold(model, dataset, outer_loop_n,
                                n_outer_loops = N_OUTER_LOOPS,
                                save_path = save_path, trainer_kwargs = trainer_kwargs)
    
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
python -c "from NM_TinyRNN.code.models import nested_jobs as nj; \
nj.train_outers('{train_info.data_path}','{train_info.save_path}','{train_info.model_type}',{int(train_info.hidden_size)},{int(train_info.nm_size)},{int(train_info.nm_dim)},'{train_info.nm_mode}', '{train_info.input_encoding}', '{train_info.nonlinearity}','{train_info.constraint}')"
"""
    script_path = JOBS_PATH/'slurm'/f'{session_ID}.sh'  
    with open(script_path, "w") as f:
        f.write(script)

    return script_path
