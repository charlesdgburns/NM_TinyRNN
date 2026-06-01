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

# global variables

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
    if test == True:
       train_df = get_ws_all_info_df()
    if not overwrite:
        train_df = train_df[train_df.completed==False]
    #Computing outer loops sequentially:
    train_df = train_df.query('outer_loop_n==1') 
    for session_info in train_df.itertuples():
        print(f"Submitting model training for {session_info.subject_id} to HPC")
        script_path = get_NM_TinyRNN_SLURM_script(session_info)
        os.system(f"sbatch {script_path}")
    print("All NM_TinyRNN jobs submitted to HPC. Check progress with 'squeue -u <username>'")
    

# info dictionaries for training # 

def get_job_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if not "WS" in subject_ID:
            continue
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        
        for outer_loop_n in range(1,nested_cv.N_OUTER_LOOPS+1): #10 loops is recommended
            for model_type in ['monoGRU','GRU',]:# 'vanilla', 'constGate', 'monoGRU_abs'  ['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                decoder_biases = [True,False]
                hidden_sizes = [1,2]
                nonlinearities = ['relu','tanh']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encodings = ['unipolar','onehot']
                input_forced_choice = True
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                if model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                    hidden_sizes = [1]
                for nonlinearity in nonlinearities:
                    for input_encoding in input_encodings:
                        constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                        for hidden_size in hidden_sizes:
                            for decoder_bias in decoder_biases:
                                model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                if input_forced_choice:
                                    model_id+= '_forced'
                                if decoder_bias == False:
                                    model_id+= '_ndb'
                                model_save_path = save_path/'nested_DA'/subject_ID/model_type/constraint
                                completed = 1
                                for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                    completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                    [subject_ID,outer_loop_n,model_type,hidden_size,
                                    nonlinearity,constraint,decoder_bias,
                                    input_encoding, input_forced_choice,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)
                                
    return pd.DataFrame(df_dict)

def get_test_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'input_encoding':[],'constraint':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if "WS" in subject_ID or "MA" in subject_ID: ##OBS - ONLY SIMULATED DATA FOR TESTING
            continue
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        
        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
            for model_type in ['GRU','monoGRU']: #,'constGate','vanilla'
                nonlinearities = ['relu','tanh']
                input_encodings = ['unipolar','onehot']
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for nonlinearity in nonlinearities:
                    for input_encoding in input_encodings:
                        constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                        for hidden_size in [1,2]:
                            model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                            model_save_path = save_path/'test'/subject_ID/model_type/constraint
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


def get_ws_all_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
    #generate a list of subjects to group together
    subject_ids = ['WS_all'] 
    for subject_id in subject_ids:
        data_path = processed_data_path/'WS'
        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
           for model_type in ['monoGRU','GRU', 'monoGRU_abs','stereoGRU','stereoGRU2','stereoGRUx']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                hidden_sizes = [1,2]
                nonlinearities = ['relu','tanh', 'softplus']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encodings = ['unipolar','onehot']
                input_forced_choice = False
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                if model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                for nonlinearity in nonlinearities:
                    for input_encoding in input_encodings:
                        for input_forced_choice in [True, False]:
                            constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                            for hidden_size in hidden_sizes:
                                model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                if input_forced_choice:
                                    model_id+='_forced'
                                model_save_path = save_path/'no_decoder_bias'/subject_id/model_type/constraint
                                completed = 1
                                for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                    completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                    [subject_id,outer_loop_n,model_type,hidden_size,
                                    nonlinearity, constraint,
                                    input_encoding, input_forced_choice,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)
                            
    return pd.DataFrame(df_dict)


# SLURM functions to call the server # 


def get_NM_TinyRNN_SLURM_script(train_info, RAM="64GB", time_limit="23:59:00"):
    """
    Writes a SLURM script to run sleep tracking on the video from a session specified in video_info.
    Input: train_info: pd.Series
    Output: script_path: str, path to the SLURM script (saved in /jobs/slurm/)
    """
    session_ID = f"{train_info.model_id}"
    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN
#SBATCH --output={JOBS_PATH}/out/{train_info.subject_id}_{session_ID}.out
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
python -c "from NM_TinyRNN.code.models import nested_cv as nc; \
nc.train_outers('{train_info.data_path}',\
                '{train_info.save_path}',\
                '{train_info.model_type}',\
                {int(train_info.hidden_size)},\
                {int(train_info.nm_size)},\
                {int(train_info.nm_dim)},\
                '{train_info.nm_mode}', \
                '{train_info.input_encoding}', \
                {train_info.input_forced_choice}, \
                '{train_info.nonlinearity}',\
                '{train_info.constraint}')"
"""
    script_path = JOBS_PATH/'slurm'/f'{session_ID}.sh'  
    with open(script_path, "w") as f:
        f.write(script)

    return script_path
