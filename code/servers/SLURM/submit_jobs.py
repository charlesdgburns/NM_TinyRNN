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
import shlex
from pathlib import Path
import pandas as pd
import numpy as np

from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import datasets
from NM_TinyRNN.code.models import rnns

from NM_TinyRNN.code.models import nested_cv

# global variables

JOBS_PATH  = Path("./Jobs/NM_TinyRNN")
SAVE_PATH = Path('./NM_TinyRNN/data/rnns') #path to save out trained rnns.  
PROCESSED_DATA_PATH = Path('./NM_TinyRNN/data/AB_behaviour/') #subfolders here are subjects


def training_folder_name(sequence_length=datasets.SEQUENCE_LENGTH,
                         batch_size=training.BATCH_SIZE, prefix='nested_DA'):
    """S = predictions per sequence; B = sequences per training minibatch."""
    sequence_length = datasets.validate_sequence_length(sequence_length)
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    return f'{prefix}_S{sequence_length}_B{batch_size}'


def _finish_job_info(df_dict, sequence_length, batch_size):
    frame = pd.DataFrame(df_dict)
    frame['sequence_length'] = sequence_length
    frame['batch_size'] = batch_size
    return frame


# functions

### top level ###


def run_training(overwrite=False, test = True, ws_all = False,
                 sequence_length=datasets.SEQUENCE_LENGTH,
                 batch_size=training.BATCH_SIZE, folder_name=None):
    """Submit training with shared sequence/batch defaults.

    Automatic output roots include S{sequence_length}_B{batch_size}; an explicit
    folder_name is used verbatim, including for legacy result lookup.
    """
    settings = dict(sequence_length=sequence_length, batch_size=batch_size, folder_name=folder_name)
    if ws_all:
        train_df = get_ws_all_info_df(**settings)
    elif test:
        train_df = get_test_info_df(**settings).iloc[0:2] #only run 2 jobs for testing
    else:
        train_df = get_DA_info_df(**settings)
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


def get_DA_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name = None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE, model_types=None):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    if model_types is not None:
        model_list = list(model_types)
    elif folder_name == 'nested_DA_128': # legacy analysis notebook architecture set
        model_list = ['vanilla','GRU', 'lightGRU', 'monoGRU','constGate','monoGRU_abs','monoGRU_no_hidden','monoGRU_hidden_only']
    else:
        model_list = ['GRU','monoGRU']

    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if not "WS" in subject_ID:
            continue
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue
        
        for outer_loop_n in range(1,nested_cv.N_OUTER_LOOPS+1): #10 loops is recommended
            
            for model_type in model_list:
                #Default parameters
                hidden_sizes = [1,2]
                input_encodings = ['unipolar']
                nonlinearities = ['relu']
                input_forced_choices = [True]
                decoder_biases = [True]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored

                if model_type == 'GRU':
                    nonlinearities = ['relu','tanh']
                    input_encodings = ['unipolar'] #'encoder','onehot'
                if model_type == 'vanilla':
                    nonlinearities = ['relu','tanh']
                    input_encodings = ['unipolar'] #'encoder','onehot'
                if  model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                    hidden_sizes = [1]
                if model_type in ['monoGRU_no_hidden','monoGRU_hidden_only']:
                    hidden_sizes = [2]
                    decoder_biases = [False]
                if model_type == 'monoGRU':
                    decoder_biases = [True,False]
                    
                for nonlinearity in nonlinearities:
                    for input_forced_choice in input_forced_choices:
                        for input_encoding in input_encodings:
                            for hidden_size in hidden_sizes:
                                if input_encoding == 'unipolar' and nonlinearity =='relu' and model_type in ['vanilla', 'GRU','monoGRU']:
                                    decoder_biases = [True,False] 
                                elif model_type in ['monoGRU_no_hidden','monoGRU_hidden_only']:
                                    decoder_biases = [False]
                                else:
                                    decoder_biases = [True]
                                for decoder_bias in decoder_biases:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'
                                        
                                    constraint = 'energy' if nonlinearity == 'relu' else 'sparsity'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
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
                                    
    return _finish_job_info(df_dict, sequence_length, batch_size)

def get_job_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
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
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue
        
        for outer_loop_n in range(1,nested_cv.N_OUTER_LOOPS+1): #10 loops is recommended
            for model_type in ['monoGRU','GRU','vanilla']:# 'vanilla', 'constGate', 'monoGRU_abs'  ['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                decoder_biases = [True,False]
                hidden_sizes = [1,2]
                nonlinearities = ['relu','tanh']
                #nonlinearity = 'relu' if constraint =='energy' else 'tanh'
                input_encodings = ['unipolar','onehot']
                input_forced_choices = [True,False]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                #if model_type in ['GRU','monoGRU']:
                #    input_forced_choices = [True, False]
                #    decoder_biases = [True,False]
                if model_type == 'monoGRU_abs':
                    nonlinearities = ['tanh']
                    hidden_sizes = [1]
                if model_type in ['constGate','lightGRU']:
                    input_encodings = ['unipolar']
                    nonlinearities = ['relu']
                    input_forced_choices = [True]
                    decoder_biases = [False]
                for nonlinearity in nonlinearities:
                    for input_forced_choice in input_forced_choices:
                        for input_encoding in input_encodings:
                            constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                            for hidden_size in hidden_sizes:
                                for decoder_bias in decoder_biases:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
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
                                    
    return _finish_job_info(df_dict, sequence_length, batch_size)

def get_test_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size)
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[],'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        if "WS" in subject_ID or "MA" in subject_ID: ##OBS - ONLY SIMULATED DATA FOR TESTING
            continue
        data_path = processed_data_path/subject_ID
        if not subdir.is_dir():
            continue
        
        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
            for model_type in ['GRU','monoGRU']: #,'constGate','vanilla'
                nonlinearities = ['relu','tanh']
                input_encodings = ['unipolar','onehot']
                decoder_biases = [True, False]
                input_forced_choices = [True, False]
                hidden_sizes = [1,2]
                nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                for input_forced_choice in input_forced_choices:
                    for nonlinearity in nonlinearities:
                        for input_encoding in input_encodings:
                            for decoder_bias in decoder_biases:
                                constraint= 'energy' if nonlinearity == 'relu' else 'sparsity'
                                for hidden_size in hidden_sizes:
                                    model_id =  f'{hidden_size}_unit_{model_type}_{nonlinearity}_{input_encoding}'
                                    if input_forced_choice:
                                        model_id+= '_forced'
                                    if decoder_bias == False:
                                        model_id+= '_ndb'
                                    model_save_path = Path(save_path)/folder_name/subject_ID/model_type/constraint
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
                                
    return _finish_job_info(df_dict, sequence_length, batch_size)


def get_ws_all_info_df(processed_data_path = PROCESSED_DATA_PATH,
                    save_path = SAVE_PATH, folder_name=None,
                    sequence_length=datasets.SEQUENCE_LENGTH,
                    batch_size=training.BATCH_SIZE):
    '''Organise the architecture information in a large dataframe. 
    Key arguments here are 'model_id', 'data_path','save_path' and 'outer_loop_n'
    '''
    default_folder = training_folder_name(sequence_length, batch_size, prefix="test_input_encodings")
    folder_name = default_folder if folder_name is None else folder_name
    #see each subject
    df_dict = {'subject_id':[],'outer_loop_n':[],'model_type':[],'hidden_size':[],
               'nonlinearity':[],'constraint':[], 'decoder_bias':[],
               'input_encoding':[],'input_forced_choice':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
    #generate a list of subjects to group together
    subject_ids = ['WS_all'] 
    for subject_id in subject_ids:
        data_path = processed_data_path/'WS'
        for outer_loop_n in range(1,11): #10 loops is recommended. Minimum is 3.
           for model_type in ['monoGRU','GRU', 'monoGRU_abs','stereoGRU','stereoGRU2','stereoGRUx']:#['vanilla','GRU','LSTM','NMRNN', 'monoGRU','monoGRU2','stereoGRU']:
                decoder_bias = True
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
                                model_save_path = Path(save_path)/folder_name/subject_id/model_type/constraint
                                completed = 1
                                for inner_loop_n in range(0,nested_cv.N_OUTER_LOOPS-1):
                                    completed *= (model_save_path/f'outer_fold_{outer_loop_n}'/f'inner_fold_{inner_loop_n}'/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                    [subject_id,outer_loop_n,model_type,hidden_size,
                                    nonlinearity, constraint, decoder_bias,
                                    input_encoding, input_forced_choice,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)
                            
    return _finish_job_info(df_dict, sequence_length, batch_size)


# SLURM functions to call the server # 


def get_NM_TinyRNN_SLURM_script(train_info, RAM="64GB", time_limit="23:59:00"):
    """
    Writes a SLURM script with explicit sequence length and minibatch size.
    Input: train_info: pd.Series
    Output: script_path: str, path to the SLURM script (saved in /jobs/slurm/)
    """
    sequence_length = getattr(train_info, 'sequence_length', datasets.SEQUENCE_LENGTH)
    batch_size = getattr(train_info, 'batch_size', training.BATCH_SIZE)
    training_folder_name(sequence_length, batch_size) # validate before writing scripts
    sequence_length, batch_size = int(sequence_length), int(batch_size)
    session_ID = f"{train_info.subject_id}_{train_info.model_id}_{train_info.constraint}_S{sequence_length}_B{batch_size}"
    for jobs_folder in ["slurm", "out", "err"]:
        (JOBS_PATH/jobs_folder).mkdir(parents=True, exist_ok=True)
    arguments = dict(data_path=str(train_info.data_path), save_path=str(train_info.save_path),
                     model_type=str(train_info.model_type), hidden_size=int(train_info.hidden_size),
                     nm_size=int(train_info.nm_size), nm_dim=int(train_info.nm_dim),
                     nm_mode=str(train_info.nm_mode), input_encoding=str(train_info.input_encoding),
                     input_forced_choice=bool(train_info.input_forced_choice),
                     nonlinearity=str(train_info.nonlinearity), constraint=str(train_info.constraint),
                     decoder_bias=bool(train_info.decoder_bias),
                     sequence_length=sequence_length, batch_size=batch_size)
    python_command = "from NM_TinyRNN.code.models import nested_cv as nc; nc.train_outers(" + ", ".join(
        f"{key}={value!r}" for key, value in arguments.items()) + ")"
    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN
#SBATCH --output={JOBS_PATH}/out/{session_ID}.out
#SBATCH --error={JOBS_PATH}/err/{session_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
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
python -c {shlex.quote(python_command)}
"""
    script_path = JOBS_PATH/'slurm'/f'{session_ID}.sh'  
    with open(script_path, "w") as f:
        f.write(script)

    return script_path
