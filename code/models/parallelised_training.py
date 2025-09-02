'''Code to run training jobs in parallel on a HPC cluster via SLURM'''

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np


from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import datasets
from NM_TinyRNN.code.models import rnns

JOBS_PATH  = Path("./Jobs/NM_TinyRNN")
for jobs_folder in ["slurm", "out", "err"]:
    (JOBS_PATH/jobs_folder).mkdir(parents=True, exist_ok=True)
SAVE_PATH = Path('./NM_TinyRNN/data/rnns') #path to save out trained rnns.  
PROCESSED_DATA_PATH = Path('./NM_TinyRNN/data/AB_behaviour/') #subfolders here are subjects


def run_training(overwrite=False):
    '''Submit jobs to HPC cluster via slurm to run training'''
    train_df = get_train_info_df()
    if not overwrite:
        train_df = train_df[~train_df.completed]
    if train_df.empty:
        print("All files have been registered. No new videos to track.")
        return
    for session_info in train_df.itertuples():
        print(f"Submitting model training for {session_info.subject_ID} to HPC")
        script_path = get_NM_TinyRNN_SLURM_script(session_info)
        os.system(f"sbatch {script_path}")
    print("All NM_TinyRNN jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def train_model_AB(data_path, 
                   save_path,
                   model_type:str, 
                   hidden_size:int, 
                   nm_size:str=1, 
                   nm_dim:str=1, 
                   nm_mode:str=1,
                   random_seed:int=1):
    '''Minimal inputs required to test fit all model types.'''
    dataset = datasets.AB_Dataset(data_path)
    model = rnns.TinyRNN(rnn_type = model_type, # GRU, LSTM, NMRNN, vanilla,
                        input_size=3, # past forced choice, past choice, past outcome, 
                        hidden_size=hidden_size, # hidden unit
                        out_size=2, # one-hot code for choice A, choice B
                        nm_mode = nm_mode, nm_dim=nm_dim, nm_size=nm_size,)
    trainer = training.Trainer(save_path, random_seed=random_seed)
    trainer.fit(model,dataset)
    return None


def get_NM_TinyRNN_SLURM_script(train_info, RAM="64GB", time_limit="23:59:00"):
    """
    Writes a SLURM script to run sleap tracking on the video from a session specified in video_info.
    Input: train_info: pd.Series
    Output: script_path: str, path to the SLURM script (saved in /jobs/slurm/)
    """
    session_ID = f"{train_info.model_id}"
    script = f"""#!/bin/bash
#SBATCH --job-name=NM_TinyRNN{session_ID}
#SBATCH --output={JOBS_PATH}/out/{session_ID}.out
#SBATCH --error={JOBS_PATH}/err/{session_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}
set -euo pipefail

echo "Node: $SLURMD_NODENAME"

# Load conda module (cluster-specific name/version)
source /etc/profile.d/modules.sh
module load miniconda

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
pat.train_model_AB('{train_info.data_path}','{train_info.save_path}','{train_info.model_type}',{int(train_info.hidden_size)},{int(train_info.nm_size)},{int(train_info.nm_dim)},'{train_info.nm_mode}', {train_info.random_seed})"
"""
    script_path = JOBS_PATH/'slurm'/f'{session_ID}.sh'
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def get_train_info_df(processed_data_path = PROCESSED_DATA_PATH, save_path = SAVE_PATH):
    '''Builds a dataframe with each subject and for each model combination.
    Checks whether a model combination has been run before or not.'''
    #see each subject
    df_dict = {'subject_ID':[],'random_seed':[],'model_type':[],'hidden_size':[],
               'nm_size':[],'nm_dim':[],'nm_mode':[],
               'model_id':[],'save_path':[],'data_path':[], 'completed':[]}
   
    for subdir in processed_data_path.iterdir():
        subject_ID = subdir.stem
        data_path = PROCESSED_DATA_PATH/subject_ID
        if not subdir.is_dir():
            continue
        for random_seed in [1,2,3,4,5,6,7,8,9,10]: #later add more seeds
            for model_type in ['vanilla','GRU','LSTM','NMRNN']:
                for hidden_size in [1,2]:
                    nm_size = nm_dim = 1; nm_mode = 'row' # simply standard inputs which will get ignored
                    #nmrnns are tricky since we're testing this.
                    if not model_type == 'NMRNN':
                        model_id =  f'{hidden_size}_unit_{model_type}'
                        model_save_path = save_path/f'random_seed_{random_seed}'/model_type
                        completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                        for k,v in zip(df_dict.keys(),
                                    [subject_ID,random_seed,model_type,hidden_size,
                                    nm_size,nm_dim,nm_mode,
                                    model_id,model_save_path,data_path,completed]): #NaN all the nm stuff
                                    df_dict[k].append(v)
                    elif model_type == 'NMRNN':
                        for nm_size, nm_dim in [(1,1),(2,1),(1,2),(2,2)]:
                            for nm_mode in ['low_rank']: #later have a look at 'row' and 'column'
                                if nm_dim>hidden_size:
                                    continue
                                model_id = f'{hidden_size}_unit_{model_type}_{nm_size}_subunits_{nm_dim}_{nm_mode}'
                                model_save_path = save_path/subject_ID/f'random_seed_{random_seed}'/model_type/nm_mode
                                completed = (model_save_path/f'{model_id}_trials_data.htsv').exists()
                                for k,v in zip(df_dict.keys(),
                                            [subject_ID,str(random_seed),model_type,hidden_size, 
                                            nm_size,nm_dim,nm_mode,
                                            model_id,model_save_path,data_path,completed]):
                                    df_dict[k].append(v)
        return pd.DataFrame(df_dict)

    
                    
    