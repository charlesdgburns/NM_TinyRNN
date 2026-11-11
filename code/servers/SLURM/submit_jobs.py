"""Generate and submit SLURM jobs from model experiment manifests.

Experiment definitions live in code.models.experiments. The builder names are
re-exported here for compatibility with existing submission/analysis callers.
"""
import os
import shlex
from pathlib import Path

from NM_TinyRNN.code.models import datasets, experiments, training
from NM_TinyRNN.code.models.experiments import (
    PROCESSED_DATA_PATH, SAVE_PATH, training_folder_name,
    get_DA_info_df, get_job_info_df, get_test_info_df, get_ws_all_info_df,
)

JOBS_PATH = Path("./Jobs/NM_TinyRNN")


def run_training(experiment_df=None, *, overwrite=False, test=True, ws_all=False,
                 sequence_length=datasets.SEQUENCE_LENGTH,
                 batch_size=training.BATCH_SIZE, folder_name=None):
    """Submit an experiment manifest, or build a legacy preset if omitted.

    A supplied experiment_df is used as provided: no preset is rebuilt and
    test/ws_all/sequence_length/batch_size/folder_name do not alter its rows.
    overwrite controls whether completed configurations are submitted again.
    Each generated job trains all outer folds, so dispatch uses outer-fold-1
    rows, preserving the existing whole-experiment submission behavior.
    SLURM resources remain arguments to get_NM_TinyRNN_SLURM_script.
    """
    if experiment_df is None:
        settings = dict(sequence_length=sequence_length, batch_size=batch_size,
                        folder_name=folder_name)
        if ws_all:
            experiment_df = experiments.get_ws_all_info_df(**settings)
        elif test:
            experiment_df = experiments.get_test_info_df(**settings).iloc[0:2]
        else:
            experiment_df = experiments.get_DA_info_df(**settings)
    if not overwrite:
        experiment_df = experiment_df[experiment_df.completed == False]
    experiment_df = experiment_df.query('outer_loop_n == 1')
    for experiment in experiment_df.itertuples():
        print(f"Submitting model training for {experiment.subject_id} to HPC")
        script_path = get_NM_TinyRNN_SLURM_script(experiment)
        os.system(f"sbatch {shlex.quote(str(script_path))}")
    print("All NM_TinyRNN jobs submitted to HPC. Check progress with 'squeue -u <username>'")


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
