''' Some code to plot dynamical analysis to investigate different RNNS'''

from importlib.resources import path
from joblib import Parallel, delayed
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests

from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import parallelised_training as pat

# Global variables # 

DATA_PATH = Path('./NM_TinyRNN/data/')
RNNS_PATH = DATA_PATH/'rnns' #this folder should contain a folder per subject, and then all models fit to said subject.

# functions #
# info_df
## Our approach for analyses will be based on pandas dataframes, first just getting one that points to relevant data.

def add_data_to_info_df(info_df, n_jobs=-1):
    '''Parallelised version - processes all model rows concurrently.'''
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_best_model_data)(row) for row in info_df.itertuples()
    ) # see the function get_best_model_data below.
    for each_data_column in results[0].keys():
        info_df[each_data_column] = [r[each_data_column] for r in results]
    return info_df


def get_best_model_data(each_model):
    '''Process a single model row - find best weight seed by val CE.'''
    empty_row = {k: None for k in ["info_path", "model_state_path", "training_losses_path",
                                    "trials_data_path", "eval_CE", "best_weight_seed", "best_val_CE"]}
    
    if each_model.completed == False:
        print(f"Model {each_model.model_id} has not completed training, skipping.")
        return empty_row

    train_seed_path = each_model.save_path
    best_val_CE = np.inf
    result = empty_row.copy()

    for each_weight_seed in train_seed_path.iterdir():
        if not each_weight_seed.is_dir():
            continue
        temp_info_path = each_weight_seed / f'{each_model.model_id}_info.json'
        info_dict = load_data(temp_info_path)
        val_CE = info_dict['best_val_pred_loss']
        if val_CE < best_val_CE:
            best_val_CE = val_CE
            result = {
                "info_path": temp_info_path,
                "model_state_path": each_weight_seed / f'{each_model.model_id}_model_state.pth',
                "training_losses_path": each_weight_seed / f'{each_model.model_id}_training_losses.htsv',
                "trials_data_path": each_weight_seed / f'{each_model.model_id}_trials_data.htsv',
                "eval_CE": info_dict['eval_pred_loss'],
                "best_weight_seed": str(each_weight_seed).split('_')[-1],
                "best_val_CE": best_val_CE,
            }
    return result

# functions for generating plots with a good overview #

# utilities #

def load_data(filepath):
    if isinstance(filepath, Path):
        filepath = str(filepath)
    if filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
    elif filepath.endswith(".htsv"):
        # assuming htsv = tab-separated values
        data = pd.read_csv(filepath, sep="\t")
    elif filepath.endswith(".pth"):
        data = torch.load(filepath, weights_only = True)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    return data