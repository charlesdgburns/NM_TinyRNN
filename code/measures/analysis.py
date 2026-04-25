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


def get_analysis_df(info_df, mode='all', n_jobs=-1):
    '''
    Load model data in parallel.
    
    Args:
        info_df: DataFrame with model information
        mode: 'all' to get all outer/inner combinations, 
              'best' to get best inner model for each outer loop
        n_jobs: Number of parallel jobs (-1 for all cores)
    '''
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_model_data)(row, mode) for row in info_df.itertuples()
    )
    expanded_df = pd.DataFrame([r for result_list in results for r in result_list])
    return expanded_df if not expanded_df.empty else pd.DataFrame()
 
def get_model_data(each_model, mode='all'):
    '''Load data for model. Reads all inner folders once, filters by mode.'''
    if not each_model.completed:
        print(f"Model {each_model.model_id} has not completed training.")
        return []
    
    model_save_path = Path(each_model.save_path)
    outer_folder = model_save_path/ f"outer_fold_{each_model.outer_loop_n}"
    
    base_row = {col: getattr(each_model, col) for col in each_model._fields}
    all_rows = []
    best_per_outer = {}

    best_per_outer = None
    
    for inner_idx, inner_folder in enumerate(sorted(outer_folder.iterdir())):
        if not inner_folder.is_dir():
            continue
        
        info_path = inner_folder / f'{each_model.model_id}_info.json'
        if not info_path.exists():
            continue
        
        try:
            info_dict = load_data(info_path)
            winning_config = info_dict.get('winning_config', {})
            row = {
                **base_row,
                "inner_loop_idx": inner_idx,
                "info_path": str(info_path),
                "model_state_path": str(inner_folder / f'{each_model.model_id}_model_state.pth'),
                "training_losses_path": str(inner_folder / f'{each_model.model_id}_training_losses.htsv'),
                "trials_data_path": str(inner_folder / f'{each_model.model_id}_trials_data.htsv'),
                "eval_CE": info_dict.get('eval_pred_loss'),
                "best_val_CE": info_dict.get('val_loss'),
                "weight_seed": winning_config.get('weight_seed'),
                "sparsity_lambda": winning_config.get('sparsity_lambda'),
                "energy_lambda": winning_config.get('energy_lambda')
            }
            all_rows.append(row)
            
            # Track best for this outer loop
            val_loss = info_dict.get('val_loss', float('inf'))
            if best_per_outer is None or val_loss < best_per_outer['best_val_CE']:
                best_per_outer = row
        except Exception as e:
            print(f"Error processing {info_path}: {e}")

    if mode == 'all':
        return all_rows
    elif mode == 'best':
        return [row for row in best_per_outer.values() if row is not None]
    else:
        raise ValueError(f"Unknown mode: {mode}")

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