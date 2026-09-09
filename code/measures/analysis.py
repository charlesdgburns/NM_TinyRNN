''' Some code to plot dynamical analysis to investigate different RNNS'''

from importlib.resources import path
from joblib import Parallel, delayed
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests

from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models import submit_jobs as pat

# Global variables # 

DATA_PATH = Path('./NM_TinyRNN/data/')
RNNS_PATH = DATA_PATH/'rnns' #this folder should contain a folder per subject, and then all models fit to said subject.

# functions #
# info_df
## Our approach for analyses will be based on pandas dataframes, first just getting one that points to relevant data.


def get_analysis_df(info_df, mode='all', n_jobs=-1, use_cache=True):
    '''
    Load model data in parallel with optional caching and add computed columns.
    
    Computes for each model:
    - model_type2: model type with '+BC' suffix if relu, '-DB' if no decoder bias
    - train_CE, val_CE, eval_CE_computed: cross-entropy losses on each split
    
    Args:
        info_df: DataFrame with model information
        mode: 'all' to get all outer/inner combinations, 
              'best' to get best inner model for each outer loop
        n_jobs: Number of parallel jobs (-1 for all cores)
        use_cache: If True, load from cache if available (default True)
    
    Returns:
        DataFrame with model data and computed columns. Note: eval_CE_computed
        is computed from trials_data with continuous hidden state (no resets),
        while eval_CE is from training with batch-wise resets. Typically
        eval_CE_computed is ~0.05 higher due to this difference.
    '''
    # Cache path
    cache_path = DATA_PATH / 'analysis' / 'analysis_df.htsv'
    
    # Try to load from cache if use_cache is True
    if use_cache and cache_path.exists():
        try:
            expanded_df = pd.read_csv(cache_path, sep='\t')
            print(f"Loaded analysis_df from cache: {cache_path}")
            return expanded_df
        except Exception as e:
            print(f"Warning: Could not load from cache: {e}. Recomputing...")
    
    # Compute the dataframe (model_type2 and CE values computed inline in get_model_data)
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_model_data)(row, mode) for row in info_df.itertuples()
    )
    expanded_df = pd.DataFrame([r for result_list in results for r in result_list])
    
    # Save to cache
    if not expanded_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        expanded_df.to_csv(cache_path, sep='\t', index=False)
        print(f"Saved analysis_df to cache: {cache_path}")
    
    return expanded_df if not expanded_df.empty else pd.DataFrame()


def _compute_model_type2(model_type, nonlinearity, decoder_bias):
    '''Compute model_type2 string inline.'''
    if nonlinearity == 'relu':
        model_type += '+BC'
    if decoder_bias == False:
        model_type += '-DB'
    return model_type


 
def get_model_data(each_model, mode='all'):
    '''Load data for model. Reads all inner folders once, filters by mode.
    
    Also computes cross-entropy losses for all three splits (train, val, eval)
    from the trials_data file.
    '''
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
        trials_data_path = inner_folder / f'{each_model.model_id}_trials_data.htsv'
        
        if not info_path.exists():
            continue
        
        try:
            info_dict = load_data(info_path)
            winning_config = info_dict.get('winning_config', {})
            
            # Compute model_type2 inline
            model_type2 = _compute_model_type2(
                each_model.model_type,
                each_model.nonlinearity,
                each_model.decoder_bias
            )
            
            # Compute cross-entropy for all splits from trials_data
            train_ce = float("nan")
            val_ce = float("nan")
            eval_ce = float("nan")
            
            if trials_data_path.exists():
                try:
                    trials_df = load_data(trials_data_path)
                    train_ce = cross_entropy_from_trials_df(trials_df, 'train')
                    val_ce = cross_entropy_from_trials_df(trials_df, 'val')
                    eval_ce = cross_entropy_from_trials_df(trials_df, 'eval')
                    train_val_ce = cross_entropy_from_trials_df(trials_df, 'train_val')
                except Exception as e:
                    print(f"Warning: Could not compute CE for {trials_data_path}: {e}")
            
            row = {
                **base_row,
                "model_type2": model_type2,
                "inner_loop_idx": inner_idx,
                "info_path": str(info_path),
                "model_state_path": str(inner_folder / f'{each_model.model_id}_model_state.pth'),
                "model_pickle_path": str(inner_folder / f'{each_model.model_id}_model.pickle'),
                "training_losses_path": str(inner_folder / f'{each_model.model_id}_training_losses.htsv'),
                "trials_data_path": str(trials_data_path),
                "eval_CE": info_dict.get('eval_pred_loss'),
                "best_val_CE": info_dict.get('val_loss'),
                "weight_seed": winning_config.get('weight_seed'),
                "sparsity_lambda": winning_config.get('sparsity_lambda'),
                "energy_lambda": winning_config.get('energy_lambda'),
                "train_CE": train_ce,
                "val_CE": val_ce,
                "train_val_CE":train_val_ce,
                "eval_CE_computed": eval_ce
                
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
    elif filepath.endswith(".pickle"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    return data

# recomputing cross entropy for trian and validation splits #

def cross_entropy_from_trials_df(trials_df: pd.DataFrame, split: str) -> float:
    """
    Compute cross-entropy loss on a specified split (or aggregated splits) 
    from trial-by-trial predictions. Excludes forced-choice trials from the loss calculation.
    
    Equivalent to rnns.py compute_losses() which uses:
        free_choice = (forced_choice_mask==0)
        predictions = predictions[free_choice]
        targets = targets[free_choice]
    
    Note: probs[i] predict choice[i+1], so we SHIFT FIRST on the full 
    dataset, then filter by split to maintain temporal continuity.
    
    Hidden state is NOT reset between sequences in trials_data (unlike training),
    so this loss represents performance under continuous session conditions.
    
    Parameters
    ----------
    trials_df : DataFrame with columns 'split', 'forced_choice', 'choice', 'prob_A', 'prob_B'
    split : str, either 'train', 'val', 'eval', or 'train_val' (to aggregate train + val)
    
    Returns
    -------
    float : mean cross-entropy loss on free-choice trials in the specified split(s), or NaN if empty
    """
    df = trials_df.reset_index(drop=True)
    
    if len(df) < 2:
        return float("nan")
    
    # Shift to align: probs[i] predict choice[i+1]
    targets = torch.tensor(df['choice'].values[1:], dtype=torch.long)
    probs = torch.tensor(df[['prob_A', 'prob_B']].values[:-1], dtype=torch.float32)
    
    # Extract forced_choice status aligned with targets (time t+1)
    forced_mask = df['forced_choice'].values[1:].astype(int)
    split_labels = df['split'].values[1:]
    
    # Filter to specified split only (on the TARGET trials)
    # Check for individual splits or the aggregated 'train_val' option
    if split == 'train_val':
        split_idx = (split_labels == 'train') | (split_labels == 'val')
    else:
        split_idx = (split_labels == split)
    
    if not split_idx.any():
        return float("nan")
    
    targets = targets[split_idx]
    probs = probs[split_idx]
    forced_mask = forced_mask[split_idx]
    
    # Further filter to free-choice trials only
    free_choice_mask = (forced_mask == 0)
    
    if not free_choice_mask.any():
        return float("nan")
    
    targets = targets[free_choice_mask]
    probs = probs[free_choice_mask]
    
    # Cross-entropy loss (default is mean reduction)
    loss = F.cross_entropy(probs, targets)
    return loss.item()

def select_best_outer(analysis_df):
    # --- SELECT MODEL WITH HIGHEST (val_CE + train_CE) ACROSS INNER FOLDS ---
    group_cols = ['model_type2', 'hidden_size', 'subject_id', 'outer_loop_n']
    
    # 3. Find indices corresponding to max total_CE across inner_fold_idx
    min_idx = select_models.groupby(group_cols)['train_val_CE'].idxmin()
    
    # 4. Filter DataFrame down to the selected inner fold runs
    select_models = select_models.loc[min_idx].drop(columns=['train_val_CE'])
    return select_models