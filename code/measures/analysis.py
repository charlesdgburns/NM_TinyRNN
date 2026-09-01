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
    Load model data in parallel with optional caching and add model_type2 column.
    
    Args:
        info_df: DataFrame with model information
        mode: 'all' to get all outer/inner combinations, 
              'best' to get best inner model for each outer loop
        n_jobs: Number of parallel jobs (-1 for all cores)
        use_cache: If True, load from cache if available (default True)
    
    Returns:
        DataFrame with model data and model_type2 column added
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
    
    # Compute the dataframe (model_type2 is computed inline in get_model_data)
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
            
            # Compute model_type2 inline
            model_type2 = _compute_model_type2(
                each_model.model_type,
                each_model.nonlinearity,
                each_model.decoder_bias
            )
            
            row = {
                **base_row,
                "model_type2": model_type2,
                "inner_loop_idx": inner_idx,
                "info_path": str(info_path),
                "model_state_path": str(inner_folder / f'{each_model.model_id}_model_state.pth'),
                "model_pickle_path": str(inner_folder / f'{each_model.model_id}_model.pickle'),
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
    elif filepath.endswith(".pickle"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    return data


class TrialsDataset(Dataset):
    '''
    Convert a trials_data DataFrame into a PyTorch Dataset.
    
    Takes a dataframe with columns ['forced_choice', 'choice', 'outcome', 'good_poke']
    and creates sequences of (inputs, targets, forced_choice_mask).
    '''
    def __init__(self, trials_df, sequence_length=64, device='cpu'):
        '''
        Args:
            trials_df: DataFrame with required columns ['forced_choice', 'choice', 'outcome', 'good_poke']
            sequence_length: Length of sequences to create
            device: Device to move tensors to ('cpu' or 'cuda')
        '''
        self.device = device
        self.sequence_length = sequence_length
        
        # Validate required columns
        required_cols = ['forced_choice', 'choice', 'outcome', 'good_poke']
        assert all(col in trials_df.columns for col in required_cols), \
            f"DataFrame must contain columns {required_cols}"
        
        # Encode categorical/boolean columns
        df = trials_df.copy()
        df['forced_choice'] = df['forced_choice'].astype(int)
        df['outcome'] = df['outcome'].astype(int)
        df['choice'] = df['choice'].astype('category').cat.codes.astype(int)
        df['good_poke'] = df['good_poke'].astype('category').cat.codes.astype(int)
        
        # Create tensor from forced_choice, choice, outcome
        data_tensor = torch.tensor(
            df[['forced_choice', 'choice', 'outcome']].values, 
            dtype=torch.float32
        )
        
        num_rows = data_tensor.size(0)
        remainder = num_rows % (self.sequence_length + 1)
        
        # Trim remainder to fit sequences evenly
        if remainder != 0:
            data_tensor = data_tensor[:-remainder]
        
        # Reshape into sequences
        num_sequences = data_tensor.size(0) // (self.sequence_length + 1)
        sequences = data_tensor.view(num_sequences, self.sequence_length + 1, data_tensor.size(1))
        
        # Create inputs (t) and targets (t+1)
        # Inputs: forced_choice, choice, outcome at time t
        inputs = sequences[:, :-1, :]  # (num_seq, seq_len, 3)
        
        # Targets: choice at time t+1, one-hot encoded
        targets_codes = sequences[:, 1:, 1].long()  # (num_seq, seq_len)
        targets = torch.nn.functional.one_hot(targets_codes, num_classes=2).float()
        
        # Mask: which targets come from forced choices
        # aligned with target (not current action), so we look at forced_choice at t+1
        forced_choice_mask = sequences[:, 1:, 0]  # (num_seq, seq_len)
        
        # Move to device and store
        self.inputs = inputs.to(device)
        self.targets = targets.to(device)
        self.forced_choice_mask = forced_choice_mask.to(device)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.forced_choice_mask[idx]


def trials_data_to_dataset(trials_df, split=None, sequence_length=64, device='cpu'):
    '''
    Convert a trials_data DataFrame into a TrialsDataset.
    
    Args:
        trials_df: DataFrame with required columns ['forced_choice', 'choice', 'outcome', 'good_poke']
        split: Optional string to filter by 'split' column (e.g., 'eval', 'train', 'val')
        sequence_length: Length of sequences to create (default 64)
        device: Device to move tensors to (default 'cpu')
    
    Returns:
        TrialsDataset object ready for training/evaluation
    
    Example:
        >>> eval_trials = trials_df.query('split == "eval"')
        >>> dataset = analysis.trials_data_to_dataset(eval_trials, device='cuda')
    '''
    df = trials_df.copy()
    
    if split is not None and 'split' in df.columns:
        df = df[df['split'] == split]
    
    return TrialsDataset(df, sequence_length=sequence_length, device=device)