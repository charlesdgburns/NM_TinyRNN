"""Here we define some measures on animals and RNNs around reversals"""

import numpy as np
import pandas as pd

from NM_TinyRNN.code.models import parallelised_training as pat
from NM_TinyRNN.code.measures import analysis

## The reversal trial (x=0) is the first trial pre reversal
#  so x=1 is when the probability of reward should drop.

## get a test dataframe
info_df = pat.get_DA_info_df()
info_df = analysis.add_data_to_info_df(info_df)
best_monoGRU_idx = info_df.query("model_id == '2_unit_monoGRU_relu_unipolar'").eval_CE.idxmin()
best_monoGRU_row = info_df.loc[best_monoGRU_idx]
test_trials_data = analysis.load_data(best_monoGRU_row.trials_data_path)

def add_reversal_columns(trials_df, good_poke_col='good_poke', N=20):
    '''Add reversal indicator and trial index relative to reversal window.'''
    df = trials_df.copy()
    
    # Reversal occurs when good_poke changes
    df['is_reversal'] = df[good_poke_col] != df[good_poke_col].shift(1)
    df['is_reversal'].iloc[0] = False  # first trial is never a reversal
    
    # Trial index relative to reversal window
    reversal_indices = df.index[df['is_reversal']].tolist()
    
    df['reversal_trial_index'] = np.nan
    for rev_idx in reversal_indices:
        # x=0 is the last pre-reversal trial, x=1 is the reversal trial
        window = range(rev_idx - 1, rev_idx + N)  
        for trial_offset, trial_idx in enumerate(window):
            if trial_idx in df.index:
                df.loc[trial_idx, 'reversal_trial_index'] = trial_offset
    
    return df