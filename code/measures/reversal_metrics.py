"""Here we define some measures on animals and RNNs around reversals"""

import numpy as np
import pandas as pd

from NM_TinyRNN.code.models import parallelised_training as pat
from NM_TinyRNN.code.measures import analysis

## The reversal trial (x=0) is the first trial pre reversal
#  so x=1 is when the probability of reward should drop.
import pandas as pd
import numpy as np

def add_reversal_columns(trials_df, good_poke_col='good_poke', N=20):
    '''
    Adds a relative trial index and reversal type (LR/RL) for a window 
    of [-N, N] trials around each reversal point.
    '''
    df = trials_df.copy()
    
    # 1. Identify Reversals: Where the target changes
    # .diff() gives 1.0 for 0->1 and -1.0 for 1->0
    df['rev_diff'] = df[good_poke_col].diff()
    df['is_reversal'] = df['rev_diff'].fillna(0) != 0
    
    # Initialize columns
    df['reversal_trial_index'] = np.nan
    df['reversal_type'] = np.nan
    
    # Get indices where reversals occur
    reversal_indices = df.index[df['is_reversal']].tolist()
    
    for rev_idx in reversal_indices:
        # Determine LR vs RL based on the difference
        # 1.0 (0->1) = LR | -1.0 (1->0) = RL
        diff_val = df.loc[rev_idx, 'rev_diff']
        rev_label = 'LR' if diff_val > 0 else 'RL'
        
        # Window from -N to +N (0 is the first trial of the new block)
        for offset in range(-N, N + 1):
            target_idx = rev_idx + offset
            
            if target_idx in df.index:
                df.loc[target_idx, 'reversal_trial_index'] = offset
                df.loc[target_idx, 'reversal_type'] = rev_label
                
    # Clean up the helper column before returning
    return df.drop(columns=['rev_diff'])


## OPTIMAL BEHAVIOUR ON A SIMULATED REVERSAL TASK ##


if __name__ == "__main__":        
    ## get a test dataframe
    info_df = pat.get_DA_info_df()
    info_df = analysis.add_data_to_info_df(info_df)
    best_monoGRU_idx = info_df.query("model_id == '2_unit_monoGRU_relu_unipolar'").eval_CE.idxmin()
    best_monoGRU_row = info_df.loc[best_monoGRU_idx]
    test_trials_data = analysis.load_data(best_monoGRU_row.trials_data_path)
