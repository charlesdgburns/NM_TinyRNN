"""Here we define some measures on animals and RNNs around reversals"""

import numpy as np
import pandas as pd

from NM_TinyRNN.code.models import submit_jobs as pat
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.models import datasets as ds
## The reversal trial (x=0) is the first trial pre reversal
#  so x=1 is when the probability of reward should drop.
import pandas as pd
import numpy as np
import torch

def add_reversal_columns(
    trials_df,
    good_poke_col='good_poke',
    session_col='session_folder_name',
    N=20,
):
    '''
    Adds a relative trial index and reversal type (LR/RL) for a window 
    of [-N, N] trials around each reversal point.
    '''
    df = trials_df.copy()
    
    # 1. Identify reversals within sessions only.
    if session_col in df.columns:
        df['rev_diff'] = df.groupby(session_col, sort=False)[good_poke_col].diff()
    else:
        df['rev_diff'] = df[good_poke_col].diff()
    df['is_reversal'] = df['rev_diff'].fillna(0) != 0
    
    # Initialize columns
    df['reversal_trial_index'] = np.nan
    df['reversal_type'] = pd.Series(index=df.index, dtype='object')
    
    # Get positional indices where reversals occur.
    reversal_positions = np.flatnonzero(df['is_reversal'].to_numpy())
    
    for rev_position in reversal_positions:
        rev_idx = df.index[rev_position]
        # Determine LR vs RL based on the difference
        # 1.0 (0->1) = LR | -1.0 (1->0) = RL
        diff_val = df.loc[rev_idx, 'rev_diff']
        rev_label = 'LR' if diff_val > 0 else 'RL'
        session_value = (
            df.iloc[rev_position][session_col]
            if session_col in df.columns
            else None
        )
        
        # Window from -N to +N (0 is the first trial of the new block)
        for offset in range(-N, N + 1):
            target_position = rev_position + offset
            
            if 0 <= target_position < len(df):
                target_idx = df.index[target_position]
                if (
                    session_col in df.columns
                    and df.iloc[target_position][session_col] != session_value
                ):
                    continue
                df.loc[target_idx, 'reversal_trial_index'] = offset
                df.loc[target_idx, 'reversal_type'] = rev_label
                
    # Clean up the helper column before returning
    return df.drop(columns=['rev_diff'])


def summarise_reversal_behaviour(
    trials_df,
    good_poke_col='good_poke',
    choice_col='choice',
    prob_a_col='prob_A',
    prob_b_col='prob_B',
    N=20,
    return_trials=False,
):
    '''Summarise choice and good-side probability around LR and RL reversals.

    Returns mean ``choice_good`` and ``prob_good`` for each reversal type and
    relative trial index. ``choice_good`` is the proportion of choices of the
    currently good side, while ``prob_good`` is the model probability assigned
    to that side.
    '''
    df = add_reversal_columns(trials_df, good_poke_col=good_poke_col, N=N)

    if prob_a_col not in df.columns and 'prob_a' in df.columns:
        prob_a_col = 'prob_a'
    if prob_b_col not in df.columns and 'prob_b' in df.columns:
        prob_b_col = 'prob_b'

    df['choice_good'] = (
        df[choice_col].astype('boolean') == df[good_poke_col].astype('boolean')
    ).astype(float)
    df['prob_good'] = np.where(
        df[good_poke_col] == 0,
        df[prob_a_col],
        df[prob_b_col],
    )

    summary = (
        df.dropna(subset=['reversal_type', 'reversal_trial_index'])
        .groupby(['reversal_type', 'reversal_trial_index'], as_index=False)
        .agg(
            choice_good=('choice_good', 'mean'),
            prob_good=('prob_good', 'mean'),
            n=('choice_good', 'size'),
        )
        .sort_values(['reversal_type', 'reversal_trial_index'])
        .reset_index(drop=True)
    )

    if return_trials:
        return summary, df
    return summary


def collapse_reversal_data(
    trials_df,
    good_poke_col='good_poke',
    session_col='session_folder_name',
    choice_col='choice',
    prob_a_col='prob_A',
    prob_b_col='prob_B',
    N=20,
):
    '''Return trial-level reversal data pooled across LR and RL reversals.

    Choices and model probabilities are recoded relative to the arm that was
    good before the reversal, so the y-axis remains P(choose the initial best
    arm) throughout the reversal-aligned window.
    '''
    df = add_reversal_columns(trials_df, good_poke_col=good_poke_col, N=N)

    if prob_a_col not in df.columns and 'prob_a' in df.columns:
        prob_a_col = 'prob_a'
    if prob_b_col not in df.columns and 'prob_b' in df.columns:
        prob_b_col = 'prob_b'

    if session_col in df.columns:
        df[prob_a_col] = df.groupby(session_col, sort=False)[prob_a_col].shift(1)
        df[prob_b_col] = df.groupby(session_col, sort=False)[prob_b_col].shift(1)
    else:
        df[prob_a_col] = df[prob_a_col].shift(1)
        df[prob_b_col] = df[prob_b_col].shift(1)

    initial_best = (df['reversal_type'] == 'RL').astype('boolean')
    df['choice_initial'] = (
        df[choice_col].astype('boolean') == initial_best
    ).astype(float)
    df['prob_initial'] = np.where(
        initial_best == 0,
        df[prob_a_col],
        df[prob_b_col],
    )

    return df.dropna(
        subset=['reversal_trial_index']
    ).sort_values('reversal_trial_index').reset_index(drop=True)


def _run_model_on_trials_data(model,trials_data, fixed_gates = None):
    trials_data_new = trials_data.copy()

    raw = torch.tensor(
        trials_data[["forced_choice", "choice", "outcome"]].to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(0)
    inputs = ds.input_encoder(raw, model.input_encoding, model.input_forced_choice)

    model.eval()
    with torch.no_grad():
        if model.input_encoding == 'encoder':
            inputs = model.encoder(inputs)
        hidden_states, gate_activations = model.rnn(
            inputs, return_gate_activations=True,
            fixed_gates = fixed_gates
        )
        predictions = model.decoder(hidden_states)

    for each_hidden in range(hidden_states.shape[-1]):
        trials_data_new[f'hidden_{each_hidden+1}'] = hidden_states[0, :, each_hidden].cpu().numpy()

    for gate_name, gate_values in gate_activations.items():
        for unit in range(gate_values.shape[-1]):
            trials_data_new[f"gate_{gate_name}_{unit + 1}"] = (
                gate_values[0, :, unit].cpu().numpy()
            )

    log_probs = predictions.log_softmax(dim=2)
    trials_data_new["logit_value"] = (
        log_probs[0, :, 0] - log_probs[0, :, 1]
    ).cpu().numpy()
    trials_data_new["logit_past"] = np.concat([[np.nan], trials_data_new["logit_value"].values[:-1]])
    trials_data_new["logit_change"] = trials_data_new["logit_value"] - trials_data_new["logit_past"]
    trials_data_new["prob_A"] = log_probs[0, :, 0].exp().cpu().numpy()
    trials_data_new["prob_B"] = log_probs[0, :, 1].exp().cpu().numpy()
    return trials_data_new


## OPTIMAL BEHAVIOUR ON A SIMULATED REVERSAL TASK ##


if __name__ == "__main__":        
    ## get a test dataframe
    info_df = pat.get_DA_info_df()
    info_df = analysis.add_data_to_info_df(info_df)
    best_monoGRU_idx = info_df.query("model_id == '2_unit_monoGRU_relu_unipolar'").eval_CE.idxmin()
    best_monoGRU_row = info_df.loc[best_monoGRU_idx]
    test_trials_data = analysis.load_data(best_monoGRU_row.trials_data_path)
