""" 
The following script contains code to perform an analysis of mechanistic variability.
This is written for the 2-armed bandit reversal task modelled trial-by-trial.

1. train models on a subjects with multiple train seeds and weight seeds.
2. compare evaluation performance as across train and weight seeds.
3. compare similarity of activations (hidden units and gates).
4. compare the parameters (weights) of models.

Note that comparisons must be somehow aligned in the hidden unit activations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## local imports
from NM_TinyRNN.code.models import parallelised_training as pat
from NM_TinyRNN.code.measures import analysis

# GLOBAL VARIABLES 
AB_DATA_PATH = Path("NM_TinyRNN/data/AB_behaviour")
SAVE_PATH = Path("NM_TinyRNN/data/rnns/mech_var")

# --- FUNCTIONS --- #

def train_models(train_seeds=list(range(1, 3)),
                 weight_seeds=list(range(1, 11)),
                 subjects=["WS16"]):
    """Performs parallel training of RNN models."""
    for each_subject in subjects:
        data_path = AB_DATA_PATH / f"{each_subject}"
        for each_train_seed in train_seeds:
            for each_model_type in ['monoGRU']:
                save_path = SAVE_PATH / f"{each_subject}/train_seed_{each_train_seed}"
                
                # Train with 'biological constraints'
                pat.train_parallel(
                    data_path=data_path,
                    save_path=save_path,
                    train_seed=each_train_seed,
                    weight_seeds=weight_seeds,
                    n_jobs=-1,
                    model_type=each_model_type,
                    nonlinearity='relu',
                    constraint='energy'
                )
    print("Training complete.")
    return None

def build_analysis_df(save_path=SAVE_PATH):
    """Iterates over the saved models and builds a dataframe for analysis."""
    analysis_path = Path(save_path)
    data_rows = []

    for path in analysis_path.glob("*/*/*"):
        if path.is_dir():
            subject_id = path.parts[-3]
            train_seed = path.parts[-2][-1]
            weight_seed = path.parts[-1].split('_')[-1]

            info_files = list(path.glob("*_info.json"))
            for info_file in info_files:
                model_id = info_file.name.replace("_info.json", "")
                data_rows.append({
                    "subject_id": subject_id,
                    "train_seed": train_seed,
                    "weight_seed": weight_seed,
                    "model_id": model_id,
                    "info_path": path / f"{model_id}_info.json",
                    "model_state_path": path / f"{model_id}_model_state.pth",
                    "training_losses_path": path / f"{model_id}_training_losses.htsv",
                    "trials_data_path": path / f"{model_id}_trials_data.htsv"
                })
    return pd.DataFrame(data_rows)

def add_data(analysis_df):
    """Extracts evaluation metrics and hyperparameters from info files."""
    evaluation_CEs, validation_CEs, sparsity_lambdas, energy_lambdas = [], [], [], []
    
    for each_row in analysis_df.itertuples():
        info_dict = analysis.load_data(str(each_row.info_path))
        evaluation_CEs.append(info_dict['eval_pred_loss'])
        validation_CEs.append(info_dict['best_val_pred_loss'])
        sparsity_lambdas.append(info_dict['options_dict']['sparsity_lambda'])
        energy_lambdas.append(info_dict['options_dict']['energy_lambda'])

    analysis_df['eval_CE'] = evaluation_CEs
    analysis_df['val_CE'] = validation_CEs
    analysis_df['sparsity_lambda'] = sparsity_lambdas
    analysis_df['energy_lambda'] = energy_lambdas
    return analysis_df

COMPONENTS = ['hidden', 'gate_update', 'gate_reset', 'logit_value']

def compute_similarities(analysis_df, n_jobs=-1):
    """Compute similarity of activations within each (model_id, subject_id) group, in parallel."""
    groups = list(analysis_df.groupby(['model_id', 'subject_ID']))
 
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_group)(group_rows.reset_index(drop=True), model_id, subject_id)
        for (model_id, subject_id), group_rows in groups
    )
 
    results = [r for r in results if r is not None]
    return pd.DataFrame([row for result in results for row in result]) if results else pd.DataFrame()
 
 
def _process_group(group_rows, model_id, subject_id):
    """
    For a single (model_id, subject_id):
    - Builds a (n_trials, n_outer, n_inner, n_comp) matrix
    - For each component, computes a vectorized (n_outer*n_inner, n_outer*n_inner) Pearson correlation matrix
    - Reads off-diagonal pairs back into rows
    """
    outer_vals = sorted(group_rows['outer_loop_n'].unique())
    inner_vals = sorted(group_rows['inner_loop_idx'].unique())
    n_outer, n_inner = len(outer_vals), len(inner_vals)
    n_folds = n_outer * n_inner
    outer_idx_map = {v: i for i, v in enumerate(outer_vals)}
    inner_idx_map = {v: i for i, v in enumerate(inner_vals)}
 
    # Load and standardize all data into the grid
    loaded = {}
    for _, row in group_rows.iterrows():
        data = analysis.load_data(row.trials_data_path)
        if data is not None:
            loaded[(row.outer_loop_n, row.inner_loop_idx)] = _standardize_hidden_units(data)
 
    if len(loaded) < 2:
        return None
 
    ref_df = next(iter(loaded.values()))
    n_trials = len(ref_df)
 
    # Identify component columns from reference df
    comp_cols = {comp: [c for c in ref_df.columns if c.startswith(comp)] for comp in COMPONENTS}
    all_cols = [c for cols in comp_cols.values() for c in cols]
    col_idx = {c: i for i, c in enumerate(all_cols)}
    n_comp = len(all_cols)
 
    # Build big matrix: (n_trials, n_outer, n_inner, n_comp)
    big_matrix = np.full((n_trials, n_outer, n_inner, n_comp), np.nan)
    for (outer, inner), df in loaded.items():
        oi = outer_idx_map[outer]
        ii = inner_idx_map[inner]
        big_matrix[:, oi, ii, :] = df[all_cols].values
 
    # Flatten (n_outer, n_inner) -> n_folds: (n_trials, n_folds, n_comp)
    # Fold order: outer varies slowest, inner fastest (C-order)
    X = big_matrix.reshape(n_trials, n_folds, n_comp)
 
    # Upper triangle pairs (i < j), maps flat fold index back to (outer, inner)
    fi, fj = np.triu_indices(n_folds, k=1)
    fold_outer = np.array(outer_vals)[np.arange(n_folds) // n_inner]
    fold_inner = np.array(inner_vals)[np.arange(n_folds) % n_inner]
 
    results = []
    for comp, cols in comp_cols.items():
        if not cols:
            continue
        cidx = [col_idx[c] for c in cols]
 
        # Extract component slice: (n_trials, n_folds, n_cidx)
        X_comp = X[:, :, cidx]
 
        # Concatenate component cols along trials axis -> (n_cidx * n_trials, n_folds)
        # For each fold column: [col0_trial0..col0_trialN, col1_trial0..col1_trialN, ...]
        X_comp = X_comp.transpose(2, 0, 1).reshape(-1, n_folds)  # (n_cidx * n_trials, n_folds)
 
        # Pearson r via normalised dot product:
        # subtract column mean, divide by column norm -> each column has mean=0, norm=1
        # then corr(i,j) = col_i · col_j  (no further division needed)
        X_comp = X_comp - np.nanmean(X_comp, axis=0, keepdims=True)
        norms = np.linalg.norm(X_comp, axis=0, keepdims=True)
        norms[norms == 0] = np.nan
        X_comp = X_comp / norms  # each column now unit norm
 
        corr_matrix = X_comp.T @ X_comp  # (n_folds, n_folds) — exact Pearson r
 
        pair_sims = corr_matrix[fi, fj]  # upper triangle, shape (n_pairs,)
 
        for k in range(len(fi)):
            results.append({
                "model_id":       model_id,
                "subject_id":     subject_id,
                "model_A_outer_n": fold_outer[fi[k]],
                "model_A_inner_n": fold_inner[fi[k]],
                "model_B_outer_n": fold_outer[fj[k]],
                "model_B_inner_n": fold_inner[fj[k]],
                "component":      comp,
                "similarity":     pair_sims[k],
            })
 
    return results
 

def _standardize_hidden_units(trials_df: pd.DataFrame, verbose: bool = False):
    """Standardizes a 2-unit network for alignment."""
    if 'hidden_2' not in trials_df:
        return trials_df #nothing to do here.
    if trials_df is None: return None
    corr1 = np.corrcoef(trials_df.hidden_1, trials_df.logit_value)[0,1]
    corr2 = np.corrcoef(trials_df.hidden_2, trials_df.logit_value)[0,1]
    corr1 = -np.inf if np.isnan(corr1) else corr1
    corr2 = -np.inf if np.isnan(corr2) else corr2

    if corr2 > corr1:
        for prefix in ['hidden_', 'gate_update_', 'gate_reset_']:
            col1, col2 = f"{prefix}1", f"{prefix}2"
            if col1 in trials_df.columns and col2 in trials_df.columns:
                trials_df[col1], trials_df[col2] = trials_df[col2].copy(), trials_df[col1].copy()
        corr1, corr2 = corr2, corr1

    if corr1 < 0:
        if trials_df.hidden_1.min() < -0.1: trials_df['hidden_1'] = -trials_df['hidden_1']
        else: trials_df['hidden_1'] = trials_df['hidden_1'].max() - trials_df['hidden_1']

    corr2_curr = np.corrcoef(trials_df.hidden_2, trials_df.logit_value)[0,1]
    if not np.isnan(corr2_curr) and corr2_curr > 0.0:
        if trials_df.hidden_2.min() < -0.1: trials_df['hidden_2'] = -trials_df['hidden_2']
        else: trials_df['hidden_2'] = trials_df['hidden_2'].max() - trials_df['hidden_2']
    return trials_df

def parameter_contribution_df(best_models_df):
    """Computes the normalized contribution of inputs to gated components."""
    contributions_dict = {'model_id':[], 'outer_loop_n':[], 'weight_seed':[], 'performance':[], 'variable':[], 'value':[]}
    for each_model in best_models_df.itertuples():
        params_dict = {k:v.numpy() for k,v in analysis.load_data(each_model.model_state_path).items()}
        for each_input in ['outcome','past_choice','past_hidden']:
            for each_output in ['update_gate','reset_gate','hidden_state']:
                if each_output == 'hidden_state': param_keys = ['rnn.W_ih', 'rnn.W_hh']
                elif each_output == 'update_gate': param_keys = ['rnn.W_iz', 'rnn.W_hz']
                elif each_output == 'reset_gate': param_keys = ['rnn.W_ir', 'rnn.W_hr']

                contributions_dict['variable'].append(f"{each_input}_to_{each_output}")
                if not all(x in params_dict for x in param_keys):
                    contributions_dict['value'].append(np.nan)
                else:
                    total_abs_weights = sum(np.sum(np.abs(params_dict[k])) for k in param_keys)
                    if each_input == 'outcome': input_weights = params_dict[param_keys[0]][0,:]
                    elif each_input == 'past_choice': input_weights = params_dict[param_keys[0]][1,:]
                    elif each_input == 'past_hidden': input_weights = params_dict[param_keys[1]]
                    
                    contributions_dict['value'].append(np.sum(np.abs(input_weights)) / total_abs_weights)

                contributions_dict['model_id'].append(each_model.model_id)
                contributions_dict['outer_loop_n'].append(each_model.outer_loop_n)
                contributions_dict['weight_seed'].append(each_model.weight_seed)
                contributions_dict['performance'].append(each_model.eval_CE)
    return pd.DataFrame(contributions_dict)

# --- MAIN EXECUTION --- #

if __name__ == "__main__":
    # 1. Training
    train_models()

    # 2. Build Analysis DataFrame
    analysis_df = build_analysis_df(SAVE_PATH)
    analysis_df = add_data(analysis_df)

    # Visualization: Hidden units scatter
    mono_df = analysis_df.query("model_id=='2_unit_monoGRU_relu_unipolar'").sort_values(['train_seed','weight_seed'])
    if not mono_df.empty:
        n_t, n_w = mono_df.train_seed.nunique(), mono_df.weight_seed.nunique()
        fig, ax = plt.subplots(n_t, n_w, figsize=(n_w*3, n_t*3))
        flat_ax = ax.flatten() if hasattr(ax, 'flatten') else [ax]
        for i, row in enumerate(mono_df.itertuples()):
            td = analysis.load_data(row.trials_data_path)
            sns.scatterplot(data=td, x='hidden_1', y='hidden_2', hue='logit_value', palette='coolwarm', legend=False, ax=flat_ax[i])
        plt.show()

    # Visualization: Performance stripping
    best_idx = analysis_df.groupby(['model_id','train_seed'])['val_CE'].idxmin()
    best_models_df = analysis_df.loc[best_idx]
    sns.stripplot(data=analysis_df, x='train_seed', y='eval_CE', hue='model_id')
    plt.show()

    fig, ax = plt.subplots()
    sns.stripplot(data=best_models_df, x='model_id', y='eval_CE', hue='model_id', dodge=True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    # 3. Similarities
    sim_df = compute_similarities(best_models_df)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.stripplot(data=best_models_df, x='model_id', y='eval_CE', ax=ax[0])
    sns.stripplot(data=sim_df, x='model_id', y='hidden_state_similarity', ax=ax[1])
    sns.stripplot(data=sim_df, x='model_id', y='update_gate_similarity', ax=ax[2])
    ax[0].set(title='performance'); ax[1].set(title='hidden states'); ax[2].set(title='gating mechanism')
    plt.tight_layout()
    plt.show()

    # 4. Parameters
    cont_df = parameter_contribution_df(best_models_df)
    update_gate_df = cont_df[cont_df.variable.str.contains('update')]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=update_gate_df, x='variable', y='value', hue='model_id', dodge=True, ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()
    
    # Example Logit Plot
    v_models = best_models_df.query('model_id=="2_unit_vanilla_relu_unipolar"')
    if not v_models.empty:
        min_idx = v_models.eval_CE.idxmin()
        model_row = best_models_df.loc[min_idx]
        td = analysis.load_data(model_row.trials_data_path)
        plt.figure()
        sns.scatterplot(data=td, x='logit_past', y='logit_change', hue='trial_type')
        plt.tight_layout()
        plt.show()