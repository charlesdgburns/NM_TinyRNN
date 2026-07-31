""" 
OUTDATED AFTER TRAINING REFACTOR.

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
import torch
from joblib import Parallel, delayed
## local imports
from NM_TinyRNN.code.models import submit_jobs
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.models import datasets as ds

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
                submit_jobs.train_outers(
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
    groups = list(analysis_df.groupby(['model_id', 'subject_id']))
 
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
            loaded[(row.outer_loop_n, row.inner_loop_idx)] = _standardize_activations(data)
 
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
 
from scipy.optimize import linear_sum_assignment
#
def _standardize_activations(trials_df: pd.DataFrame, verbose: bool = False):
    """Standardizes a 2-unit network for alignment."""
    if 'hidden_2' not in trials_df:
        corr1 = np.corrcoef(trials_df.hidden_1, trials_df.logit_value)[0,1]
        corr1 = -np.inf if np.isnan(corr1) else corr1
        if corr1<-0.1:
            trials_df.hidden_1 = -trials_df.hidden_1 #flip sign
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
def standardize_weights(model: torch.nn.Module, verbose: bool = False):
    """
    Standardizes a 2-unit TinyRNN model in-place to a canonical form 
    using only its decoder weights. No trial data required.
    
    Supports both 'tanh' and 'relu' networks gracefully.
    """
    rnn = model.rnn
    rnn_type = model.rnn_type
    
    # Check if the model uses ReLU or Tanh
    is_relu = getattr(model, 'nonlinearity', 'tanh') == 'relu'

    with torch.no_grad():
        # The decoder weights have shape (out_size, hidden_size) -> e.g., (2, 2)
        dec_w = model.decoder.weight.data
        
        # Calculate the effective weight vector: class_0_weights - class_1_weights
        # This represents the actual linear direction driving the binary logit
        w_eff = dec_w[0, :] - dec_w[1, :] # Shape: (2,)
        
        # ----------------------------------------------------
        # STEP 1: Determine Permutation (Swapping Unit 1 and 2)
        # ----------------------------------------------------
        # Force the unit with the larger absolute impact on the logit to be Unit 0
        was_swapped = torch.abs(w_eff[1]) > torch.abs(w_eff[0])
        perm_idx = [1, 0] if was_swapped else [0, 1]
        
        # Mentally apply the permutation to see what the effective weights look like post-swap
        w_eff_aligned = w_eff[perm_idx]
        
        # ----------------------------------------------------
        # STEP 2: Determine Sign Flips
        # ----------------------------------------------------
        flip_h1 = False
        flip_h2 = False
        
        if not is_relu:
            # For Tanh/Linear networks, we can safely fix the sign orientations
            # Force Unit 0's readout weight to be positive
            flip_h1 = w_eff_aligned[0] < 0
            # Force Unit 1's readout weight to be positive as well
            flip_h2 = w_eff_aligned[1] < 0
            
        sign_multipliers = torch.tensor([-1.0 if flip_h1 else 1.0, -1.0 if flip_h2 else 1.0], dtype=torch.float32)

        # ----------------------------------------------------
        # STEP 3: Apply Symmetries Upstream In-Place
        # ----------------------------------------------------
        
        # A. Update Decoder (Output Layer)
        if dec_w.shape[1] == 2:
            model.decoder.weight.data = model.decoder.weight.data[:, perm_idx] * sign_multipliers

        # B. Update Initial Hidden States
        if hasattr(rnn, 'hidden_0') and rnn.hidden_0.data.shape[0] == 2:
            rnn.hidden_0.data = rnn.hidden_0.data[perm_idx] * sign_multipliers

        # C. Update Hidden RNN / GRU / monoGRU Layer Blocks
        if rnn_type in ['vanilla', 'LightGRU', 'GRU'] or 'monoGRU' in rnn_type:
            
            # --- Input-to-Hidden Tensors ---
            # Candidate state (W_ih): columns represent target units -> apply sign flip
            if hasattr(rnn, 'W_ih') and rnn.W_ih is not None:
                if rnn.W_ih.data.shape[1] == 2:
                    rnn.W_ih.data = rnn.W_ih.data[:, perm_idx] * sign_multipliers
            
            # Sigmoid Gates (W_iz, W_ir): permute columns, but DO NOT flip signs
            for attr in ['W_iz', 'W_ir']:
                if hasattr(rnn, attr) and getattr(rnn, attr) is not None:
                    param = getattr(rnn, attr)
                    if param.data.shape[1] == 2:
                        param.data = param.data[:, perm_idx]

            # --- Biases ---
            # Candidate bias (bias_h): maps to hidden unit outputs -> apply sign flip
            if hasattr(rnn, 'bias_h') and rnn.bias_h is not None:
                if rnn.bias_h.data.shape[0] == 2:
                    rnn.bias_h.data = rnn.bias_h.data[perm_idx] * sign_multipliers
            
            # Gate biases (bias_z, bias_r): only permute, DO NOT flip signs
            for attr in ['bias_z', 'bias_r']:
                if hasattr(rnn, attr) and getattr(rnn, attr) is not None:
                    param = getattr(rnn, attr)
                    if param.data.ndim > 0 and param.data.shape[0] == 2:
                        param.data = param.data[perm_idx]

            # --- Recurrent Tensors ---
            # Full candidate Recurrent matrix (W_hh): scales both source and target axes
            if hasattr(rnn, 'W_hh') and rnn.W_hh is not None:
                if rnn.W_hh.data.shape == (2, 2):
                    rnn.W_hh.data = rnn.W_hh.data[perm_idx, :][:, perm_idx]
                    rnn.W_hh.data = rnn.W_hh.data * sign_multipliers.unsqueeze(1) # row scale (source)
                    rnn.W_hh.data = rnn.W_hh.data * sign_multipliers.unsqueeze(0) # col scale (target)

            # Recurrent Gate matrices (W_hz, W_hr):
            # Rows represent incoming hidden states -> apply sign flip to counteract input
            # Columns represent gate targets going to Sigmoid -> DO NOT flip signs
            for attr in ['W_hz', 'W_hr']:
                if hasattr(rnn, attr) and getattr(rnn, attr) is not None:
                    param = getattr(rnn, attr)
                    if param.data.shape == (2, 2):
                        param.data = param.data[perm_idx, :][:, perm_idx]
                        param.data = param.data * sign_multipliers.unsqueeze(1) # row scale ONLY
                    elif param.data.shape == (2, 1):
                        param.data = param.data[perm_idx, :]
                        param.data = param.data * sign_multipliers.unsqueeze(1) # row scale ONLY
                        
        elif rnn_type == 'LSTM':
            # Support for LSTM layout [i, f, g, o] gates stacked side-by-side
            for i, gate_name in enumerate(['i', 'f', 'g', 'o']):
                start, end = i * 2, (i + 1) * 2
                if rnn.W.data.shape[1] >= end:
                    # Input matrix W
                    if gate_name == 'g':
                        rnn.W.data[:, start:end] = rnn.W.data[:, start:end][:, perm_idx] * sign_multipliers
                        rnn.bias.data[start:end] = rnn.bias.data[start:end][perm_idx] * sign_multipliers
                    else:
                        rnn.W.data[:, start:end] = rnn.W.data[:, start:end][:, perm_idx]
                        rnn.bias.data[start:end] = rnn.bias.data[start:end][perm_idx]
                    
                    # Recurrent matrix U
                    rnn.U.data[perm_idx, :] = rnn.U.data[perm_idx, :] 
                    rnn.U.data = rnn.U.data * sign_multipliers.unsqueeze(1) 
                    rnn.U.data[:, start:end] = rnn.U.data[:, start:end][:, perm_idx] 
                    if gate_name == 'g':
                        rnn.U.data[:, start:end] = rnn.U.data[:, start:end] * sign_multipliers.unsqueeze(0)

    if verbose:
        print(f"Data-Free Alignment Complete! [Type: {'ReLU' if is_relu else 'Tanh'}] "
              f"Swapped: {was_swapped} | Flip H1: {flip_h1} | Flip H2: {flip_h2}")
    return was_swapped, flip_h1, flip_h2

def parameter_contribution_df(best_models_df):
    """Computes the normalized contribution of inputs to gated components."""
    contributions_dict = {'model_id':[], 'outer_loop_n':[], 'weight_seed':[], 'performance':[], 'variable':[], 'value':[]}
    for each_model in best_models_df.itertuples():
        model =  analysis.load_data(each_model.model_pickle_path)
        params_dict = {k:v.detach().numpy() for k,v in model.named_parameters()}
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


## Inspecting weights

import numpy as np
import matplotlib.pyplot as plt
import itertools

def _load_weights(model_path):
    """Load model and extract rnn weight matrices as numpy arrays."""
    model = analysis.load_data(str(model_path))
    standardize_weights(model)
    params = {k: v.detach().numpy() for k, v in model.named_parameters()
                if k.startswith('rnn.')}
    return params

def _concat_weights(params):
    """Concatenate all rnn weight matrices into a single flat vector."""
    keys = sorted(params.keys())
    return np.concatenate([params[k].ravel() for k in keys])

def _similarity(a, b):
    """Pearson correlation between two flat vectors."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return np.dot(a, b) / denom

def _gate_dim(params):
    """Return the gate dimensionality (number of cols in W_hz) or None."""
    if 'rnn.W_hz' in params:
        return params['rnn.W_hz'].shape[1]
    return None

def plot_pairwise_parameters(subject_models_df, model_pickle_col='model_pickle_path'):
    """
    Plot pairwise parameter correlations across trained models for a single subject.
    
    subject_models_df: DataFrame filtered to a single subject, 
                       with one row per trained model
    """
    # Load all weights
    weights = {}
    for idx, row in subject_models_df.iterrows():
        try:
            params = _load_weights(row[model_pickle_col])
            weights[idx] = _concat_weights(params)
        except Exception as e:
            print(f"Could not load model at index {idx}: {e}")

    indices = list(weights.keys())
    n_models = len(indices)
    param_dim = len(next(iter(weights.values())))
    
    print(f"Loaded {n_models} models, {param_dim} parameters each")

    # --- Plot 1: Pairwise scatter of raw parameter vectors ---
    n_pairs = min(len(list(itertools.combinations(indices, 2))), 20)  # cap for readability
    pairs = list(itertools.combinations(indices, 2))[:n_pairs]
    
    ncols = 4
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for ax, (idx_a, idx_b) in zip(axes, pairs):
        va, vb = weights[idx_a], weights[idx_b]
        sim = _similarity(va, vb)
        ax.scatter(va, vb, alpha=0.6, s=20)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(f"r={sim:.2f}", fontsize=9)
        ax.set_xlabel(f"Model {idx_a}", fontsize=7)
        ax.set_ylabel(f"Model {idx_b}", fontsize=7)

    for ax in axes[n_pairs:]:
        ax.set_visible(False)

    plt.suptitle("Pairwise parameter scatter (single subject)", y=1.02)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Similarity matrix heatmap ---
    sim_matrix = np.full((n_models, n_models), np.nan)
    for i, idx_a in enumerate(indices):
        for j, idx_b in enumerate(indices):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                sim = _similarity(weights[idx_a], weights[idx_b])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels([f"M{i}" for i in indices], rotation=45)
    ax.set_yticklabels([f"M{i}" for i in indices])
    ax.set_title("Parameter similarity matrix (single subject)")
    plt.tight_layout()
    plt.show()

    return sim_matrix

# validation of reordering of the weights

def data_rerun(model, trials_data):
    data = trials_data.copy()

    # Build a single (1, T, 3) input tensor from the raw subject_df so that
    # every trial appears in temporal order regardless of sequence chunking.
    raw = torch.tensor(
        data[["forced_choice", "choice","outcome"]].values,
        dtype=torch.float32,
    ).unsqueeze(0)   # (1, T, 3)
    inputs = ds.input_encoder(raw, model.input_encoding, model.input_forced_choice)

    with torch.no_grad():
        predictions, hidden_states = model(inputs)
        # hidden_states: (1, T, H)
        for u in range(model.H):
            data[f"hidden_{u+1}"] = hidden_states[0, :, u].cpu().numpy()

        rnn_type = getattr(model, "rnn_type", "vanilla")
        if rnn_type != "vanilla":
            _, gate_activations = model.rnn(inputs, return_gate_activations=True)
            for gate_name, acts in gate_activations.items():
                for u in range(acts.shape[-1]):
                    data[f"gate_{gate_name}_{u+1}"] = acts[0, :, u].cpu().numpy()

    log_probs = predictions.log_softmax(dim=2)   # (1, T, 2)
    logits    = (log_probs[0, :, 0] - log_probs[0, :, 1]).cpu().numpy()
    data["logit_value"] = logits
    data["logit_past"]   = np.concatenate([[np.nan], logits[:-1]])
    data["logit_change"] = np.concatenate([[np.nan], np.diff(logits)])
    data["prob_A"]       = log_probs[0, :, 0].exp().cpu().numpy()
    data["prob_B"]       = log_probs[0, :, 1].exp().cpu().numpy()
    data

    return data



##k-means clustering on weights

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def cluster_models_full(models_df, model_pickle_col='model_pickle_path',
                        loss_col='eval_CE', k_range=range(2, 10), n_clusters=None):
    """
    Full clustering pipeline:
    1. Elbow + silhouette to pick k (or use n_clusters if provided)
    2. K-means clustering
    3. PCA visualization colored by cluster and loss
    4. Cluster summary
    """
    # --- Load weights ---
    indices, vecs, meta = [], [], []
    for idx, row in models_df.iterrows():
        try:
            params = _load_weights(row[model_pickle_col])
            if '1_unit_GRU' in row['model_id']:
                params = canonicalize_1unit_gru(params)
        
            vecs.append(_concat_weights(params))
            indices.append(idx)
            meta.append(row)
        except Exception as e:
            print(f"Failed to load {idx}: {e}")

    X = np.stack(vecs)
    meta_df = pd.DataFrame(meta).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow + silhouette ---
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k_silhouette = list(k_range)[np.argmax(silhouettes)]
    chosen_k = n_clusters if n_clusters is not None else best_k_silhouette
    print(f"Best k by silhouette: {best_k_silhouette}")
    print(f"Using k={chosen_k}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(k_range, inertias, 'o-')
    axes[0].axvline(chosen_k, color='red', linestyle='--', label=f'k={chosen_k}')
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow plot"); axes[0].legend()

    axes[1].plot(k_range, silhouettes, 'o-', color='orange')
    axes[1].axvline(chosen_k, color='red', linestyle='--', label=f'k={chosen_k}')
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score"); axes[1].legend()
    plt.tight_layout(); plt.show()

    # --- Final clustering ---
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    meta_df['cluster'] = labels

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    meta_df['pc1'] = X_pca[:, 0]
    meta_df['pc2'] = X_pca[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Cluster plot
    ax = axes[0]
    for c in range(chosen_k):
        mask = labels == c
        n = mask.sum()
        mean_loss = meta_df.loc[mask, loss_col].mean() if loss_col in meta_df else None
        label = f"Cluster {c} (n={n}" + (f", CE={mean_loss:.3f})" if mean_loss else ")")
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, s=50, alpha=0.8)

    # Mark cluster centroids in PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               marker='x', s=150, c='black', linewidths=2, label='Centroids')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of parameter vectors")
    ax.legend(fontsize=8)

    # Loss plot
    ax = axes[1]
    if loss_col in meta_df.columns:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=meta_df[loss_col], cmap='viridis_r', s=50, alpha=0.8)
        plt.colorbar(sc, ax=ax, label=loss_col)
        ax.set_title(f"PCA colored by {loss_col}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")

    plt.tight_layout(); plt.show()

    # --- Summary ---
    print("\nCluster summary:")
    cluster_stats = []
    for c in range(chosen_k):
        mask = meta_df['cluster'] == c
        row = {'cluster': c, 'n': mask.sum()}
        if loss_col in meta_df.columns:
            row['mean_loss'] = meta_df.loc[mask, loss_col].mean()
            row['min_loss'] = meta_df.loc[mask, loss_col].min()
            row['std_loss'] = meta_df.loc[mask, loss_col].std()
        cluster_stats.append(row)
        print(f"  Cluster {c}: n={row['n']}", end="")
        if loss_col in meta_df.columns:
            print(f", mean={row['mean_loss']:.4f}, min={row['min_loss']:.4f}, std={row['std_loss']:.4f}")

    if loss_col in meta_df.columns:
        best_cluster = min(cluster_stats, key=lambda x: x['mean_loss'])['cluster']
        print(f"\nBest cluster by mean loss: Cluster {best_cluster}")

    meta_df['param_vec'] = list(X)  # attach raw vecs for downstream use
    return meta_df, kmeans, pca, X_scaled


# computing weight similarities (NEED TO CONSIDER SYMMETRIES MORE CAREFULLY)
def canonicalize_1unit_gru(params):
    """
    Canonicalize a 1-unit GRU by fixing the sign of the hidden state.
    
    Convention: force W_hh > 0 (recurrent self-connection is positive).
    This is arbitrary but deterministic — any single weight that 
    participates in the hidden state path works as a reference.
    
    Affected by h sign flip:
        - W_ih   : input -> candidate h (writes to h)
        - W_hh   : h -> candidate h     (writes and reads h)
        - W_hz   : h -> update gate     (reads h)
        - W_hr   : h -> reset gate      (reads h)
        - bias_h : bias of candidate h  (additive to h path)
    
    NOT affected (sigmoid gates; their inputs are not h itself):
        - W_iz, W_ir : input -> gates
        - bias_z, bias_r : gate biases
    """
    p = {k: v.copy() for k, v in params.items()}
    
    sign = np.sign(p['rnn.W_hh'].item())
    
    if sign == 0:
        print("Warning: W_hh is zero, cannot determine sign convention")
        return p
    
    if sign < 0:
        p['rnn.W_ih']    *= -1
        p['rnn.W_hh']    *= -1
        p['rnn.W_hz']    *= -1
        p['rnn.W_hr']    *= -1
        p['rnn.bias_h']  *= -1
    
    return p

def compute_weight_similarities(models_df, model_pickle_col='model_pickle_path',
                                model_id_col='model_id', subject_col='subject_id'):
    """
    Computes pairwise weight similarity across subjects for each model type,
    accounting for the single hidden-unit permutation symmetry.

    For each pair of subjects sharing the same model_id, computes Pearson
    correlation between their concatenated RNN weight vectors. Subject A is
    treated as the fixed reference; subject B is tested in both its original
    and permuted form, and the better match is recorded.

    Permutation (swap units 1 <-> 2) affects:
      - W_ih, W_iz, W_ir : swap rows   (write to hidden units)
      - W_hh, W_hz, W_hr : swap rows AND columns  (read from + write to hidden)

    Mixed gate dimensionality (1D vs 2D W_hz) pairs are skipped with a warning.

    Parameters
    ----------
    models_df : pd.DataFrame
        One row per model, must contain columns for model path, model_id,
        and subject_id (column names configurable via keyword args).
    model_pickle_col : str
        Column name containing paths to saved model (.pth or pickle).
    model_id_col : str
        Column name identifying the model architecture type.
    subject_col : str
        Column name identifying the subject.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per model pair:
        model_id, subject_A, outer_loop_n_A, inner_loop_idx_A,
        subject_B, outer_loop_n_B, inner_loop_idx_B,
        within_subject, similarity, permutation_applied (bool),
        sim_original, sim_permuted.
    """
    import itertools
    import warnings

    # --- main loop ---

    rows = []

    for model_id, group in models_df.groupby(model_id_col):
        group = group.reset_index(drop=True)

        # Load all weights once, keyed by row index
        weights = {}
        for idx, row in group.iterrows():
            try:
                weights[idx] = _load_weights(row[model_pickle_col])
                if '1_unit_GRU' in model_id:
                    print('Standardising parameters')
                    weights[idx] = canonicalize_1unit_gru(weights[idx])
            except Exception as e:
                warnings.warn(
                    f"Could not load model for subject {row[subject_col]} "
                    f"(outer={row.get('outer_loop_n', '?')}, "
                    f"inner={row.get('inner_loop_idx', '?')}): {e}"
                )

        # Pairwise comparisons across all rows (includes within-subject pairs)
        for idx_a, idx_b in itertools.combinations(group.index, 2):
            if idx_a not in weights or idx_b not in weights:
                continue

            row_a = group.loc[idx_a]
            row_b = group.loc[idx_b]
            params_a = weights[idx_a]
            params_b = weights[idx_b]

            # Skip mixed gate dimensionality
            dim_a, dim_b = _gate_dim(params_a), _gate_dim(params_b)
            if dim_a != dim_b:
                warnings.warn(
                    f"Skipping {row_a[subject_col]} vs {row_b[subject_col]} "
                    f"for model '{model_id}': gate dims differ ({dim_a} vs {dim_b})."
                )
                continue

            vec_a = _concat_weights(params_a)

            # Test both orientations of subject B
            vec_b_orig = _concat_weights(params_b)
            vec_b_perm = _concat_weights(_permute_weights(params_b))

            sim_orig = _similarity(vec_a, vec_b_orig)
            sim_perm = _similarity(vec_a, vec_b_perm)

            if np.isnan(sim_orig) and np.isnan(sim_perm):
                best_sim, permuted = np.nan, False
            elif np.isnan(sim_perm) or sim_orig >= sim_perm:
                best_sim, permuted = sim_orig, False
            else:
                best_sim, permuted = sim_perm, True

            rows.append({
                'model_id':             model_id,
                'subject_A':            row_a[subject_col],
                'outer_loop_n_A':       row_a.get('outer_loop_n'),
                'inner_loop_idx_A':     row_a.get('inner_loop_idx'),
                'subject_B':            row_b[subject_col],
                'outer_loop_n_B':       row_b.get('outer_loop_n'),
                'inner_loop_idx_B':     row_b.get('inner_loop_idx'),
                'within_subject':       row_a[subject_col] == row_b[subject_col],
                'similarity':           best_sim,
                'permutation_applied':  permuted,
                'sim_original':         sim_orig,
                'sim_permuted':         sim_perm,
            })

    return pd.DataFrame(rows)


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