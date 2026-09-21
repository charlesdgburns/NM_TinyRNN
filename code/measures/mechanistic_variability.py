""" 
OUTDATED AFTER TRAINING REFACTOR.

The following script contains code to perform an analysis of mechanistic variability.
This is written for the 2-armed bandit reversal task modelled trial-by-trial.

1. train models on a subjects with multiple train seeds and weight seeds.
2. compare evaluation performance as across train and weight seeds.
3. compare similarity of activations (hidden units and gates).
4. compare the parameters (weights) of models.

Refactored to utilize a canonical structural alignment and 
similarity metrics to be completely self-sufficient.
"""

import os
import itertools
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

## local imports
from NM_TinyRNN.code.models import submit_jobs
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.measures import performance as perf
from NM_TinyRNN.code.models import datasets as ds

# GLOBAL VARIABLES 
AB_DATA_PATH = Path("NM_TinyRNN/data/AB_behaviour")
SAVE_PATH = Path("NM_TinyRNN/data/rnns/mech_var")

# --- CORE ALIGNMENT & SIMILARITY FUNCTIONS --- #

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

def compute_weight_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """
    Computes Cosine Similarity between the flattened parameters of Model A 
    and Model B. Assumes models are already canonicalized.
    """
    def extract_flat_params(model):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param.data.detach().cpu().numpy().ravel())
        return np.concatenate(params)

    vec_a = extract_flat_params(model_a)
    vec_b = extract_flat_params(model_b)

    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)
    return float(cosine_sim)

def compute_activation_similarity(
    model_a: nn.Module, 
    model_b: nn.Module, 
    trial_inputs: torch.Tensor,
    metric: str = 'pearson'
) -> float:
    """
    Runs data through both models and compares their internal hidden states.
    Assumes models are already canonicalized so unit i in A matches unit i in B.
    
    metric: 'pearson' (unit-wise correlation averaged) or 'cka' (Centered Kernel Alignment)
    """
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        out_a, _ = model_a.rnn(trial_inputs)
        out_b, _ = model_b.rnn(trial_inputs)

        act_a = out_a.reshape(-1, out_a.shape[-1]).cpu().numpy()
        act_b = out_b.reshape(-1, out_b.shape[-1]).cpu().numpy()

    if metric == 'pearson':
        correlations = []
        for i in range(act_a.shape[1]):
            std_a = np.std(act_a[:, i])
            std_b = np.std(act_b[:, i])
            if std_a > 1e-5 and std_b > 1e-5:
                cov = np.cov(act_a[:, i], act_b[:, i])[0, 1]
                correlations.append(cov / (std_a * std_b))
        
        return float(np.mean(correlations)) if correlations else 0.0

    elif metric == 'cka':
        act_a_centered = act_a - act_a.mean(axis=0)
        act_b_centered = act_b - act_b.mean(axis=0)
        
        hsic = np.linalg.norm(act_b_centered.T @ act_a_centered, ord='fro') ** 2
        var_a = np.linalg.norm(act_a_centered.T @ act_a_centered, ord='fro')
        var_b = np.linalg.norm(act_b_centered.T @ act_b_centered, ord='fro')
        
        return float(hsic / (var_a * var_b + 1e-8))
    
    else:
        raise ValueError("Metric must be 'pearson' or 'cka'")

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
            outer_loop_n = path.parts[-2].replace('outer_fold_', '')
            inner_loop_idx = path.parts[-1].replace('inner_fold_', '')

            info_files = list(path.glob("*_info.json"))
            for info_file in info_files:
                model_id = info_file.name.replace("_info.json", "")
                data_rows.append({
                    "subject_id": subject_id,
                    "outer_loop_n": outer_loop_n,
                    "inner_loop_idx": inner_loop_idx,
                    "model_id": model_id,
                    "info_path": path / f"{model_id}_info.json",
                    "model_state_path": path / f"{model_id}_model_state.pth",
                    "model_pickle_path": path / f"{model_id}_model.pickle",
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

def compute_similarities(
    analysis_df,
    n_jobs=-1,
    similarity_measure="pearson",
    use_cache=True,
    cache_path=None,
):
    """Compute or load activation similarities for each model pair."""
    if similarity_measure not in {"pearson", "cosine", "cka"}:
        raise ValueError(
            "similarity_measure must be 'pearson', 'cosine', or 'cka'"
        )

    if cache_path is None:
        cache_path = Path(
            f"NM_TinyRNN/data/analysis/mechanistic_variability_"
            f"{similarity_measure}_similarities_v2.htsv"
        )
    else:
        cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        cached_df = pd.read_csv(cache_path, sep="\t")
        print(f"Loaded similarity dataframe from cache: {cache_path}")
        return cached_df

    groups = list(analysis_df.groupby('model_id'))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_group)(
            group_rows.reset_index(drop=True),
            model_id,
            similarity_measure,
        )
        for model_id, group_rows in groups
    )

    results = [r for r in results if r is not None]
    similarity_df = (
        pd.DataFrame([row for result in results for row in result])
        if results else pd.DataFrame()
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(cache_path, sep="\t", index=False)
    print(f"Saved similarity dataframe to cache: {cache_path}")
    return similarity_df


def _process_group(
    group_rows,
    model_id,
    similarity_measure="pearson",
):
    """Compare model pairs using canonically aligned activations on every subject's trials."""
    fold_records = []
    for _, row in group_rows.sort_values(
        ['subject_id', 'outer_loop_n', 'inner_loop_idx']
    ).iterrows():
        model = analysis.load_data(str(row.model_pickle_path))
        trials = analysis.load_data(str(row.trials_data_path))
        if model is not None and trials is not None:
            # Standardize model in place so it aligns canonically to all other models
            standardize_weights(model)
            fold_records.append({
                'subject_id': row.subject_id,
                'outer_loop_n': row.outer_loop_n,
                'inner_loop_idx': row.inner_loop_idx,
                'model': model,
                'trials': trials,
            })
 
    if len(fold_records) < 2:
        return None
 
    n_folds = len(fold_records)
    trial_records = {}
    for record in fold_records:
        trial_records.setdefault(record['subject_id'], record['trials'])

    results = []
    for index_a, index_b in itertools.combinations(range(n_folds), 2):
        record_a = fold_records[index_a]
        record_b = fold_records[index_b]
        
        model_a = record_a['model']
        model_b = record_b['model']

        for trial_subject in trial_records:
            if trial_subject != record_b['subject_id']:
                continue
                
            trials = trial_records[trial_subject]
            
            raw = torch.tensor(
                trials[["forced_choice", "choice", "outcome"]].to_numpy(),
                dtype=torch.float32,
            ).unsqueeze(0)
            inputs = ds.input_encoder(raw, model_a.input_encoding, model_a.input_forced_choice)
            model_device = next(model_a.parameters()).device
            inputs = inputs.to(model_device)
            
            if similarity_measure in ['pearson', 'cka']:
                hidden_sim = compute_activation_similarity(model_a, model_b, inputs, metric=similarity_measure)
            else:
                hidden_sim = compute_activation_similarity(model_a, model_b, inputs, metric='pearson')

            activations_a = _run_model_on_trials(model_a, trials, inputs)
            activations_b = _run_model_on_trials(model_b, trials, inputs)
            
            component_columns_a = _component_columns(activations_a)
            component_columns_b = _component_columns(activations_b)
            
            for comp in COMPONENTS:
                if comp == 'hidden':
                    similarity = hidden_sim
                else:
                    cols = [
                        column for column in activations_a.columns
                        if column.startswith(comp) and column in activations_b.columns
                    ]
                    if (comp not in component_columns_a or comp not in component_columns_b or not cols):
                        continue
                    
                    similarity = _similarity(
                        activations_a[cols].to_numpy().ravel(),
                        activations_b[cols].to_numpy().ravel(),
                        "pearson" if similarity_measure == 'cka' else similarity_measure, 
                    )
                
                results.append({
                    "model_id": model_id,
                    "subject_A": record_a['subject_id'],
                    "model_A_outer_n": record_a['outer_loop_n'],
                    "model_A_inner_n": record_a['inner_loop_idx'],
                    "subject_B": record_b['subject_id'],
                    "model_B_outer_n": record_b['outer_loop_n'],
                    "model_B_inner_n": record_b['inner_loop_idx'],
                    "trial_subject": trial_subject,
                    "component": comp,
                    "similarity": similarity,
                    "similarity_measure": similarity_measure,
                    "within_subject": record_a['subject_id'] == record_b['subject_id'],
                })

    return results


def _component_columns(activations):
    return {component for component in COMPONENTS
            if any(column.startswith(component) for column in activations.columns)}


def _run_model_on_trials(model, trials_data, inputs):
    """Run a model on raw trial columns and return comparable activations.
       Models are canonically standardized upstream."""
    model.eval()
    with torch.no_grad():
        if model.input_encoding == 'encoder':
            inputs = model.encoder(inputs)
        hidden_states, gate_activations = model.rnn(
            inputs, return_gate_activations=True
        )
        predictions = model.decoder(hidden_states)

    activations = pd.DataFrame({
        f"hidden_{unit + 1}": hidden_states[0, :, unit].cpu().numpy()
        for unit in range(hidden_states.shape[-1])
    })
    for gate_name, gate_values in gate_activations.items():
        for unit in range(gate_values.shape[-1]):
            activations[f"gate_{gate_name}_{unit + 1}"] = (
                gate_values[0, :, unit].cpu().numpy()
            )

    log_probs = predictions.log_softmax(dim=2)
    activations["logit_value"] = (
        log_probs[0, :, 0] - log_probs[0, :, 1]
    ).cpu().numpy()
    
    return activations

def parameter_contribution_df(best_models_df):
    """Computes the normalized contribution of inputs to gated components."""
    contributions_dict = {'model_id':[], 'outer_loop_n':[], 'weight_seed':[], 'performance':[], 'variable':[], 'value':[]}
    for each_model in best_models_df.itertuples():
        model =  analysis.load_data(each_model.model_pickle_path)
        params_dict = {k:v.detach().numpy() for k,v in model.named_parameters()}
        for each_input in ['context','past_choice','outcome','past_hidden']:
            for each_output in ['update_gate','reset_gate','hidden_state']:
                if each_output == 'hidden_state': param_keys = ['rnn.W_ih', 'rnn.W_hh']
                elif each_output == 'update_gate': param_keys = ['rnn.W_iz', 'rnn.W_hz']
                elif each_output == 'reset_gate': param_keys = ['rnn.W_ir', 'rnn.W_hr']

                contributions_dict['variable'].append(f"{each_input}_to_{each_output}")
                if not all(x in params_dict for x in param_keys):
                    contributions_dict['value'].append(np.nan)
                else:
                    total_abs_weights = sum(np.sum(np.abs(params_dict[k])) for k in param_keys)
                    if each_input == 'context': input_weights = params_dict[param_keys[0]][0,:]
                    elif each_input == 'past_choice': input_weights = params_dict[param_keys[0]][1,:]
                    elif each_input == 'outcome': input_weights = params_dict[param_keys[0]][2,:]
                    elif each_input == 'past_hidden': input_weights = params_dict[param_keys[1]]
                    contributions_dict['value'].append(np.sum(np.abs(input_weights)) / total_abs_weights)

                contributions_dict['model_id'].append(each_model.model_id)
                contributions_dict['outer_loop_n'].append(each_model.outer_loop_n)
                contributions_dict['weight_seed'].append(each_model.weight_seed)
                contributions_dict['performance'].append(each_model.eval_CE)
    return pd.DataFrame(contributions_dict)


## Inspecting weights

def _extract_flat_params(model):
    """Extract flattened parameters for similarity/clustering."""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.data.detach().cpu().numpy().ravel())
    return np.concatenate(params)

def _similarity(a, b, similarity_measure="pearson"):
    """Compute Pearson or cosine similarity between two vectors."""
    if similarity_measure not in {"pearson", "cosine"}:
        raise ValueError(
            "similarity_measure must be either 'pearson' or 'cosine'"
        )

    if similarity_measure == "pearson":
        a = a - a.mean()
        b = b - b.mean()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan

    return np.dot(a, b) / denom

def plot_pairwise_parameters(subject_models_df, model_pickle_col='model_pickle_path'):
    """
    Plot pairwise parameter correlations across trained models for a single subject.
    Models are canonically standardizing prior to extraction.
    """
    weights = {}
    
    for idx, row in subject_models_df.iterrows():
        try:
            model = analysis.load_data(row[model_pickle_col])
            standardize_weights(model)
            weights[idx] = _extract_flat_params(model)
        except Exception as e:
            print(f"Could not load model at index {idx}: {e}")

    indices = list(weights.keys())
    n_models = len(indices)
    param_dim = len(next(iter(weights.values())))
    
    print(f"Loaded {n_models} models, {param_dim} parameters each")

    # --- Plot 1: Pairwise scatter of raw parameter vectors ---
    n_pairs = min(len(list(itertools.combinations(indices, 2))), 20)
    pairs = list(itertools.combinations(indices, 2))[:n_pairs]
    
    ncols = 4
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows * ncols == 1: axes = [axes]
    else: axes = axes.flatten()

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

    plt.suptitle("Pairwise parameter scatter (single subject, canonically aligned)", y=1.02)
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

def cluster_models_full(models_df, model_pickle_col='model_pickle_path',
                        loss_col='eval_CE', k_range=range(2, 10), n_clusters=None):
    """
    Full clustering pipeline mapping models into an aligned shared canonical parameter space.
    """
    indices, vecs, meta = [], [], []
    
    for idx, row in models_df.iterrows():
        try:
            model = analysis.load_data(row[model_pickle_col])
            standardize_weights(model)
            vecs.append(_extract_flat_params(model))
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

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               marker='x', s=150, c='black', linewidths=2, label='Centroids')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of aligned parameter vectors")
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

    return meta_df, kmeans, pca, X_scaled

def compute_weight_similarities(
    models_df,
    model_pickle_col="model_pickle_path",
    model_id_col="model_id",
    subject_col="subject_id",
    similarity_measure="cosine", 
):
    """
    Computes pairwise weight similarity utilizing the standardized canonical network forms.
    """
    rows = []

    for model_id, group in models_df.groupby(model_id_col):
        group = group.reset_index(drop=True)

        # Load and standardize all models once, keyed by row index
        models = {}
        for idx, row in group.iterrows():
            try:
                model = analysis.load_data(row[model_pickle_col])
                standardize_weights(model)
                models[idx] = model
            except Exception as e:
                warnings.warn(
                    f"Could not load/standardize model for subject {row[subject_col]} "
                    f"(outer={row.get('outer_loop_n', '?')}, "
                    f"inner={row.get('inner_loop_idx', '?')}): {e}"
                )

        # Pairwise comparisons across all rows using standardized forms
        for idx_a, idx_b in itertools.combinations(group.index, 2):
            if idx_a not in models or idx_b not in models:
                continue

            row_a = group.loc[idx_a]
            row_b = group.loc[idx_b]
            model_a = models[idx_a]
            model_b = models[idx_b]
            
            # Calculate similarity of flat canonical params
            similarity = compute_weight_similarity(model_a, model_b)

            rows.append({
                "model_id": model_id,
                "subject_A": row_a[subject_col],
                "outer_loop_n_A": row_a.get("outer_loop_n"),
                "inner_loop_idx_A": row_a.get("inner_loop_idx"),
                "subject_B": row_b[subject_col],
                "outer_loop_n_B": row_b.get("outer_loop_n"),
                "inner_loop_idx_B": row_b.get("inner_loop_idx"),
                "within_subject": row_a[subject_col] == row_b[subject_col],
                "similarity": similarity,
                "sim_original": similarity,
                "similarity_measure": "cosine", 
            })

    return pd.DataFrame(rows)


# --- BATCH ANALYSIS --- #

def run_across_subject_similarity_analysis(
    model_list=None,
    subject_list=None,
    use_best_models=True,
    n_jobs=-1,
):
    """Build, compute, and save the across-subject similarity analysis."""
    if model_list is None:
        model_list = ['GRU+BC', 'GRU+BC-DB', 'GRU', 'monoGRU+BC', 'monoGRU+BC-DB']
    if subject_list is None:
        subject_list = [
            'WS01', 'WS02','WS05', 'WS08', 'WS09', 'WS10',
            'WS13', 'WS14', 'WS16', 'WS20', 'WS22',
        ]

    info_df = submit_jobs.get_DA_info_df()
    all_models_df = analysis.get_analysis_df(info_df, mode='all')
    performance_df = perf.get_performance_df(all_models_df)
    best_model_df = perf.select_best_outer(performance_df)

    source_df = best_model_df if use_best_models else all_models_df
    select_models = source_df.query(
        'hidden_size == 2 and input_encoding == "unipolar"'
    ).copy()
    select_models = select_models[
        select_models.model_type2.isin(model_list)
        & select_models.subject_id.isin(subject_list)
    ].copy()
    select_models['model_id'] = select_models.model_type2

    output_dir = Path('NM_TinyRNN/data/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / 'mechanistic_variability_selected_models.htsv'
    similarity_path = output_dir / 'mechanistic_variability_pearson_similarities_v2.htsv'
    select_models.to_csv(selected_path, sep='\t', index=False)
    print(f'Saved selected model dataframe to {selected_path}')

    sim_df = compute_similarities(
        select_models,
        n_jobs=n_jobs,
        similarity_measure='pearson',
        use_cache=False,
        cache_path=similarity_path,
    )
    print(f'Saved similarity dataframe to {similarity_path}')
    return select_models, sim_df


if __name__ == "__main__":
    print('Running analysis')
    run_across_subject_similarity_analysis()
    print('done!')