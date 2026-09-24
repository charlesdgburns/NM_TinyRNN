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



This script is structured as follows:

1. Code to align two models

2. Code to compue similarity metrics between models

3. Code to run the analysis across all models in parallel and save results
"""

import os
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

# For finding optimal alignment
import copy
import itertools
from scipy.optimize import linear_sum_assignment

## local imports
from NM_TinyRNN.code.models import submit_jobs
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.measures import performance as perf
from NM_TinyRNN.code.models import datasets as ds

# GLOBAL VARIABLES 
AB_DATA_PATH = Path("NM_TinyRNN/data/AB_behaviour")
SAVE_PATH = Path("NM_TinyRNN/data/rnns/mech_var")

## CORE ALIGNMENT FUNCTIONS ##


def align_rnns(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    nonlinearity: str | None = None,
    scale_steps: int = 5,
) -> tuple[torch.nn.Module, float, dict]:
    """Aligns Model B to Model A by fully searching the symmetry group

    (all permutations and sign/scale transformations) to minimize global parameter distance.

    Parameters
    ----------
    model_a : torch.nn.Module
        Reference RNN model.
    model_b : torch.nn.Module
        Model to align.
    nonlinearity : str, optional
        'relu' or 'tanh'. If None, inferred from `model_a.nonlinearity`.
    scale_steps : int
        Number of candidate positive scale factors per unit for ReLU (default: 5).

    Returns
    -------
    model_b_aligned : torch.nn.Module
        Copy of Model B with aligned weights.
    aligned_distance : float
        Euclidean parameter distance after symmetry alignment.
    info : dict
        Contains 'perm' (permutation array), 'transform_vec' (D or S),
        and 'unaligned_distance' (distance before alignment).
    """
    if nonlinearity is None:
        if hasattr(model_a, "nonlinearity"):
            nonlinearity = str(model_a.nonlinearity).lower()
        elif hasattr(model_a.rnn, "nonlinearity"):
            nonlinearity = str(model_a.rnn.nonlinearity).lower()
        else:
            raise AttributeError("Could not infer nonlinearity.")

    nonlinearity = nonlinearity.lower()
    model_b_aligned = copy.deepcopy(model_b)

    # --- Extract reference parameters (Model A) ---
    with torch.no_grad():
        W_ih_A = model_a.rnn.W_ih.data.float().cpu().numpy()
        W_hh_A = model_a.rnn.W_hh.data.float().cpu().numpy()
        b_h_A = (
            model_a.rnn.bias_h.data.float().cpu().numpy()
            if hasattr(model_a.rnn, "bias_h")
            else np.zeros(W_hh_A.shape[0])
        )
        W_dec_A = model_a.decoder.weight.data.float().cpu().numpy()

        W_ih_B = model_b.rnn.W_ih.data.float().cpu().numpy()
        W_hh_B = model_b.rnn.W_hh.data.float().cpu().numpy()
        b_h_B = (
            model_b.rnn.bias_h.data.float().cpu().numpy()
            if hasattr(model_b.rnn, "bias_h")
            else np.zeros(W_hh_B.shape[0])
        )
        W_dec_B = model_b.decoder.weight.data.float().cpu().numpy()

        # Enforce shape (H, I) for input weights
        if W_ih_A.shape[0] != W_hh_A.shape[0]:
            W_ih_A = W_ih_A.T
            W_ih_B = W_ih_B.T

    H = W_hh_A.shape[0]

    # Calculate unaligned baseline parameter distance
    unaligned_sq_dist = (
        np.sum((W_ih_A - W_ih_B) ** 2)
        + np.sum((b_h_A - b_h_B) ** 2)
        + np.sum((W_hh_A - W_hh_B) ** 2)
        + np.sum((W_dec_A - W_dec_B) ** 2)
    )
    unaligned_distance = float(np.sqrt(unaligned_sq_dist))

    # --- Construct Candidate Transformation Space ---
    all_permutations = list(itertools.permutations(range(H)))

    if nonlinearity == "tanh":
        # Tanh transformations: sign flips S in {-1, +1}^H
        transform_candidates = list(itertools.product([1.0, -1.0], repeat=H))
    else:
        # ReLU transformations: scaling factors D in R_>0^H
        # Grid around 1.0 (e.g., [0.5, 0.8, 1.0, 1.25, 2.0])
        scales = np.linspace(0.5, 2.0, scale_steps)
        transform_candidates = list(itertools.product(scales, repeat=H))

    best_distance = float("inf")
    best_perm = np.arange(H)
    best_transform_vec = np.ones(H, dtype=np.float32)

    # --- Fully Exhaustive Search ---
    for p_tuple in all_permutations:
        perm = np.array(p_tuple, dtype=int)
        P_mat = np.eye(H)[perm]  # Permutation matrix

        for t_tuple in transform_candidates:
            t_vec = np.array(t_tuple, dtype=np.float32)
            T_mat = np.diag(t_vec)
            T_inv = np.diag(1.0 / t_vec)

            # Combined transformation mapping B -> A: PT = T @ P
            PT = T_mat @ P_mat
            inv_PT = P_mat.T @ T_inv

            # Transform main network parameters
            W_ih_B_trans = PT @ W_ih_B
            b_h_B_trans = PT @ b_h_B
            W_hh_B_trans = inv_PT.T @ W_hh_B @ PT.T
            W_dec_B_trans = W_dec_B @ inv_PT

            # Compute full Euclidean parameter distance
            cand_sq_dist = (
                np.sum((W_ih_A - W_ih_B_trans) ** 2)
                + np.sum((b_h_A - b_h_B_trans) ** 2)
                + np.sum((W_hh_A - W_hh_B_trans) ** 2)
                + np.sum((W_dec_A - W_dec_B_trans) ** 2)
            )

            if cand_sq_dist < best_distance:
                best_distance = cand_sq_dist
                best_perm = perm
                best_transform_vec = t_vec

    # Safety check: if unaligned distance is already better than any symmetry shift, keep original
    if unaligned_distance < np.sqrt(best_distance):
        best_perm = np.arange(H)
        best_transform_vec = np.ones(H, dtype=np.float32)

    perm = best_perm
    transform_vec = best_transform_vec

    # --- STEP 3: Apply Optimal Transformation to PyTorch State ---
    with torch.no_grad():
        device = next(model_b_aligned.parameters()).device
        P_mat_t = torch.eye(H, device=device)[perm]
        T_mat_t = torch.diag(
            torch.tensor(transform_vec, device=device, dtype=torch.float32)
        )
        T_inv_t = torch.diag(
            torch.tensor(1.0 / transform_vec, device=device, dtype=torch.float32)
        )

        PT_t = T_mat_t @ P_mat_t
        inv_PT_t = P_mat_t.T @ T_inv_t

        rnn = model_b_aligned.rnn

        # Input weights
        if hasattr(rnn, "W_ih"):
            w_ih = rnn.W_ih.data
            is_transposed = w_ih.shape[0] != H
            w_ih_norm = w_ih.T if is_transposed else w_ih
            w_ih_aligned = PT_t @ w_ih_norm
            rnn.W_ih.data = w_ih_aligned.T if is_transposed else w_ih_aligned

        # Hidden bias
        if hasattr(rnn, "bias_h") and rnn.bias_h is not None:
            rnn.bias_h.data = PT_t @ rnn.bias_h.data

        if hasattr(rnn, "hidden_0") and rnn.hidden_0 is not None:
            rnn.hidden_0.data = PT_t @ rnn.hidden_0.data

        # Recurrent weights (Row-vector format: h @ W_hh)
        if hasattr(rnn, "W_hh"):
            rnn.W_hh.data = inv_PT_t.T @ rnn.W_hh.data @ PT_t.T

        # Gate weights transformation
        transform_gate_weights(rnn, perm, transform_vec)

        # Decoder weights
        model_b_aligned.decoder.weight.data = (
            model_b_aligned.decoder.weight.data @ inv_PT_t
        )

    # --- STEP 4: Calculate Final Post-Alignment Distance ---
    with torch.no_grad():
        W_ih_B_aligned = model_b_aligned.rnn.W_ih.data.float().cpu().numpy()
        W_hh_B_aligned = model_b_aligned.rnn.W_hh.data.float().cpu().numpy()
        b_h_B_aligned = (
            model_b_aligned.rnn.bias_h.data.float().cpu().numpy()
            if hasattr(model_b_aligned.rnn, "bias_h")
            else np.zeros(H)
        )
        W_dec_B_aligned = model_b_aligned.decoder.weight.data.float().cpu().numpy()

        if W_ih_B_aligned.shape[0] != H:
            W_ih_B_aligned = W_ih_B_aligned.T

        aligned_sq_dist = (
            np.sum((W_ih_A - W_ih_B_aligned) ** 2)
            + np.sum((b_h_A - b_h_B_aligned) ** 2)
            + np.sum((W_hh_A - W_hh_B_aligned) ** 2)
            + np.sum((W_dec_A - W_dec_B_aligned) ** 2)
        )
        aligned_distance = float(np.sqrt(aligned_sq_dist))

    info = {
        "perm": perm,
        "transform_vec": transform_vec,
        "unaligned_distance": unaligned_distance,
        "aligned_distance": aligned_distance
    }

    return model_b_aligned, info


def transform_gate_weights(
    rnn: torch.nn.Module,
    perm: np.ndarray,
    transform_vec: np.ndarray,
) -> None:
    """Transforms the gate parameters of a gated RNN module under hidden state permutation

    and sign-flip/scaling transformations (Godfrey et al. / Ainsworth et al.).

    Parameters
    ----------
    rnn : torch.nn.Module
        The RNN layer (e.g., ManualGRU, MonoGated, StereoGated, LightGRU).
    perm : np.ndarray
        Permutation indices array of shape (H,).
    transform_vec : np.ndarray
        Diagonal scaling or sign-flip vector of shape (H,).
    nonlinearity : str
        'relu' or 'tanh'.
    """
    H = len(perm)
    device = next(rnn.parameters()).device

    P_mat = torch.eye(H, device=device)[perm]
    T_mat = torch.diag(torch.tensor(transform_vec, device=device, dtype=torch.float32))
    T_inv = torch.diag(torch.tensor(1.0 / transform_vec, device=device, dtype=torch.float32))

    # PT maps B-space hidden units to A-space: h_A = PT @ h_B.
    PT = T_mat @ P_mat
    inv_PT = P_mat.T @ T_inv

    gate_names = ["z", "r", "i"]  # update, reset, input gates

    for g in gate_names:
        w_iz_attr = f"W_i{g}"
        w_hz_attr = f"W_h{g}"
        bias_z_attr = f"bias_{g}"

        # 1. Transform Recurrent Gate Weights (W_hz, W_hr, W_hi)
        if hasattr(rnn, w_hz_attr):
            w_hz = getattr(rnn, w_hz_attr)
            if w_hz is not None and isinstance(w_hz, torch.nn.Parameter):
                data = w_hz.data
                # Full H x H gate weight matrix (e.g., ManualGRU, LightGRU, MonoGated subnetwork)
                if data.shape == (H, H):
                    # Gate outputs are permuted, while their hidden inputs use PT.
                    w_hz.data = inv_PT.T @ data @ P_mat.T
                # 1D scalar gate weight vector (H x 1) (e.g., MonoGated default, StereoGated)
                elif data.shape == (H, 1):
                    # The scalar gate is not permuted as an output. Its hidden
                    # input must therefore be mapped by the inverse hidden transform.
                    w_hz.data = inv_PT.T @ data

        # 2. Transform Input Gate Weights (W_iz, W_ir, W_ii)
        if hasattr(rnn, w_iz_attr):
            w_iz = getattr(rnn, w_iz_attr)
            if w_iz is not None and isinstance(w_iz, torch.nn.Parameter):
                data = w_iz.data
                # Full I x H gate weight matrix
                if data.shape[1] == H:
                    # Account for potential transposition in parameter orientation
                    w_iz.data = data @ P_mat.T

        # 3. Transform Gate Biases (bias_z, bias_r, bias_i)
        if hasattr(rnn, bias_z_attr):
            bias_z = getattr(rnn, bias_z_attr)
            if bias_z is not None and isinstance(bias_z, torch.nn.Parameter):
                # Full H-dimensional gate bias
                if bias_z.data.shape == (H,):
                    bias_z.data = P_mat @ bias_z.data
                # Scalar 1D gate bias (shape (1,)) remains invariant

## CORE SIMILARITY FUNCTIONS ##

def compute_weight_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """
    Computes Cosine Similarity between the flattened parameters of Model A 
    and Model B. Assumes models are already canonicalized.

    This is equivalent to the normalised frobenius norm of the difference between the two parameter vectors.
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

# Computational functions #


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
    folder_name = Path(analysis_df.iloc[0]['save_path']).parts[3]
    if cache_path is None:
        cache_path = Path(
            f"NM_TinyRNN/data/analysis/mechanistic_variability_"
            f"{similarity_measure}_similarities_{folder_name}.htsv"
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
        model_b_aligned, alignment_info = align_rnns(model_a, model_b)
        
        record_b['model'] = model_b_aligned
        record_b['unaligned_parameter_distance'] = alignment_info['unaligned_distance']
        record_b['aligned_parameter_distance'] = alignment_info['aligned_distance']
        #we create a dictionary which we append to the results list for the final dataframe.
        results_subdict = {}
        for trial_subject in trial_records:
            if trial_subject != record_b['subject_id']:
                continue
            #initialise the results we want to append
            results_subdict['model_id'] = model_id
            results_subdict['subject_A'] = record_a['subject_id']
            results_subdict['model_A_outer_n'] = record_a['outer_loop_n']
            results_subdict['model_A_inner_n'] = record_a['inner_loop_idx']
            results_subdict['subject_B'] = record_b['subject_id']
            results_subdict['model_B_outer_n'] = record_b['outer_loop_n']
            results_subdict['model_B_inner_n'] = record_b['inner_loop_idx']
            results_subdict['trial_subject'] = trial_subject
            results_subdict['within_subject'] = record_a['subject_id'] == record_b['subject_id']
            results_subdict['unaligned_param_dist'] = record_b['unaligned_parameter_distance']
            results_subdict['aligned_param_dist'] = record_b['aligned_parameter_distance']

            #simplest thing to do is compute the distance in weight space.
            results_subdict['weight_similarity'] = compute_weight_similarity(model_a, model_b_aligned)

            #next, we run the model through trials and compute activation similarities:
            #extract trials for the subject we are comparing
            trials = trial_records[trial_subject]
            
            raw = torch.tensor(
                trials[["forced_choice", "choice", "outcome"]].to_numpy(),
                dtype=torch.float32,
            ).unsqueeze(0)
            inputs = ds.input_encoder(raw, model_a.input_encoding, model_a.input_forced_choice)
            model_device = next(model_a.parameters()).device
            inputs = inputs.to(model_device)
            
            if similarity_measure in ['pearson', 'cka']:
                hidden_sim = compute_activation_similarity(model_a, model_b_aligned, inputs, metric=similarity_measure)
            else:
                hidden_sim = compute_activation_similarity(model_a, model_b_aligned, inputs, metric='pearson')

            activations_a = _run_model_on_trials(model_a, trials, inputs)
            activations_b = _run_model_on_trials(model_b_aligned, trials, inputs)
            
            component_columns_a = _component_columns(activations_a)
            component_columns_b = _component_columns(activations_b)

            ## now we compute similarities
            for comp in COMPONENTS:
                column_name = f"{comp}_similarity"
                if comp == 'hidden':
                    similarity = hidden_sim
                else:
                    cols = [
                        column for column in activations_a.columns
                        if column.startswith(comp) and column in activations_b.columns
                    ]
                    if (comp not in component_columns_a or comp not in component_columns_b or not cols):
                        results_subdict[column_name] = np.nan
                        continue
                    
                    similarity = _similarity(
                        activations_a[cols].to_numpy().ravel(),
                        activations_b[cols].to_numpy().ravel(),
                        "pearson" if similarity_measure == 'cka' else similarity_measure, 
                    )
                    #appending activation similarities
                results_subdict[column_name] = similarity
            
           

            results.append(results_subdict)

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