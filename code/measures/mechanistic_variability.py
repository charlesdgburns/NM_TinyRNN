"""Reusable mechanistic-variability computations and statistics.

The maintained post-training workflow is code/analyses/mechanistic_variability.py;
plot functions live in code/figures/mechanistic_variability.py. This module owns
canonical alignment, activation/parameter similarity, balanced block probes,
cache-selection validation, and subject-level reference contrasts.

compute_similarities retains the notebook cache API. Its legacy uncached path
uses continuous subject trials; the maintained study now uses that same
subject-B reference protocol. Historical shared-block helpers remain available
for explicit comparisons only.
"""

import os
import hashlib
import io
import pickle
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
from scipy.optimize import linear_sum_assignment, least_squares

## local imports
from NM_TinyRNN.code.models import experiments
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
    refine_scaling: bool = False,
    scale_values: np.ndarray | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Align Model B to A using exhaustive permutations and sign/scale candidates.

    ReLU uses a scale grid, with optional continuous refinement of its winner.
    The objective is Euclidean distance over the main network parameters.

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
    refine_scaling : bool
        For ReLU, continuously refine positive scales from the winning grid
        candidate, keeping its permutation fixed. Tanh is unchanged. Opt-in
        preserves existing discrete analyses and their caches.
    scale_values : array-like, optional
        Custom positive ReLU grid. Unit scale is always included.

    Returns
    -------
    model_b_aligned : torch.nn.Module
        Copy of Model B with aligned weights.
    info : dict
        Contains 'perm' (permutation array), 'transform_vec' (D or S),
        'unaligned_distance' and 'aligned_distance'.
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
        # Include unit scale for every permutation, not only the identity fallback.
        supplied_scales = (np.linspace(0.5, 2.0, scale_steps)
                           if scale_values is None else np.asarray(scale_values, dtype=float))
        if supplied_scales.ndim != 1 or supplied_scales.size == 0 or not np.all(np.isfinite(supplied_scales)) or np.any(supplied_scales <= 0):
            raise ValueError("scale_values must be a nonempty finite positive one-dimensional grid")
        scales = np.unique(np.r_[supplied_scales, 1.0])
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

    grid_transform_vec = best_transform_vec.copy()
    grid_squared_distance = min(float(best_distance), float(unaligned_sq_dist))
    refinement_success = None
    refinement_nfev = 0
    if refine_scaling and nonlinearity == "relu":
        # h @ W_hh convention: scaling contributes x_j - x_i to W_hh[i,j].
        eye = np.eye(H)
        powers = np.concatenate([
            np.repeat(eye, W_ih_B.shape[1], axis=0), eye,
            np.array([eye[j] - eye[i] for i in range(H) for j in range(H)]),
            np.tile(-eye, (W_dec_B.shape[0], 1)),
        ])
        def flatten_core(w_ih, bias, w_hh, decoder):
            return np.concatenate([w_ih.ravel(), bias.ravel(), w_hh.ravel(), decoder.ravel()]).astype(np.float64)
        target = flatten_core(W_ih_A, b_h_A, W_hh_A, W_dec_A)
        coeff = flatten_core(W_ih_B[best_perm], b_h_B[best_perm],
                             W_hh_B[np.ix_(best_perm, best_perm)], W_dec_B[:, best_perm])
        def residual(x):
            return coeff * np.exp(powers @ x) - target
        def jacobian(x):
            return (coeff * np.exp(powers @ x))[:, None] * powers
        initial_log_scales = np.log(best_transform_vec.astype(np.float64))
        fit = least_squares(residual, initial_log_scales, method="trf",
                            jac=jacobian, bounds=(-20., 20.), ftol=1e-10,
                            xtol=1e-10, gtol=1e-10, max_nfev=300)
        refinement_success = bool(fit.success)
        refinement_nfev = int(fit.nfev)
        scales = np.exp(fit.x).astype(np.float32)
        actual_residual = residual(np.log(scales.astype(np.float64)))
        if fit.success and float(actual_residual @ actual_residual) < grid_squared_distance:
            best_transform_vec = scales

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
        "aligned_distance": aligned_distance,
        "grid_distance": float(np.sqrt(grid_squared_distance)),
        "grid_transform_vec": grid_transform_vec,
        "refinement_success": refinement_success,
        "refinement_nfev": refinement_nfev,
        "refinement_method": "trf" if refine_scaling and nonlinearity == "relu" else None,
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
                correlations.append(float(np.corrcoef(act_a[:, i], act_b[:, i])[0, 1]))
        
        return float(np.mean(correlations)) if correlations else float('nan')

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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def extract_block_activations(model, inputs):
    model.eval()
    with torch.no_grad():
        if model.input_encoding == 'encoder':
            inputs = model.encoder(inputs)
        hidden, gates = model.rnn(inputs, return_gate_activations=True)
        logits = model.decoder(hidden)
    return dict(hidden=hidden.reshape(-1, hidden.shape[-1]).cpu().numpy(),
                logit=(logits[..., 0]-logits[..., 1]).flatten().cpu().numpy(),
                **{'gate_'+name: values.reshape(-1, values.shape[-1]).cpu().numpy() for name, values in gates.items()})


def component_pearson(a, b):
    # Same component score as mv._process_group; float64 avoids long-vector rounding.
    return float(_similarity(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), 'pearson'))


def unitwise_pearson(a, b):
    """Equal-weight mean of nonconstant corresponding unit correlations."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    valid = (a.std(0) > 1e-5) & (b.std(0) > 1e-5)
    scores = [component_pearson(a[:, i], b[:, i]) if valid[i] else float('nan')
              for i in range(a.shape[1])]
    return (float(np.mean(np.asarray(scores)[valid])) if valid.any() else float('nan'), scores)


def normalized_rows(values):
    values=np.asarray(values,dtype=np.float64)
    centered=values-values.mean(axis=1,keepdims=True)
    norms=np.linalg.norm(centered,axis=1,keepdims=True)
    return np.divide(centered,norms,out=np.full_like(centered,np.nan),where=norms>0)


def extract_subject_activations(models, raw):
    """Run every model on every complete subject sequence, without pooling trials.

    Subject sequences are padded only for batched evaluation. Real prefixes are
    kept; their state is initialized once, matching the original notebook.
    """
    subjects = sorted(raw)
    if models[0].input_encoding == 'encoder':
        return {subject: [extract_block_activations(model, raw[subject]) for model in models]
                for subject in subjects}, 0.
    lengths = [raw[subject].shape[1] for subject in subjects]
    encoded = [ds.input_encoder(raw[subject], models[0].input_encoding,
                               models[0].input_forced_choice)[0] for subject in subjects]
    padded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True)
    params, buffers = torch.func.stack_module_state([model.rnn for model in models])
    template = copy.deepcopy(models[0].rnn).to('meta')
    def forward(parameters, state):
        return torch.func.functional_call(template, (parameters, state), (padded,),
                                         dict(return_gate_activations=True))
    with torch.no_grad():
        hidden, gates = torch.vmap(forward, in_dims=(0, 0))(params, buffers)
        weights = torch.stack([model.decoder.weight for model in models])
        logits = torch.einsum('mbth,moh->mbto', hidden, weights)
        if models[0].decoder.bias is not None:
            logits += torch.stack([model.decoder.bias for model in models])[:, None, None, :]
    cached = {subject: [dict(hidden=hidden[i, j, :lengths[j]].numpy(),
        logit=(logits[i, j, :lengths[j], 0]-logits[i, j, :lengths[j], 1]).numpy(),
        **{'gate_'+key: values[i, j, :lengths[j]].numpy() for key, values in gates.items()})
        for i in range(len(models))] for j, subject in enumerate(subjects)}
    drift = 0.
    for i, j in [(0, 0), (len(models)-1, len(subjects)-1), (0, len(subjects)-1)]:
        subject = subjects[j]
        inputs = ds.input_encoder(raw[subject], models[i].input_encoding, models[i].input_forced_choice)
        actual = extract_block_activations(models[i], inputs)
        drift = max(drift, float(np.max(np.abs(actual['logit']-cached[subject][i]['logit']))))
        if not np.allclose(actual['logit'], cached[subject][i]['logit'], atol=2e-4, rtol=2e-4):
            raise ValueError('Batched subject evaluation changed model predictions')
    return cached, drift


def build_subject_trial_inputs(selected):
    """Recover each subject's actual saved trial order, as in the notebook."""
    inputs, records = {}, []
    for subject, rows in selected.sort_values(['subject_id', 'outer_loop_n']).groupby('subject_id'):
        path = Path(rows.iloc[0].trials_data_path)
        trials = pd.read_csv(path, sep='\t')
        inputs[subject] = torch.tensor(trials[['forced_choice', 'choice', 'outcome']].to_numpy(),
                                       dtype=torch.float32).unsqueeze(0)
        records.append(dict(subject_id=subject, n_trials=len(trials),
                            n_sessions=int(trials.session_folder_name.nunique()), source_trials_path=str(path)))
    return inputs, pd.DataFrame(records)


def _reference_correlations(cached, hidden_size):
    n = len(cached)
    hidden = np.stack([values['hidden'] for values in cached])
    normalized = normalized_rows(hidden.transpose(0, 2, 1).reshape(n*hidden_size, -1))
    logit = normalized_rows(np.stack([values['logit'] for values in cached]))
    permutations = list(itertools.permutations(range(hidden_size)))
    gates = {}
    for gate in ['update', 'reset']:
        key = 'gate_'+gate
        if key not in cached[0]:
            continue
        width = cached[0][key].shape[1]
        gate_perms = permutations if width == hidden_size else [tuple(range(width))]
        vectors = np.stack([values[key][:, perm].ravel() for values in cached for perm in gate_perms])
        norm = normalized_rows(vectors)
        gates[gate] = (norm@norm.T, len(gate_perms))
    return dict(hidden=hidden, std=hidden.std(axis=1), hidden_corr=normalized@normalized.T,
                logit_corr=logit@logit.T, gate_corr=gates)


def compute_reference_group(rows, inputs, output):
    """Score pairs on B's own trial sequence; tensor inputs retain legacy probe support."""
    torch.set_num_threads(1)
    rows = rows.sort_values(['subject_id', 'outer_loop_n']).reset_index(drop=True)
    model_id = rows.model_id.iloc[0]
    path = Path(output)/'groups'/f'{model_id}.htsv'
    models = []
    for row in rows.itertuples():
        with open(row.model_pickle_path, 'rb') as handle:
            models.append(CPUUnpickler(handle).load().cpu().eval())
    h = models[0].H
    permutations = list(itertools.permutations(range(h)))
    if isinstance(inputs, dict):
        cached_by_reference, batching_drift = extract_subject_activations(models, inputs)
    else:
        cached_by_reference = {'balanced_shared_reference': [extract_block_activations(model, inputs) for model in models]}
        batching_drift = 0.
    correlations = {reference: _reference_correlations(values, h)
                    for reference, values in cached_by_reference.items()}
    results, validated = [], set()
    max_drift = 0.
    for ia, ib in itertools.combinations(range(len(models)), 2):
        a, b = models[ia], models[ib]
        ra, rb = rows.iloc[ia], rows.iloc[ib]
        reference = rb.subject_id if isinstance(inputs, dict) else 'balanced_shared_reference'
        cached = cached_by_reference[reference]
        stats = correlations[reference]
        aligned, info = align_rnns(a, b)
        perm = tuple(int(value) for value in info['perm'])
        scale = np.asarray(info['transform_vec'])
        valid = (stats['std'][ia] > 1e-5) & (stats['std'][ib, list(perm)]*np.abs(scale) > 1e-5)
        unit_corr = np.array([stats['hidden_corr'][ia*h+u, ib*h+perm[u]]*np.sign(scale[u]) for u in range(h)])
        record = dict(model_id=model_id, model_type=ra.model_type, model_type2=ra.model_type2,
            hidden_size=h, nonlinearity=ra.nonlinearity, decoder_bias=bool(ra.decoder_bias),
            subject_A=ra.subject_id, model_A_outer_n=int(ra.outer_loop_n), model_A_inner_n=int(ra.inner_loop_idx),
            subject_B=rb.subject_id, model_B_outer_n=int(rb.outer_loop_n), model_B_inner_n=int(rb.inner_loop_idx),
            within_subject=ra.subject_id == rb.subject_id, trial_subject=reference,
            n_reference_steps=len(cached[0]['logit']),
            unaligned_param_dist=info['unaligned_distance'], aligned_param_dist=info['aligned_distance'],
            weight_similarity=compute_weight_similarity(a, aligned), valid_hidden_units=int(valid.sum()),
            hidden_similarity=float(unit_corr[valid].mean()) if valid.any() else np.nan,
            logit_value_similarity=float(stats['logit_corr'][ia, ib]))
        for gate in ['update', 'reset']:
            if gate not in stats['gate_corr']:
                record['gate_'+gate+'_similarity'] = np.nan
                continue
            matrix, count = stats['gate_corr'][gate]
            permutation_index = permutations.index(perm) if count > 1 else 0
            record['gate_'+gate+'_similarity'] = float(matrix[ia*count, ib*count+permutation_index])
        category = record['within_subject']
        if category not in validated:
            reference_inputs = (ds.input_encoder(inputs[reference], a.input_encoding, a.input_forced_choice)
                                if isinstance(inputs, dict) else inputs)
            actual = extract_block_activations(aligned, reference_inputs)
            drift = float(np.max(np.abs(actual['logit']-cached[ib]['logit'])))
            max_drift = max(max_drift, drift)
            if not np.allclose(actual['logit'], cached[ib]['logit'], atol=2e-4, rtol=2e-4):
                raise ValueError('Alignment changed logits')
            if not np.allclose(actual['hidden'], stats['hidden'][ib][:, perm]*scale, atol=2e-4, rtol=2e-4):
                raise ValueError('Hidden transform mismatch')
            score, _ = unitwise_pearson(cached[ia]['hidden'], actual['hidden'])
            if not np.isclose(score, record['hidden_similarity'], atol=2e-5, equal_nan=True):
                raise ValueError('Hidden correlation mismatch')
            for gate in stats['gate_corr']:
                direct = component_pearson(cached[ia]['gate_'+gate].ravel(), actual['gate_'+gate].ravel())
                if not np.isclose(direct, record['gate_'+gate+'_similarity'], atol=2e-5, equal_nan=True):
                    raise ValueError('Gate correlation mismatch')
            validated.add(category)
        results.append(record)
    result = pd.DataFrame(results)
    path.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(path, sep='\t', index=False)
    print(f'{model_id}: {len(result)} subject-reference pairs complete', flush=True)
    return dict(model_id=model_id, pairs=len(result), validated_pair_types=len(validated),
                max_validated_logit_drift=max_drift, batching_logit_drift=batching_drift)


def build_reference_probes(selected,config,data_root=None):
    from NM_TinyRNN.code.analyses import baseline
    from NM_TinyRNN.code.models import nested_cv
    data_root = Path(data_root) if data_root is not None else Path(__file__).resolve().parents[2]/'data/AB_behaviour'
    datasets={}
    for subject,rows in selected.groupby('subject_id'):
        row=rows.iloc[0].to_dict()
        datasets[subject]=baseline.load_dataset(data_root,subject,row,config['sequence_length'],config['sessions_per_subject'])
    count=min(len(d.inputs) for d in datasets.values());inputs=[];records=[];block_counts=[]
    for subject,dataset in datasets.items():
        indices=np.linspace(0,len(dataset.inputs)-1,count,dtype=int)
        inputs.append(dataset.inputs[indices])
        mapping=dataset.subject_df.dropna(subset=['sequence_block_idx']).groupby('sequence_block_idx').first()
        for i in indices:
            row=mapping.loc[i]
            records.append(dict(subject_id=subject,block_index=int(i),session=row.session_folder_name,first_trial=int(row.session_trial_idx)))
        splits={outer:nested_cv.nested_cv_splits(dataset,10,outer)['outer_eval'] for outer in range(1,11)}
        for outer,indices_ in splits.items():
            block_counts.append(dict(subject_id=subject,outer_loop_n=outer,eval_block_n_free=int((dataset.forced_choice_mask[indices_]==0).sum())))
    return torch.cat(inputs).cpu(),pd.DataFrame(records),pd.DataFrame(block_counts)


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p)
    adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result=np.empty_like(p);result[order]=np.minimum(adjusted,1)
    return result


def contrast(frame):
    """Matching subject/dyad means; return paired effect and node-jackknife SE."""
    from scipy.stats import t
    frame=frame.dropna(subset=['reference','model']).copy()
    subjects=sorted(set(frame.subject_A)|set(frame.subject_B));n=len(subjects)
    differences=frame.model-frame.reference;mean=float(differences.mean())
    by_subject=np.array([differences[frame.subject_A.eq(s)|frame.subject_B.eq(s)].mean() for s in subjects])
    sd=float(by_subject.std(ddof=1))
    dz=mean/sd if sd>0 else (0. if mean==0 else np.nan)
    replicate=np.array([differences[~(frame.subject_A.eq(s)|frame.subject_B.eq(s))].mean() for s in subjects])
    se=float(np.sqrt((n-1)/n*np.sum((replicate-replicate.mean())**2)))
    p=float(2*t.sf(abs(mean/se),n-1)) if se>0 else (1. if mean==0 else 0.)
    margin=float(t.ppf(.975,n-1)*se)
    return dict(mean_similarity=float(frame.model.mean()),reference_mean=float(frame.reference.mean()),
        delta_similarity=mean,cohens_dz=dz,n_subjects=n,n_subject_pairs=len(frame),
        jackknife_se=se,delta_ci95_low=mean-margin,delta_ci95_high=mean+margin,p_raw=p)



SIMILARITY_METRICS = ['weight_similarity', 'hidden_similarity', 'gate_update_similarity',
                      'gate_reset_similarity', 'logit_value_similarity']
STANDARD_GRU_REFERENCE = '2_unit_GRU_tanh_unipolar_forced'


def summarize_reference_pairs(pairs):
    """Average overlapping fold pairs within each subject or unordered subject pair."""
    long = pairs.melt(id_vars=['model_id', 'within_subject', 'subject_A', 'subject_B'],
                      value_vars=SIMILARITY_METRICS, var_name='metric', value_name='similarity')
    subject = long.groupby(['model_id', 'within_subject', 'subject_A', 'subject_B', 'metric'],
                          as_index=False).agg(mean_similarity=('similarity', 'mean'),
                                              n_pairs=('similarity', 'count'))
    summary = subject.groupby(['model_id', 'within_subject', 'metric']).mean_similarity.agg(
        ['mean', 'std', 'min', 'max', 'count']).reset_index()
    return subject, summary


def compare_similarity_to_reference(subject_pairs, reference=STANDARD_GRU_REFERENCE):
    """Paired subject-level dz and subject-deletion inference, globally Holm corrected."""
    if reference not in set(subject_pairs.model_id):
        raise ValueError(f'Reference model missing: {reference}')
    rows = []
    for scope, metric in itertools.product([True, False], SIMILARITY_METRICS):
        subset = subject_pairs[subject_pairs.within_subject.eq(scope) & subject_pairs.metric.eq(metric)]
        baseline = subset[subset.model_id.eq(reference)][['subject_A', 'subject_B', 'mean_similarity']].rename(
            columns={'mean_similarity': 'reference'})
        for key, group in subset.groupby('model_id'):
            if group.mean_similarity.notna().sum() == 0:
                rows.append(dict(model_id=key, within_subject=scope, metric=metric, available=False))
                continue
            merged = baseline.merge(group[['subject_A', 'subject_B', 'mean_similarity']],
                on=['subject_A', 'subject_B'], validate='one_to_one').rename(columns={'mean_similarity': 'model'})
            rows.append(dict(model_id=key, within_subject=scope, metric=metric,
                             available=True, **contrast(merged)))
    table = pd.DataFrame(rows)
    tests = table.available & table.model_id.ne(reference)
    table['p_holm'] = np.nan
    table.loc[tests, 'p_holm'] = holm(table.loc[tests, 'p_raw'])
    table['significance'] = ''
    for cutoff, mark in [(0.05, '*'), (.01, '**'), (.001, '***')]:
        table.loc[table.p_holm.lt(cutoff), 'significance'] = mark
    return table


COMPONENTS = ['hidden', 'gate_update', 'gate_reset', 'logit_value']

def validate_similarity_cache_models(cached_df, analysis_df):
    """Reject a cache from different model/subject/fold selections."""
    keys = ['model_id', 'subject_id', 'outer_loop_n', 'inner_loop_idx']
    missing = set(keys)-set(analysis_df.columns)
    if missing:
        raise ValueError(f'Cannot validate similarity cache: input is missing {sorted(missing)}')
    counts = analysis_df.groupby('model_id').size()
    expected_frame = analysis_df.loc[analysis_df.model_id.isin(counts[counts >= 2].index), keys]
    expected = set(expected_frame.itertuples(index=False, name=None))
    if len(expected) != len(expected_frame):
        raise ValueError('Similarity input contains duplicate model/subject/fold rows')
    observed = set()
    for side in ['A', 'B']:
        columns = ['model_id', f'subject_{side}', f'model_{side}_outer_n', f'model_{side}_inner_n']
        if not set(columns).issubset(cached_df.columns):
            raise ValueError('Similarity cache lacks model endpoint identifiers; rebuild it before loading')
        observed.update(cached_df[columns].itertuples(index=False, name=None))
    pair_counts = cached_df.groupby('model_id').size().to_dict()
    expected_counts = {key: int(n*(n-1)//2) for key, n in counts.items() if n >= 2}
    if observed != expected or pair_counts != expected_counts:
        raise ValueError(
            'Similarity cache does not match the supplied model selection '
            f'({len(expected-observed)} selected endpoints missing, '
            f'{len(observed-expected)} unexpected endpoints). '
            'Rebuild the cache for this selection or use a separate cache_path.'
        )


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
    default_cache = cache_path is None
    if cache_path is None:
        cache_path = Path("NM_TinyRNN/data/analysis/mechanistic_variability") / (
            'pairwise_similarities.htsv' if similarity_measure == 'pearson'
            else f'pairwise_similarities_{similarity_measure}.htsv')
    else:
        cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        if default_cache:
            selected_path = cache_path.parent/'selected_models.htsv'
            if not selected_path.is_file():
                raise ValueError('Default similarity cache lacks selected-model provenance; rerun the analysis.')
            saved = pd.read_csv(selected_path, sep='\t')
            from NM_TinyRNN.code.measures.analysis import cache_matches_models
            if not cache_matches_models(saved, analysis_df):
                raise ValueError('Similarity cache uses different model paths or selection; rerun the analysis.')
        cached_df = pd.read_csv(cache_path, sep="\t")
        validate_similarity_cache_models(cached_df, analysis_df)
        print(f"Loaded similarity dataframe from cache: {cache_path} (model selection verified)")
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
    if default_cache:
        analysis_df.to_csv(cache_path.parent/'selected_models.htsv', sep="\t", index=False)
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
        
        # Keep original models fixed across pairwise comparisons.
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

    experiment_df = experiments.get_DA_info_df()
    all_models_df = analysis.get_analysis_df(experiment_df, mode='all')
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

    output_dir = Path('NM_TinyRNN/data/analysis/mechanistic_variability')
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / 'selected_models.htsv'
    similarity_path = output_dir / 'pairwise_similarities.htsv'
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
