"""Additions-only constrained alignment for the action-labeled GRU exports.

The alignment function is extracted from mechanistic_variability.align_rnns at
f36568933d5cd937e3944092c63998e46a458819, adding joint tanh permutation/sign
constraints. Gate transformations still use the unchanged upstream helper.
Existing mechanistic_variability.py and all monoGRU files remain unchanged.
"""

import copy
import itertools

import numpy as np
import torch
from scipy.optimize import least_squares

from NM_TinyRNN.code.measures.mechanistic_variability import transform_gate_weights


def align_rnns(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    nonlinearity: str | None = None,
    scale_steps: int = 5,
    refine_scaling: bool = True,
    scale_values: np.ndarray | None = None,
    allowed_permutations: list[tuple[int, ...]] | None = None,
    allowed_transformations: list[tuple[tuple[int, ...], tuple[float, ...]]] | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Align Model B to A using exhaustive permutations and sign/scale candidates.

    ReLU uses a scale grid followed by continuous refinement of its winner.
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
        candidate, keeping its permutation fixed (default: True). Tanh is
        unchanged. Set False only for explicit discrete-grid comparisons.
    scale_values : array-like, optional
        Custom positive ReLU grid. Unit scale is always included.
    allowed_permutations : list of tuples, optional
        Restrict unit permutations, e.g. to preserve semantic readout labels.
        The identity fallback is used only if identity is allowed. Default
        retains the historical exhaustive search.
    allowed_transformations : list of (permutation, signs), optional
        For tanh, restrict the joint permutation/sign candidates. This permits
        hard unit-label constraints whose required signs depend on permutation.
        The identity fallback must also obey this restriction. Default leaves
        existing callers unchanged.

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
    all_permutations = (list(itertools.permutations(range(H)))
                        if allowed_permutations is None else list(allowed_permutations))
    if not all_permutations or any(sorted(p) != list(range(H)) for p in all_permutations):
        raise ValueError('allowed_permutations must contain valid hidden-unit permutations')
    identity_allowed = tuple(range(H)) in [tuple(p) for p in all_permutations]

    permitted = None
    if allowed_transformations is not None:
        if nonlinearity != "tanh":
            raise ValueError("allowed_transformations currently supports tanh signed permutations only")
        permitted = set()
        for permutation, signs in allowed_transformations:
            if (sorted(permutation) != list(range(H)) or len(signs) != H
                    or not all(sign in (-1., 1.) for sign in signs)):
                raise ValueError("Expected valid permutations and one -1/+1 sign per hidden unit")
            permitted.add((tuple(permutation), tuple(signs)))
        if not permitted:
            raise ValueError("allowed_transformations must not be empty")
        identity_allowed = identity_allowed and (tuple(range(H)), (1.,) * H) in permitted

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
            if permitted is not None and (tuple(p_tuple), tuple(t_tuple)) not in permitted:
                continue
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

    if not np.isfinite(best_distance):
        raise ValueError("No joint transformation satisfies the supplied restrictions")

    # Safety check: if unaligned distance is already better than any permitted shift, keep original
    if identity_allowed and unaligned_distance < np.sqrt(best_distance):
        best_perm = np.arange(H)
        best_transform_vec = np.ones(H, dtype=np.float32)

    grid_transform_vec = best_transform_vec.copy()
    grid_squared_distance = (min(float(best_distance), float(unaligned_sq_dist))
                             if identity_allowed else float(best_distance))
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
