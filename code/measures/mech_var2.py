
import copy
import itertools
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def align_rnns(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    nonlinearity: str | None = None,
    max_iters: int = 10,
    tol: float = 1e-5,
) -> tuple[torch.nn.Module, float, dict]:
    """Aligns Model B to Model A using exact symmetry alignment."""
    if nonlinearity is None:
        if hasattr(model_a, "nonlinearity"):
            nonlinearity = str(model_a.nonlinearity).lower()
        elif hasattr(model_a.rnn, "nonlinearity"):
            nonlinearity = str(model_a.rnn.nonlinearity).lower()
        else:
            raise AttributeError("Could not infer nonlinearity.")

    nonlinearity = nonlinearity.lower()
    model_b_aligned = copy.deepcopy(model_b)

    # --- Extract parameters as float numpy arrays ---
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

        if W_ih_A.shape[0] != W_hh_A.shape[0]:
            W_ih_A = W_ih_A.T
            W_ih_B = W_ih_B.T

    H = W_hh_A.shape[0]

    unaligned_sq_dist = (
        np.sum((W_ih_A - W_ih_B) ** 2)
        + np.sum((b_h_A - b_h_B) ** 2)
        + np.sum((W_hh_A - W_hh_B) ** 2)
        + np.sum((W_dec_A - W_dec_B) ** 2)
    )
    unaligned_distance = float(np.sqrt(unaligned_sq_dist))

    best_distance = float("inf")
    best_perm = np.arange(H)
    best_transform_vec = np.ones(H, dtype=np.float32)

    if nonlinearity == "tanh" and H <= 10:
        # --- EXHAUSTIVE SIGN SEARCH FOR TANH ---
        for signs in itertools.product([1.0, -1.0], repeat=H):
            s_vec = np.array(signs, dtype=np.float32)
            S_mat = np.diag(s_vec)
            S_inv = np.diag(1.0 / s_vec)

            W_ih_B_trans = S_mat @ W_ih_B
            b_h_B_trans = S_mat @ b_h_B
            W_dec_B_trans = W_dec_B @ S_inv

            cost_matrix = np.zeros((H, H))
            for i in range(H):
                for j in range(H):
                    c_in = np.sum((W_ih_A[i, :] - W_ih_B_trans[j, :]) ** 2)
                    c_bias = (b_h_A[i] - b_h_B_trans[j]) ** 2
                    c_out = np.sum((W_dec_A[:, i] - W_dec_B_trans[:, j]) ** 2)
                    c_rec_self = (W_hh_A[i, i] - W_hh_B[j, j]) ** 2
                    cost_matrix[i, j] = c_in + c_bias + c_out + c_rec_self

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cand_perm = col_ind

            # Total candidate error calculation
            W_ih_B_p = W_ih_B_trans[cand_perm, :]
            b_h_B_p = b_h_B_trans[cand_perm]
            W_dec_B_p = W_dec_B_trans[:, cand_perm]
            W_hh_B_p = S_inv @ W_hh_B[cand_perm, :][:, cand_perm] @ S_mat

            cand_dist = np.sqrt(
                np.sum((W_ih_A - W_ih_B_p) ** 2)
                + np.sum((b_h_A - b_h_B_p) ** 2)
                + np.sum((W_hh_A - W_hh_B_p) ** 2)
                + np.sum((W_dec_A - W_dec_B_p) ** 2)
            )

            if cand_dist < best_distance:
                best_distance = cand_dist
                best_perm = cand_perm
                best_transform_vec = s_vec

        perm = best_perm
        transform_vec = best_transform_vec

    else:
        # --- SIGN-AWARE / SCALE-AWARE PAIRWISE COST MATRIX ---
        transform_vec = np.ones(H, dtype=np.float32)

        for iteration in range(max_iters):
            t_prev = transform_vec.copy()
            cost_matrix = np.zeros((H, H))
            pair_transform = np.ones((H, H))

            for i in range(H):
                for j in range(H):
                    if nonlinearity == "tanh":
                        # Test both signs for candidate mapping (i, j)
                        err_pos = (
                            np.sum((W_ih_A[i, :] - W_ih_B[j, :]) ** 2)
                            + (b_h_A[i] - b_h_B[j]) ** 2
                            + np.sum((W_dec_A[:, i] - W_dec_B[:, j]) ** 2)
                        )
                        err_neg = (
                            np.sum((W_ih_A[i, :] + W_ih_B[j, :]) ** 2)
                            + (b_h_A[i] + b_h_B[j]) ** 2
                            + np.sum((W_dec_A[:, i] + W_dec_B[:, j]) ** 2)
                        )
                        c_rec = (W_hh_A[i, i] - W_hh_B[j, j]) ** 2

                        if err_pos <= err_neg:
                            cost_matrix[i, j] = err_pos + c_rec
                            pair_transform[i, j] = 1.0
                        else:
                            cost_matrix[i, j] = err_neg + c_rec
                            pair_transform[i, j] = -1.0

                    elif nonlinearity == "relu":
                        norm_in_A = np.sum(W_ih_A[i, :] ** 2) + (b_h_A[i] ** 2)
                        norm_in_B = np.sum(W_ih_B[j, :] ** 2) + (b_h_B[j] ** 2) + 1e-8
                        norm_out_A = np.sum(W_dec_A[:, i] ** 2)
                        norm_out_B = np.sum(W_dec_B[:, j] ** 2) + 1e-8
                        #scaling
                        d_ij = np.sqrt(np.sqrt((norm_in_A / norm_in_B) / (norm_out_A / norm_out_B)))
                        d_ij = np.clip(d_ij, 0.1, 10.0)

                        c_in = np.sum((W_ih_A[i, :] - d_ij * W_ih_B[j, :]) ** 2)
                        c_bias = (b_h_A[i] - d_ij * b_h_B[j]) ** 2
                        c_out = np.sum((W_dec_A[:, i] - W_dec_B[:, j] / d_ij) ** 2)
                        c_rec = (W_hh_A[i, i] - W_hh_B[j, j]) ** 2

                        cost_matrix[i, j] = c_in + c_bias + c_out + c_rec
                        pair_transform[i, j] = d_ij

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            perm = col_ind
            for i in range(H):
                transform_vec[i] = pair_transform[i, col_ind[i]]

            if np.max(np.abs(transform_vec - t_prev)) < tol:
                break

    # STEP 3: Apply transformation in PyTorch state
    with torch.no_grad():
        P_mat = torch.eye(H)[perm]
        T_mat_t = torch.diag(torch.tensor(transform_vec, dtype=torch.float32))
        T_inv_t = torch.diag(torch.tensor(1.0 / transform_vec, dtype=torch.float32))

        PT = T_mat_t @ P_mat
        inv_PT = P_mat.T @ T_inv_t

        rnn = model_b_aligned.rnn

        if hasattr(rnn, "W_ih"):
            w_ih = rnn.W_ih.data
            is_transposed = w_ih.shape[0] != H
            w_ih_norm = w_ih.T if is_transposed else w_ih
            w_ih_aligned = PT @ w_ih_norm
            rnn.W_ih.data = w_ih_aligned.T if is_transposed else w_ih_aligned

        if hasattr(rnn, "bias_h") and rnn.bias_h is not None:
            rnn.bias_h.data = PT @ rnn.bias_h.data

        if hasattr(rnn, "W_hh"):
            rnn.W_hh.data = inv_PT.T @ rnn.W_hh.data @ PT.T

        transform_gate_weights(rnn, perm, transform_vec, nonlinearity)

        model_b_aligned.decoder.weight.data = (
            model_b_aligned.decoder.weight.data @ inv_PT
        )

    # --- STEP 4: Calculate True Global Post-Alignment Distance ---
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
    }

    return model_b_aligned, aligned_distance, info


def transform_gate_weights(
    rnn: torch.nn.Module,
    perm: np.ndarray,
    transform_vec: np.ndarray,
    nonlinearity: str = "tanh",
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
                    w_hz.data = P_mat @ data

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