"""Code to compute effective dimensionality of hidden states,
similar to """
import torch
import numpy as np

def effective_dimensionality(Z):
    '''Z: numpy array of hidden state activations, shape (n_trials, n_hidden)'''
    matrix_product = Z.T @ Z
    # 2. Compute the eigenvalues // Since Z^T * Z is symmetric, np.linalg.eigvalsh is faster and numerically stable
    eigenvalues = np.linalg.eigvalsh(matrix_product)
    # 3. Calculate the Participation Ratio (Effective Dimensionality)
    effective_dim = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    return effective_dim
