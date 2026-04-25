"""
nested_cv.py  —  Nested cross-validation for AB_Dataset

Outer loop
----------
Split K session-chunks.  One chunk is the outer eval set (held out entirely).
The remaining K-1 chunks form the pool for inner cross-validation.

    outer_k ∈ {0 … K-1}  selects which chunk is the eval set.

Inner loop
----------
Within the K-1 inner chunks, rotate a single validation chunk through every
position.  Each rotation is one "inner fold" (K-1 folds total).

    inner_fold ∈ {0 … K-2}

Parallelism
-----------
CPU path  : joblib dispatches each inner fold to a separate subprocess.
            Inside every subprocess, TrainerGPU stacks all hyperparameter /
            seed combos and runs them together via torch.func.vmap.
GPU path  : joblib is skipped (single process); vmap still parallelises
            across configs within each inner fold.

Determinism
-----------
A global train_seed is combined with the inner-fold index to produce a
unique, reproducible seed for every subprocess:

    effective_seed = train_seed * 1000 + inner_fold_idx

This means two identical calls with the same train_seed always produce the
same result, even when joblib reorders subprocesses.

Saving
------
Pass save_path to run_outer_fold and results are written automatically:

    save_path/
      outer_fold_N/
        inner_fold_M/
          info.json
          model_state.pth
          trials_data.htsv
          training_losses.htsv   (if available)

Memory estimate (optional helper)
----------------------------------
call  estimate_memory_per_fold(base_model, dataset, n_inner_folds, configs)
to get a rough per-subprocess upper bound in MiB.
"""

from __future__ import annotations


import random

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from joblib import Parallel, delayed

# ── relative imports (adjust if your package layout differs) ──────────────────
from NM_TinyRNN.code.models.datasets import AB_Dataset
from NM_TinyRNN.code.models.nested_cv_io import save_inner_fold_results


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Session-level chunk splitting
# ─────────────────────────────────────────────────────────────────────────────

def _get_sessions(dataset: AB_Dataset) -> np.ndarray:
    """Sorted array of unique session folder names."""
    return np.array(sorted(dataset.subject_df["session_folder_name"].unique()))


def _session_chunks(sessions: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Divide sessions into k roughly equal chunks.

    Sessions are sorted before splitting so the partition is deterministic and
    independent of pandas insertion order.
    """
    return [chunk for chunk in np.array_split(sessions, k) if len(chunk)]


def _blocks_for_sessions(dataset: AB_Dataset, session_names) -> list[int]:
    """Return sorted sequence-block indices for the given session names."""
    block_map = dataset.subject_df.dropna().groupby("session_folder_name")[
        "sequence_block_idx"
    ].unique()
    blocks = sorted(
        np.concatenate([block_map[s] for s in session_names if s in block_map]).tolist()
    )
    blocks = [int(x) for x in blocks] #convert float to int, very important
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Nested split factory
# ─────────────────────────────────────────────────────────────────────────────

def nested_cv_splits(
    dataset: AB_Dataset,
    n_outer_loops: int = 5,
    outer_loop_number: int = 1,
) -> dict:
    """
    Compute index splits for one outer-loop iteration.

    Parameters
    ----------
    dataset          : AB_Dataset instance
    n_outer_loops    : total number of outer folds (K)
    outer_loop_number: which outer fold to run (1-indexed, 1 … K)

    Returns
    -------
    dict with keys:
        "outer_eval"   : list of sequence-block indices for the outer eval set
        "inner_folds"  : list of dicts, each with keys
                             "train"  : block indices
                             "val"    : block indices
                             "fold_idx": int (position within inner loop)
        "n_outer_loops": int
        "outer_loop_number": int
    """
    if not (1 <= outer_loop_number <= n_outer_loops):
        raise ValueError(
            f"outer_loop_number must be in [1, {n_outer_loops}], "
            f"got {outer_loop_number}."
        )
    
    sessions = _get_sessions(dataset)
    chunks   = _session_chunks(sessions, n_outer_loops)   # list of K arrays

    # ── outer eval: the designated chunk ─────────────────────────────────────
    outer_loop_idx = outer_loop_number-1
    outer_eval_sessions = chunks[outer_loop_idx] 
    outer_eval_blocks   = _blocks_for_sessions(dataset, outer_eval_sessions)

    # ── inner pool: the remaining K-1 chunks ─────────────────────────────────
    inner_chunks = [c for i, c in enumerate(chunks) if i != outer_loop_idx]
    # inner_chunks has K-1 elements; rotate val across each position
    inner_folds = []
    for fold_idx, val_chunk in enumerate(inner_chunks):
        train_sessions = np.concatenate(
            [c for i, c in enumerate(inner_chunks) if i != fold_idx]
        )
        val_sessions   = val_chunk

        inner_folds.append(
            {
                "fold_idx": fold_idx,
                "train":    _blocks_for_sessions(dataset, train_sessions),
                "val":      _blocks_for_sessions(dataset, val_sessions),
            }
        )

    return {
        "outer_eval":        outer_eval_blocks,
        "inner_folds":       inner_folds,
        "n_outer_loops":     n_outer_loops,
        "outer_loop_number": outer_loop_number,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Deterministic seed helper
# ─────────────────────────────────────────────────────────────────────────────

def _fold_seed(train_seed: int, inner_fold_idx: int) -> int:
    """
    Derive a unique, reproducible integer seed for a given inner fold.

    The formula is deliberately simple so it is easy to reason about:
        seed = train_seed * 10_000 + inner_fold_idx

    With train_seed < 10_000 this guarantees no collisions across folds.
    """
    return train_seed * 10_000 + inner_fold_idx


def _set_global_seeds(seed: int) -> None:
    """Pin every relevant RNG in a subprocess."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slight speed cost, worth it for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Single inner-fold worker  (runs inside a subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _run_inner_fold(
    fold_info:          dict,
    base_model,
    dataset:            AB_Dataset,
    trainer_kwargs:     dict,
    train_seed:         int,
    save_path:          Optional[Path],
    outer_loop_number:  int,
    outer_eval_blocks:  list
) -> dict:
    """
    Train across all hyperparameter configs for one inner fold.

    This function is the joblib work unit.  It:
      1. Sets all RNGs deterministically.
      2. Builds a TrainerGPU with the fold-specific splits baked in.
      3. Saves artefacts to save_path/outer_fold_N/inner_fold_M/ (if save_path given).
      4. Returns the best state-dict and config for this fold.

    Parameters
    ----------
    fold_info          : one element of nested_cv_splits()["inner_folds"]
    base_model         : un-initialised model template (deepcopied inside TrainerGPU)
    dataset            : AB_Dataset (read-only inside the subprocess)
    trainer_kwargs     : extra kwargs forwarded to TrainerGPU (lambdas, lr, …)
    train_seed         : base seed; combined with fold index for uniqueness
    save_path          : root directory for artefacts; None disables saving
    outer_loop_number  : outer fold index, used only for the directory name
    outer_eval_blocks  : list of block indices used for final evaluation
    
    Returns
    -------
    dict with keys "fold_idx", "state_dict", "config", "val_loss"
    """
    # Import here so the worker subprocess picks up the right CUDA context
    from NM_TinyRNN.code.models.nested_cv import _set_global_seeds   # self-import is fine in a subprocess

    fold_idx = fold_info["fold_idx"]
    seed     = _fold_seed(train_seed, fold_idx)
    #_set_global_seeds(seed)

    # Build split dict in the format get_dataloader already expects
    splits = {
        "train": fold_info["train"],
        "val":   fold_info["val"],
        "eval":  outer_eval_blocks,
    }

    # TrainerGPU uses dataset._session_split() internally; we monkey-patch it
    # with a lambda that returns our pre-computed splits instead.
    patched_dataset = deepcopy(dataset)
    patched_dataset._session_split = lambda **_kw: splits   # noqa: E731

    # TrainerGPU gets its own temp save path so it doesn't write into the
    # nested CV directory structure.  The canonical save is handled below.
    from NM_TinyRNN.code.models.training_fast import TrainerGPU   # adjust path
    trainer = TrainerGPU(
        train_seed = seed,
        **trainer_kwargs,
    )
  
    best_state_dict, best_config = trainer.fit(base_model, patched_dataset)
    best_val_loss = trainer._last_best_val_loss   # see note *

    result = {
        "fold_idx":   fold_idx,
        "state_dict": best_state_dict,
        "config":     best_config,
        "val_loss":   float(best_val_loss),
    }

    # ── Save artefacts ────────────────────────────────────────────────────────
    if save_path is not None:
        save_inner_fold_results(
            result            = result,
            base_model        = base_model,
            dataset           = dataset,
            splits            = splits,
            trainer           = trainer,
            save_path         = save_path,
            outer_loop_number = outer_loop_number,
            inner_fold_idx    = fold_idx,
            outer_eval_blocks = outer_eval_blocks
        )

    return result


# * TrainerGPU.fit() does not currently expose the winning val loss as a return
#   value.  Add one line to the end of TrainerGPU.fit():
#       self._last_best_val_loss = best_val_losses[best_idx].item()
#   before the return statement.  This patch keeps TrainerGPU self-contained.


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_outer_fold(
    base_model,
    dataset:            AB_Dataset,
    outer_loop_number:  int        = 1,
    n_outer_loops:      int        = 10,
    trainer_kwargs:     dict       = None,
    train_seed:         int        = 42,
    save_path:          Path       = None,
    n_jobs:             int        = -1,
    prefer_backend:     str        = "loky",
) -> dict:
    """
    Run one outer-loop iteration of nested cross-validation.

    For each inner fold (K-1 total) a separate subprocess is launched via
    joblib.  Inside each subprocess TrainerGPU runs all hyperparameter /
    weight-seed combos in parallel using torch.func.vmap.

    On CUDA machines, joblib parallelism is automatically disabled (n_jobs=1)
    because multiple processes sharing a GPU context is unsafe and vmap already
    saturates the device.

    Parameters
    ----------
    base_model         : un-initialised model template
    dataset            : AB_Dataset instance
    outer_loop_number  : which outer chunk is the eval set (1-indexed)
    n_outer_loops      : total number of outer folds K
    trainer_kwargs     : dict forwarded to TrainerGPU (sparsity_lambdas, etc.)
                         Do not include save_path here; use the save_path argument.
    train_seed         : base RNG seed; fold seeds are derived from this
    save_path          : root directory for all saved artefacts.  If None,
                         nothing is written to disk.
                         Written layout:
                           save_path/outer_fold_N/inner_fold_M/{info.json,
                           model_state.pth, trials_data.htsv, …}
    n_jobs             : joblib n_jobs.  -1 = all CPUs.  Ignored on CUDA.
    prefer_backend     : joblib backend ("loky" or "multiprocessing")

    Returns
    -------
    dict with keys:
        "outer_loop_number" : int
        "outer_eval_blocks" : list[int]  (outer eval block indices)
        "inner_results"     : list[dict] (one per inner fold, sorted by fold_idx)
        "best_inner_fold"   : dict       (inner result with lowest val_loss)
    """
    if trainer_kwargs is None:
        trainer_kwargs = {}

    # Guard: save_path must not be smuggled inside trainer_kwargs
    if "save_path" in trainer_kwargs:
        raise ValueError(
            "Do not put save_path inside trainer_kwargs. "
            "Pass it as the save_path argument to run_outer_fold instead."
        )

    save_path = Path(save_path) if save_path is not None else None

    # ── build splits ──────────────────────────────────────────────────────────
    splits = nested_cv_splits(dataset, n_outer_loops, outer_loop_number)
    inner_folds       = splits["inner_folds"]
    outer_eval_blocks = splits["outer_eval"]

    print(
        f"[outer {outer_loop_number}/{n_outer_loops}]  "
        f"outer eval: {len(outer_eval_blocks)} blocks  |  "
        f"{len(inner_folds)} inner folds"
        + (f"  |  saving to {save_path}" if save_path else "")
    )

    # ── decide on parallelism ─────────────────────────────────────────────────
    on_gpu   = torch.cuda.is_available()
    eff_jobs = 1 if on_gpu else n_jobs

    if on_gpu and n_jobs != 1:
        print(
            "  CUDA detected: running inner folds sequentially "
            "(vmap parallelises across configs within each fold)."
        )

    # ── dispatch inner folds ──────────────────────────────────────────────────
    def _worker(fold_info):
        return _run_inner_fold(
            fold_info         = fold_info,
            base_model        = deepcopy(base_model),
            dataset           = dataset,
            trainer_kwargs    = deepcopy(trainer_kwargs),
            train_seed        = train_seed,
            save_path         = save_path,
            outer_loop_number = outer_loop_number,
            outer_eval_blocks = outer_eval_blocks
        )

    if eff_jobs == 1:
        inner_results = [_worker(fi) for fi in inner_folds]
    else:
        inner_results = Parallel(n_jobs=eff_jobs, backend=prefer_backend)(
            delayed(_worker)(fi) for fi in inner_folds
        )

    inner_results.sort(key=lambda r: r["fold_idx"])

    # ── pick the inner fold whose val_loss was lowest ─────────────────────────
    best_inner = min(inner_results, key=lambda r: r["val_loss"])

    return {
        "outer_loop_number": outer_loop_number,
        "outer_eval_blocks": outer_eval_blocks,
        "inner_results":     inner_results,
        "best_inner_fold":   best_inner,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Optional: memory estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_memory_per_fold(
    base_model,
    dataset:      AB_Dataset,
    n_configs:    int,
    n_inner_folds: int,
    batch_size:   int = 16,
    dtype_bytes:  int = 4,       # float32
) -> dict:
    """
    Rough upper-bound on memory consumed by ONE inner-fold subprocess (MiB).

    This is intentionally conservative: it counts:
      - N stacked model parameter copies  (vmap)
      - One batch of activations per config (approximated as 2× param count)
      - Adam optimizer state (2× params for m/v)
      - The dataset tensors (shared read-only across forks with loky)

    Parameters
    ----------
    base_model    : model whose parameter count will be inspected
    dataset       : AB_Dataset (for tensor sizes)
    n_configs     : total number of (lambda × seed) combos in TrainerGPU
    n_inner_folds : used only to warn if total memory looks risky
    batch_size    : TrainerGPU batch_size
    dtype_bytes   : bytes per element (4 for float32)

    Returns
    -------
    dict with human-readable estimates per component and totals
    """
    n_params = sum(p.numel() for p in base_model.parameters())

    model_stack_mib   = n_params * n_configs * dtype_bytes / 1024**2
    activation_mib    = model_stack_mib * 2   # rough forward-pass buffer
    optimizer_mib     = model_stack_mib * 2   # Adam m + v
    dataset_mib       = (
        dataset.inputs.nelement() + dataset.targets.nelement()
    ) * dtype_bytes / 1024**2

    per_fold_mib  = model_stack_mib + activation_mib + optimizer_mib
    total_mib     = per_fold_mib * n_inner_folds + dataset_mib   # dataset shared once

    return {
        "params_per_model":       n_params,
        "n_configs":              n_configs,
        "model_stack_MiB":        round(model_stack_mib, 2),
        "activation_buffer_MiB":  round(activation_mib, 2),
        "optimizer_state_MiB":    round(optimizer_mib, 2),
        "dataset_tensors_MiB":    round(dataset_mib, 2),
        "per_fold_estimate_MiB":  round(per_fold_mib, 2),
        "total_parallel_MiB":     round(total_mib, 2),
        "note": (
            "Dataset tensors are copy-on-write across loky forks on Linux, "
            "so dataset_tensors_MiB is counted once regardless of n_inner_folds."
        ),
    }