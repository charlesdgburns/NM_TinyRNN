"""
nested_cv_io.py  --  Saving utilities for nested cross-validation results

Directory layout
----------------
save_path/
  outer_fold_N/
    inner_fold_M/
      info.json               hyperparams, trainer settings, split sizes, val loss
      model_state.pth         winning model weights
      training_losses.htsv   epoch-by-epoch losses (from TrainerGPU if available)
      trials_data.htsv        trial-by-trial activations / predictions

Usage
-----
Add one call at the end of _run_inner_fold():

    save_inner_fold_results(
        result            = result,
        base_model        = base_model,
        dataset           = dataset,          # full dataset, all sessions
        splits            = splits,
        trainer           = trainer,
        save_path         = save_path,
        outer_loop_number = outer_loop_number,
        inner_fold_idx    = fold_info["fold_idx"],
    )
    return result
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fold_dir(save_path: Path, outer_loop_number: int, inner_fold_idx: int) -> Path:
    d = save_path / f"outer_fold_{outer_loop_number}" / f"inner_fold_{inner_fold_idx}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_safe(v: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain Python types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {kk: _json_safe(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Eval loss on held-out (outer eval) data
# ---------------------------------------------------------------------------

def compute_eval_loss(model, dataset, eval_block_indices: list) -> float:
    """
    Compute the mean per-trial cross-entropy loss on the outer eval blocks,
    masking out forced-choice trials exactly as the training loop does.

    The masking logic mirrors _run_epoch in Trainer:
      - A trial is a free-choice trial when forced_choice == 0.
      - Because we predict the *next* trial's choice, the mask is shifted
        by one step: mask[t] = free_choice[t+1].
      - Loss is averaged over the surviving free-choice trials only.

    Parameters
    ----------
    model             : trained model in eval() mode.
    dataset           : full AB_Dataset (all sessions).
    eval_block_indices: list of sequence_block_idx values for the outer eval set.

    Returns
    -------
    float -- mean cross-entropy over free-choice trials in the eval set,
             or np.nan if no eval trials exist.
    """
    if not eval_block_indices:
        return float("nan")

    model.eval()

    # Select eval trials from subject_df using block membership
    block_col  = dataset.subject_df["sequence_block_idx"].values
    eval_mask  = np.isin(block_col, list(eval_block_indices))
    eval_df    = dataset.subject_df[eval_mask].reset_index(drop=True)

    if eval_df.empty:
        return float("nan")

    # Build (1, T_eval, 3) input tensor for the eval trials only
    raw = torch.tensor(
        eval_df[["forced_choice", "outcome", "choice"]].values,
        dtype=torch.float32,
    ).unsqueeze(0)   # (1, T, 3)

    with torch.no_grad():
        predictions, _ = model(raw)   # (1, T, 2) logits

    # Replicate the free-choice shift from _run_epoch:
    #   free_choice[t] is True when the animal made a free choice at trial t.
    #   We predict the next trial, so mask[t] = free_choice[t+1].
    forced_choice = (raw[0, :, 0] == 0)          # (T,)  True = free choice
    mask          = forced_choice.clone()
    mask[:-1]     = forced_choice[1:].clone()     # shift by one
    mask          = mask.cpu()

    if mask.sum() == 0:
        return float("nan")

    # Cross-entropy on the masked free-choice predictions
    # targets: next trial's choice, one-hot encoded as class index
    choices      = torch.tensor(eval_df["choice"].values, dtype=torch.long)
    preds_masked = predictions[0][mask]     # (N_free, 2)
    tgts_masked  = choices[mask]            # (N_free,)

    loss = F.cross_entropy(preds_masked, tgts_masked).item()
    return loss


# ---------------------------------------------------------------------------
# Main save function
# ---------------------------------------------------------------------------

def save_inner_fold_results(
    result:             dict,
    base_model,
    dataset,
    splits:             dict,
    trainer,
    save_path:          Path,
    outer_loop_number:  int,
    inner_fold_idx:     int,
    outer_eval_blocks:  list = None,
) -> Path:
    """
    Persist all artefacts for one winning inner-fold model.

    Parameters
    ----------
    result            : dict returned by _run_inner_fold.
                        Must contain "state_dict", "config", "val_loss".
                        Optionally "training_losses" (dict-of-lists or DataFrame).
    base_model        : model template; weights replaced by result["state_dict"]
                        before the trial-by-trial forward pass.
    dataset           : the FULL AB_Dataset (all sessions, not the patched subset
                        used for training).  Outer eval trials will appear in the
                        trial output labelled "unused".
    splits            : {"train", "val", "eval"} block-index dict for this fold.
    trainer           : TrainerGPU instance; used to read hyperparameter ranges.
    save_path         : root save directory.
    outer_loop_number : outer fold index.
    inner_fold_idx    : inner fold index.
    outer_eval_blocks : sequence_block_idx values for the outer held-out set.
                        When provided, the eval prediction loss is computed and
                        stored in info.json.  Pass run_outer_fold()'s
                        result["outer_eval_blocks"].

    Returns
    -------
    Path to the fold directory that was written.
    """
    model_id = base_model.get_model_id()
    save_path = Path(save_path)
    fold_dir  = _fold_dir(save_path, outer_loop_number, inner_fold_idx)

    # 1. model_state.pth -- saved first so a later crash never loses the weights
    torch.save(result["state_dict"], fold_dir / f"{model_id}_model_state.pth")

    # 2. Build eval model once -- reused for both eval loss and trial extraction
    winning_config = result["config"]   # tuple: (sparsity, energy, hebbian, seed)
    config_keys    = ["sparsity_lambda", "energy_lambda", "hebbian_lambda", "weight_seed"]
    config_dict    = dict(zip(config_keys, winning_config))

    eval_model = deepcopy(base_model)
    eval_model.load_state_dict(result["state_dict"])
    eval_model.sparsity_lambda = config_dict["sparsity_lambda"]
    eval_model.energy_lambda   = config_dict["energy_lambda"]
    eval_model.hebbian_lambda  = config_dict["hebbian_lambda"]
    eval_model.weight_seed     = config_dict["weight_seed"]
    eval_model.eval()

    # 3. Eval loss on outer held-out data (free-choice trials only)
    eval_pred_loss = (
        compute_eval_loss(eval_model, dataset, outer_eval_blocks)
        if outer_eval_blocks is not None
        else float("nan")
    )

    # 4. info.json
    hparam_ranges: dict = {}
    for attr in (
        "sparsity_lambdas", "energy_lambdas", "hebbian_lambdas",
        "weight_seeds", "learning_rate", "batch_size",
        "max_epochs", "early_stop",
    ):
        if hasattr(trainer, attr):
            hparam_ranges[attr] = _json_safe(getattr(trainer, attr))

    info = {
        "outer_loop_number":    outer_loop_number,
        "inner_fold_idx":       inner_fold_idx,
        "winning_config":       _json_safe(config_dict),
        "val_loss":             _json_safe(result["val_loss"]),
        "eval_pred_loss":       _json_safe(eval_pred_loss),
        "split_sizes":          {k: len(v) for k, v in splits.items()},
        "n_outer_eval_blocks":  len(outer_eval_blocks) if outer_eval_blocks else 0,
        "hparam_ranges":        hparam_ranges,
        **(
            {"model_options": _json_safe(eval_model.get_options_dict())}
            if hasattr(eval_model, "get_options_dict") else {}
        ),
    }

    with open(fold_dir / f"{model_id}_info.json", "w") as fh:
        json.dump(info, fh, indent=2)

    # 5. training_losses.htsv (optional -- only written if TrainerGPU surfaces them)
    if "training_losses" in result and result["training_losses"] is not None:
        tl = result["training_losses"]
        if isinstance(tl, pd.DataFrame):
            losses_df = tl
        elif isinstance(tl, dict):
            losses_df = pd.DataFrame(tl)
        else:
            losses_df = None
        if losses_df is not None:
            losses_df.to_csv(fold_dir / f"{model_id}_training_losses.htsv", sep="\t", index=False)

    # 6. trials_data.htsv
    trial_df = get_model_trial_by_trial_df(eval_model, dataset, splits)
    trial_df.to_csv(fold_dir / f"{model_id}_trials_data.htsv", sep="\t", index=False)

    print(
        f"  Saved outer_fold_{outer_loop_number}/inner_fold_{inner_fold_idx}"
        f" -> {fold_dir}"
        f"  (val_loss={result['val_loss']:.4f}"
        + (f", eval_loss={eval_pred_loss:.4f}" if not np.isnan(eval_pred_loss) else "")
        + ")"
    )
    return fold_dir

# ---------------------------------------------------------------------------
# Standalone trial-by-trial extractor
# ---------------------------------------------------------------------------


def get_model_trial_by_trial_df(model, dataset, splits: dict) -> pd.DataFrame:
    """
    Run a trained model over the full dataset and return a trial-by-trial
    DataFrame.  Standalone -- does not depend on any Trainer instance.
 
    Parameters
    ----------
    model   : trained model with weights already loaded, in eval() mode.
              Expected attributes:
                H (int)         number of hidden units
                rnn_type (str)  "vanilla" skips gate extraction
              Optional:
                input_forced_choice (bool)
                input_encoding (str)  "bipolar" -> {0,1} mapped to {-1,+1}
    dataset : AB_Dataset covering ALL sessions (not a patched inner-fold subset).
              Must expose subject_df with a 'sequence_block_idx' column.
    splits  : {"train", "val", "eval"} block-index dict.  Every trial whose
              block is not in any split is labelled "unused".
 
    Returns
    -------
    pd.DataFrame with one row per trial containing:
      - all columns from dataset.subject_df
      - hidden_1 ... hidden_H
      - gate_<n>_1 ... _H   (GRU / LSTM only)
      - logit_value, logit_past, logit_change
      - prob_A, prob_B
      - trial_type              e.g. "A1,R=0"
      - split                   "train" / "val" / "eval" / "unused"
    """
    if dataset.subject_df.empty:
        raise ValueError(
            "dataset.subject_df is empty -- pass the full dataset "
            "(all sessions), not the patched inner-fold subset."
        )
 
    model.eval()
    data: dict = {}
 
    # Build a single (1, T, 3) input tensor from the raw subject_df so that
    # every trial appears in temporal order regardless of sequence chunking.
    raw = torch.tensor(
        dataset.subject_df[["forced_choice", "outcome", "choice"]].values,
        dtype=torch.float32,
    ).unsqueeze(0)   # (1, T, 3)
 
    with torch.no_grad():
        predictions, hidden_states = model(raw)
        # hidden_states: (1, T, H)
        for u in range(model.H):
            data[f"hidden_{u+1}"] = hidden_states[0, :, u].cpu().numpy()
 
        rnn_type = getattr(model, "rnn_type", "vanilla")
        if rnn_type != "vanilla":
            use_fc = getattr(model, "input_forced_choice", True)
            inp    = raw if use_fc else raw[:, :, 1:]
            if getattr(model, "input_encoding", None) == "bipolar":
                inp = inp * 2 - 1
            _, gate_activations = model.rnn(inp, return_gate_activations=True)
            for gate_name, acts in gate_activations.items():
                for u in range(acts.shape[-1]):
                    data[f"gate_{gate_name}_{u+1}"] = acts[0, :, u].cpu().numpy()
 
    log_probs = predictions.log_softmax(dim=2)   # (1, T, 2)
    logits    = (log_probs[0, :, 0] - log_probs[0, :, 1]).cpu().numpy()
 
    data["logit_value"]  = logits
    data["logit_past"]   = np.concatenate([[np.nan], logits[:-1]])
    data["logit_change"] = np.concatenate([[np.nan], np.diff(logits)])
    data["prob_A"]       = log_probs[0, :, 0].exp().cpu().numpy()
    data["prob_B"]       = log_probs[0, :, 1].exp().cpu().numpy()
 
    for col in dataset.subject_df.columns:
        data[col] = dataset.subject_df[col].values
 
    labels_key = ["A1,R=0", "A1,R=1", "A2,R=0", "A2,R=1"]
    data["trial_type"] = [
        labels_key[int(c * 2 + o)]
        for c, o in zip(data["choice"], data["outcome"])
    ]
 
    # Use our own block-index-aware labeller rather than AB_Dataset.get_split_labels,
    # which assumes block indices are contiguous from zero.
    data["split"] = get_split_labels(dataset.subject_df, splits)
 
    return pd.DataFrame(data)


def get_split_labels(subject_df: pd.DataFrame, splits: dict) -> np.ndarray:
    """
    Assign a split label to every row in subject_df.
 
    Unlike AB_Dataset.get_split_labels(), this works directly from the
    'sequence_block_idx' column rather than assuming block indices are
    contiguous from zero.  This makes it safe to use with the global block
    indices produced by the nested CV splitting logic.
 
    Parameters
    ----------
    subject_df : the dataset's trial-level DataFrame (one row per trial).
                 Must contain a 'sequence_block_idx' column.
    splits     : dict mapping split name -> list of sequence_block_idx values.
                 e.g. {"train": [0,1,4,5], "val": [2], "eval": []}
 
    Returns
    -------
    np.ndarray of dtype object, length len(subject_df), with values
    "train" / "val" / "eval" / "unused".
    """
    labels = np.full(len(subject_df), "unused", dtype=object)
    block_col = subject_df["sequence_block_idx"].values
    for split_name, block_indices in splits.items():
        if not block_indices:
            continue
        block_set = set(block_indices)
        mask = np.isin(block_col, list(block_set))
        labels[mask] = split_name
    return labels
 
 
# 