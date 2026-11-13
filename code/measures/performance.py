'''Compute performance metrics from an analysis dataframe's saved paths.'''

import json
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch

from NM_TinyRNN.code.measures.analysis import DATA_PATH


def get_performance_df(analysis_df, n_jobs=-1, use_cache=True):
    '''Add saved performance values and trial-level metrics to path rows.'''

    folder_name = Path(analysis_df.save_path.iloc[0]).parts[3]
    cache_path = DATA_PATH / 'analysis' / f'performance_df_{folder_name}_final.htsv'
    required_columns = {
        'eval_CE', 'best_val_CE', 'train_CE', 'val_CE', 'eval_CE_computed',
        'train_n_free', 'val_n_free', 'eval_n_free'
    }

    if use_cache and cache_path.exists():
        try:
            performance_df = pd.read_csv(cache_path, sep='\t')
            if required_columns.issubset(performance_df.columns):
                print(f"Loaded performance dataframe from cache: {cache_path}")
                return performance_df
        except Exception as error:
            print(f"Warning: Could not load performance cache: {error}. Recomputing...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_model_performance)(row) for row in analysis_df.itertuples()
    )
    performance_df = analysis_df.copy()
    if results:
        metrics_df = pd.DataFrame(results, index=performance_df.index)
        performance_df = pd.concat([performance_df, metrics_df], axis=1)

    if not performance_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(cache_path, sep='\t', index=False)
        print(f"Saved performance dataframe to cache: {cache_path}")

    return performance_df


def get_model_performance(each_model):
    '''Compute performance metrics for one path row.'''
    info_dict = load_data(each_model.info_path)
    metrics = {
        'eval_CE': info_dict.get('eval_pred_loss'),
        'best_val_CE': info_dict.get('val_loss'),
        'weight_seed': None,
        'sparsity_lambda': None,
        'energy_lambda': None,
        'train_CE': float('nan'),
        'val_CE': float('nan'),
        'train_val_CE': float('nan'),
        'eval_CE_computed': float('nan'),
        'train_n_free': float('nan'),
        'val_n_free': float('nan'),
        'eval_n_free': float('nan'),
    }

    winning_config = info_dict.get('winning_config', {})
    metrics['weight_seed'] = winning_config.get('weight_seed')
    metrics['sparsity_lambda'] = winning_config.get('sparsity_lambda')
    metrics['energy_lambda'] = winning_config.get('energy_lambda')

    trials_path = Path(each_model.trials_data_path)
    if trials_path.exists():
        trials_df = load_data(trials_path)
        metrics['train_CE'], metrics['train_n_free'] = cross_entropy_from_trials_df(trials_df, 'train')
        metrics['val_CE'], metrics['val_n_free'] = cross_entropy_from_trials_df(trials_df, 'val')
        metrics['train_val_CE'], _ = cross_entropy_from_trials_df(trials_df, 'train_val')
        metrics['eval_CE_computed'], metrics['eval_n_free'] = cross_entropy_from_trials_df(trials_df, 'eval')

    return metrics


def load_data(filepath):
    """Load JSON, HTSV, pickle, or PyTorch checkpoint data."""
    filepath = str(filepath)
    if filepath.endswith('.json'):
        with open(filepath, 'r') as file:
            return json.load(file)
    if filepath.endswith('.htsv'):
        return pd.read_csv(filepath, sep='\t')
    if filepath.endswith('.pickle'):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    if filepath.endswith('.pth'):
        return torch.load(filepath, weights_only=True)
    raise ValueError(f'Unsupported file type: {filepath}')


def cross_entropy_from_trials_df(trials_df: pd.DataFrame, split: str) -> tuple[float, int]:
    '''Compute NLL from softmax probabilities and count free-choice targets.'''
    dataframe = trials_df.reset_index(drop=True)
    if len(dataframe) < 2:
        return float('nan'), 0

    targets = dataframe['choice'].values[1:].astype(int)
    probabilities = dataframe[['prob_A', 'prob_B']].values[:-1].astype(float)
    forced_mask = dataframe['forced_choice'].values[1:].astype(int)
    split_labels = dataframe['split'].values[1:]
    if split == 'train_val':
        split_mask = (split_labels == 'train') | (split_labels == 'val')
    else:
        split_mask = split_labels == split

    free_choice_mask = split_mask & (forced_mask == 0)
    n_free = int(free_choice_mask.sum())
    if not free_choice_mask.any():
        return float('nan'), n_free

    target_probabilities = probabilities[free_choice_mask, targets[free_choice_mask]]
    loss = float(-np.log(np.clip(target_probabilities, np.finfo(float).tiny, 1.0)).mean())
    return loss, n_free


def select_best_outer(performance_df):
    '''Select the lowest train-plus-validation CE model per outer fold.'''
    group_columns = ['model_id', 'hidden_size', 'subject_id', 'outer_loop_n']
    selected_indices = performance_df.groupby(group_columns)['train_val_CE'].idxmin()
    return performance_df.loc[selected_indices].drop(columns=['train_val_CE'])


def compute_outer_mean(performance_df):
    '''Aggregate outer-fold performance while preserving model metadata.

    ``eval_CE_computed`` is averaged across outer folds using
    ``eval_n_free`` as the weight for each fold. All non-fold columns are
    retained using their first value within each subject/model group.
    '''
    required_columns = {
        'subject_id', 'model_id', 'eval_CE_computed', 'eval_n_free'
    }
    missing_columns = required_columns.difference(performance_df.columns)
    if missing_columns:
        raise ValueError(
            f"Performance dataframe is missing columns: {sorted(missing_columns)}"
        )

    dataframe = performance_df.copy()
    dataframe['_eval_weight'] = pd.to_numeric(
        dataframe['eval_n_free'], errors='coerce'
    )
    dataframe['_eval_score'] = pd.to_numeric(
        dataframe['eval_CE_computed'], errors='coerce'
    )
    valid_rows = (
        dataframe['_eval_weight'].gt(0)
        & dataframe['_eval_score'].notna()
    )

    group_columns = ['subject_id', 'model_id']
    columns_to_drop = {
        'outer_loop_n', 'inner_loop_idx', 'eval_CE', 'eval_CE_computed',
        'eval_n_free',
        '_eval_weight', '_eval_score'
    }
    metadata_columns = [
        column for column in performance_df.columns
        if column not in group_columns and column not in columns_to_drop
    ]
    aggregated = dataframe.groupby(group_columns, sort=False)[metadata_columns].first().reset_index()
    fold_counts = dataframe.groupby(group_columns, sort=False)['outer_loop_n'].nunique()
    trial_counts = dataframe.groupby(group_columns, sort=False)['_eval_weight'].sum()
    aggregated = aggregated.merge(
        fold_counts.rename('n_outer_folds'),
        on=group_columns,
    ).merge(
        trial_counts.rename('eval_n_free'),
        on=group_columns,
    )
    weighted_values = dataframe.loc[valid_rows].assign(
        weighted_eval_CE=lambda rows: rows['_eval_score'] * rows['_eval_weight']
    ).groupby(group_columns, sort=False).agg(
        weighted_eval_CE=('weighted_eval_CE', 'sum'),
        valid_eval_n_free=('_eval_weight', 'sum'),
    )
    aggregated = aggregated.merge(
        weighted_values,
        on=group_columns,
        how='left',
    )
    aggregated['eval_CE_computed'] = (
        aggregated['weighted_eval_CE'] / aggregated['valid_eval_n_free']
    )
    return aggregated.drop(columns=['weighted_eval_CE', 'valid_eval_n_free'])