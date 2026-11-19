'''Compute performance metrics from an analysis dataframe's saved paths.'''

import json
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch

from NM_TinyRNN.code.measures.analysis import DATA_PATH, cache_matches_models, write_cache_provenance


def get_performance_df(analysis_df, n_jobs=-1, use_cache=True):
    '''Add saved performance values and trial-level metrics to path rows.'''

    if analysis_df.empty:
        raise ValueError('Analysis dataframe is empty; no saved models were found. Check folder_name and data/rnns.')
    if 'save_path' not in analysis_df:
        raise ValueError('Analysis dataframe must contain save_path; build it with analysis.get_analysis_df first.')
    cache_path = DATA_PATH / 'analyses' / 'performance' / 'performance_df.htsv'
    required_columns = {
        'eval_CE', 'best_val_CE', 'train_CE', 'val_CE', 'eval_CE_computed',
        'train_n_free', 'val_n_free', 'eval_n_free', 'block_train_val_CE',
        'block_val_CE', 'block_train_val_n_free', 'block_eval_CE', 'block_eval_n_free'
    }

    if use_cache and cache_path.exists():
        try:
            performance_df = pd.read_csv(cache_path, sep='\t')
            if required_columns.issubset(performance_df.columns) and cache_matches_models(performance_df, analysis_df):
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
        performance_df = add_block_scores(performance_df, n_jobs=n_jobs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(cache_path, sep='\t', index=False)
        write_cache_provenance(cache_path, performance_df)
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


def add_block_scores(performance_df, n_jobs=1):
    """Add actual saved-block prediction losses without changing source models."""
    from joblib import parallel_config
    from NM_TinyRNN.code.measures.block_performance import score_group
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(score_group)(group.copy())
            for _, group in performance_df.groupby(['save_path','subject_id','model_id']))
    scores = pd.concat(results, ignore_index=True)
    columns = [c for c in scores if c != 'info_path']
    base = performance_df.drop(columns=columns, errors='ignore')
    return base.merge(scores, on='info_path', validate='one_to_one')


def select_best_outer(performance_df, selection='block_reset'):
    """Select one inner winner per outer fold; block-reset train+val CE by default.

    ``selection='continuous'`` explicitly reproduces the historical continuous
    saved-trial score. Neither criterion uses outer-test loss or regularizers.
    """
    criteria = {'block_reset':'block_train_val_CE', 'continuous':'train_val_CE'}
    if selection not in criteria:
        raise ValueError(f'selection must be one of {list(criteria)}')
    score = criteria[selection]
    if score not in performance_df:
        raise ValueError(f'Missing {score}; rebuild with get_performance_df or add_block_scores')
    group_columns = ['model_id', 'hidden_size', 'subject_id', 'outer_loop_n']
    if not np.isfinite(performance_df[score]).all():
        raise ValueError(f'Nonfinite model-selection scores in {score}')
    selected_indices = performance_df.groupby(group_columns)[score].idxmin()
    selected = performance_df.loc[selected_indices].copy()
    selected['selection_method'] = selection
    selected['selection_score'] = selected[score]
    return selected.drop(columns=['train_val_CE'], errors='ignore')


def compute_outer_mean(performance_df):
    """Trial-weighted outer-fold NLLs using each protocol's own target counts.

    block_eval_CE is the primary reported held-out loss. eval_CE is its legacy
    reporting alias. eval_CE_computed retains continuous-rollout NLL, weighted
    separately by eval_n_free. Neither evaluation protocol reselects winners.
    """
    keys=['subject_id','model_id']
    required=set(keys+['outer_loop_n','eval_CE_computed','eval_n_free'])
    missing=required-set(performance_df)
    if missing:raise ValueError(f'Performance dataframe is missing columns: {sorted(missing)}')
    block_columns={'block_eval_CE','block_eval_n_free'}
    has_block=block_columns.issubset(performance_df)
    if not has_block and ('eval_CE' in performance_df or block_columns.intersection(performance_df)):
        raise ValueError('Missing block_eval_CE or block_eval_n_free; rebuild with get_performance_df or add_block_scores')
    scores={'eval_CE_computed':'eval_n_free'}
    if has_block:scores['block_eval_CE']='block_eval_n_free'
    drop={'outer_loop_n','inner_loop_idx','eval_CE',*scores,*scores.values()}
    metadata=[c for c in performance_df if c not in keys and c not in drop]
    result=performance_df.groupby(keys,sort=False)[metadata].first().reset_index()
    counts=performance_df.groupby(keys,sort=False).outer_loop_n.nunique().rename('n_outer_folds').reset_index()
    result=result.merge(counts,on=keys,validate='one_to_one')
    for score,weight in scores.items():
        d=performance_df[keys].copy()
        d['_score']=pd.to_numeric(performance_df[score],errors='coerce')
        d['_weight']=pd.to_numeric(performance_df[weight],errors='coerce')
        valid=d['_weight'].gt(0)&np.isfinite(d['_score'])&np.isfinite(d['_weight'])
        if not valid.all():raise ValueError(f'Nonfinite or zero-target fold in {score}; cannot silently drop an outer winner')
        d['_weighted']=d['_score']*d['_weight']
        aggregate=d.groupby(keys,sort=False).agg(_sum=('_weighted','sum'),_count=('_weight','sum'))
        aggregate[score]=aggregate['_sum']/aggregate['_count'];aggregate[weight]=aggregate['_count']
        result=result.merge(aggregate[[score,weight]].reset_index(),on=keys,validate='one_to_one')
    if has_block:
        result['eval_CE']=result['block_eval_CE']
        result['nll_evaluation_method']='block_reset'
    return result
