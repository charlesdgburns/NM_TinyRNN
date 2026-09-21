'''Build dataframes containing saved model paths for downstream analyses.'''

from joblib import Parallel, delayed
import json
import pickle
import pandas as pd
from pathlib import Path
import torch


DATA_PATH = Path('./NM_TinyRNN/data/')
RNNS_PATH = DATA_PATH / 'rnns'


def get_analysis_df(info_df, mode='all', n_jobs=-1, use_cache=True):
    '''Build a dataframe with model metadata and paths to saved artifacts.

    Performance values are deliberately not computed here. Use
    ``get_performance_df`` from ``code.measures.performance`` to add metrics.
    '''
    if mode != 'all':
        raise ValueError("Path discovery only supports mode='all'; select best models from a performance dataframe.")

    cache_path = DATA_PATH / 'analysis' / 'analysis_df_final.htsv'
    if use_cache and cache_path.exists():
        try:
            expanded_df = pd.read_csv(cache_path, sep='\t')
            path_columns = _path_columns(expanded_df)
            if path_columns:
                print(f"Loaded analysis paths from cache: {cache_path}")
                return expanded_df[path_columns]
        except Exception as error:
            print(f"Warning: Could not load analysis paths from cache: {error}. Recomputing...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_model_paths)(row) for row in info_df.itertuples()
    )
    expanded_df = pd.DataFrame([row for result in results for row in result])

    if not expanded_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        expanded_df.to_csv(cache_path, sep='\t', index=False)
        print(f"Saved analysis paths to cache: {cache_path}")

    return expanded_df


def get_model_paths(each_model):
    '''Return one path row for every saved inner-fold model.'''
    if not each_model.completed:
        print(f"Model {each_model.model_id} has not completed training.")
        return []

    model_save_path = Path(each_model.save_path)
    outer_folder = model_save_path / f'outer_fold_{each_model.outer_loop_n}'
    base_row = {column: getattr(each_model, column) for column in each_model._fields}
    rows = []

    if not outer_folder.exists():
        return rows

    model_type2 = _compute_model_type2(
        each_model.model_type,
        each_model.nonlinearity,
        each_model.decoder_bias,
    )

    for inner_idx, inner_folder in enumerate(sorted(outer_folder.iterdir())):
        if not inner_folder.is_dir():
            continue

        info_path = inner_folder / f'{each_model.model_id}_info.json'
        if not info_path.exists():
            continue

        rows.append({
            **base_row,
            'model_type2': model_type2,
            'inner_loop_idx': inner_idx,
            'info_path': str(info_path),
            'model_state_path': str(inner_folder / f'{each_model.model_id}_model_state.pth'),
            'model_pickle_path': str(inner_folder / f'{each_model.model_id}_model.pickle'),
            'training_losses_path': str(inner_folder / f'{each_model.model_id}_training_losses.htsv'),
            'trials_data_path': str(inner_folder / f'{each_model.model_id}_trials_data.htsv'),
        })

    return rows


def _compute_model_type2(model_type, nonlinearity, decoder_bias):
    '''Compute the display model type used by existing analyses.'''
    if nonlinearity == 'relu':
        model_type += '+BC'
    if decoder_bias is False:
        model_type += '-DB'
    return model_type


def _path_columns(dataframe):
    required_columns = {'info_path', 'trials_data_path'}
    if not required_columns.issubset(dataframe.columns):
        return []
    return [
        column for column in dataframe.columns
        if not column.endswith('_CE') and not column.endswith('_n_trials')
    ]


def select_best_outer(analysis_df):
    '''Compatibility wrapper for selecting models from performance data.'''
    from NM_TinyRNN.code.measures.performance import select_best_outer as _select_best_outer

    return _select_best_outer(analysis_df)

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
