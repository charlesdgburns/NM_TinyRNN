"""Export the best inner fit for every subject/configuration/outer fold.

Run from any directory with --help for options. Selection uses the existing
performance.select_best_outer function (lowest train-plus-validation CE).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = [
    '2_unit_GRU_tanh_unipolar_forced',
    '2_unit_monoGRU_relu_unipolar_forced_ndb',
]
SUFFIXES = ('_info.json', '_model.pickle', '_trials_data.htsv')


def copy_winners(winners, output):
    """Preflight all source files, then copy into separate outer-fold folders."""
    output = Path(output)
    planned = []
    destinations = set()
    for row in winners.itertuples():
        folder = output / str(row.subject_id) / str(row.model_type) / f'outer_fold_{row.outer_loop_n}'
        for suffix in SUFFIXES:
            source = Path(row.info_path).parent / f'{row.model_id}{suffix}'
            destination = folder / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.resolve() == destination.resolve():
                raise ValueError(f'Export destination equals source: {source}')
            if destination in destinations:
                raise ValueError(f'Duplicate export destination: {destination}')
            destinations.add(destination)
            planned.append((source, destination))
    artifacts = []
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise IOError(f'Copy verification failed: {destination}')
        artifacts.append(dict(source=str(source), destination=str(destination), sha256=digest))
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder_name', help='Source run folder; default: nested_DA_S32_B8')
    parser.add_argument('--model_id', action='append', help='Repeat to override the two default configurations')
    parser.add_argument('--sequence-length', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--output_dir', type=Path, default=REPO / 'data/rnns/subject_examples_latest')
    args = parser.parse_args()
    # Resolve caller-relative output before switching to the package's data-path convention.
    output = args.output_dir.resolve()
    sys.path.insert(0, str(REPO.parent))
    os.chdir(REPO.parent)
    from NM_TinyRNN.code.models import experiments
    from NM_TinyRNN.code.measures import analysis, performance

    model_ids = args.model_id or DEFAULT_MODELS
    manifest = experiments.get_DA_info_df(
        folder_name=args.folder_name, sequence_length=args.sequence_length,
        batch_size=args.batch_size, model_types=['GRU', 'monoGRU'])
    manifest = manifest[manifest.model_id.isin(model_ids)].copy()
    missing = set(model_ids) - set(manifest.model_id)
    if missing:
        raise ValueError(f'Configurations not in the DA manifest: {sorted(missing)}')
    incomplete = manifest.loc[~manifest.completed.astype(bool)]
    if not incomplete.empty:
        raise ValueError(f'{len(incomplete)} requested subject/configuration/outer folds are incomplete; no export written.')
    paths = analysis.get_analysis_df(manifest, use_cache=False, n_jobs=args.n_jobs)
    scores = performance.get_performance_df(paths, use_cache=False, n_jobs=args.n_jobs)
    keys = ['subject_id', 'model_id', 'outer_loop_n']
    valid = scores['train_val_CE'].map(lambda value: value is not None and float('-inf') < float(value) < float('inf'))
    eligible = scores.loc[valid].copy()
    winners = performance.select_best_outer(eligible).copy()
    expected = set(manifest[keys].itertuples(index=False, name=None))
    observed = set(winners[keys].itertuples(index=False, name=None))
    if expected != observed:
        raise ValueError(f'{len(expected - observed)} outer folds lack a finite selection score; no export written.')
    winners = winners.sort_values(keys).reset_index(drop=True)
    artifacts = copy_winners(winners, output)
    for column, suffix in [('example_info_path', '_info.json'),
                           ('example_model_pickle_path', '_model.pickle'),
                           ('example_trials_data_path', '_trials_data.htsv')]:
        winners[column] = [str(output / str(r.subject_id) / str(r.model_type)
                               / f'outer_fold_{r.outer_loop_n}' / f'{r.model_id}{suffix}')
                           for r in winners.itertuples()]
    winners.to_csv(output / 'outer_fold_winners.htsv', sep='\t', index=False)
    metadata = dict(source_run=args.folder_name or experiments.training_folder_name(
        args.sequence_length, args.batch_size), models=len(winners), model_ids=model_ids,
        selection='performance.select_best_outer: minimum train_val_CE per outer fold',
        outer_selection='all', artifacts=artifacts)
    (output / 'export_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {len(winners)} models ({len(artifacts)} files) to {output}')


if __name__ == '__main__':
    main()
