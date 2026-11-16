#!/usr/bin/env python3
"""Export representative subject models and their saved trial-level data.

Example, from the repository root:
    .venv/bin/python code/models/save_examples.py \
        --folder_name nested_DA_last10_P100_S32_B8 \
        --output_dir data/rnns/subject_examples_DA_last10_P100_S32_B8

First select one inner winner per outer fold (default: block-reset train+validation
free-choice NLL), then choose the median outer-test NLL representative per subject/model.
As in the original representative mode, outer fold 10 is excluded by default.
Use --all_outer_folds to export every outer winner, including fold 10.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
import pandas as pd
from NM_TinyRNN.code.measures import analysis, performance
from NM_TinyRNN.code.models import experiments

SUFFIXES = ['_info.json', '_model.pickle', '_trials_data.htsv']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_most_recent(subdf):
    """Retain the original policy: exclude the highest outer-fold number."""
    if subdf.empty:
        return subdf
    for column in ['outer_loop_number', 'outer_loop_n', 'outer_fold', 'outer_loop_idx']:
        if column in subdf:
            return subdf.loc[subdf[column].ne(subdf[column].max())]
    if 'info_path' in subdf:
        folds = subdf.info_path.astype(str).str.extract(r'outer_fold_(\d+)', expand=False).astype(float)
        return subdf.loc[folds.ne(folds.max())]
    return subdf


def closest_to_median(subdf):
    """Find the row closest to median saved outer-test block NLL."""
    if subdf.empty:
        return None
    return subdf.loc[(subdf.eval_CE-subdf.eval_CE.median()).abs().idxmin()]


def idx_min(subdf):
    """Find the row with minimum saved outer-test block NLL."""
    if subdf.empty:
        return None
    return subdf.loc[subdf.eval_CE.idxmin()]


def select_outer_winners(frame, selection='block_reset'):
    """Use exactly the notebook selector, retaining the original score columns."""
    if selection not in ['block_reset', 'trainval', 'validation']:
        raise ValueError('Unknown inner selection rule')
    if frame.empty:
        raise ValueError('No fitted models available for selection')
    keys = ['subject_id', 'model_id', 'outer_loop_n']
    ordered = frame.sort_values(keys+['inner_loop_idx']).reset_index(drop=True)
    criterion = {'block_reset':'block_train_val_CE','trainval':'train_val_CE','validation':'best_val_CE'}[selection]
    if ordered[criterion].isna().any() or ordered.eval_CE.isna().any():
        raise ValueError('Selection or outer-test scores are missing')
    if selection in ['block_reset','trainval']:
        best_model_df = performance.select_best_outer(ordered,selection='block_reset' if selection=='block_reset' else 'continuous')
        return ordered.loc[best_model_df.index].reset_index(drop=True)
    return ordered.loc[ordered.groupby(keys, sort=False)[criterion].idxmin()].reset_index(drop=True)


def select_representatives(frame, metric='median', holdout_recent=True, selection='block_reset'):
    """Choose inner winners first, then one outer representative per subject/model."""
    if metric not in ['median', 'min']:
        raise ValueError('Unknown representative metric')
    winners = select_outer_winners(frame, selection)
    rows = []
    for _, group in winners.groupby(['subject_id', 'model_id'], sort=True):
        eligible = holdout_most_recent(group) if holdout_recent else group
        if eligible.empty:
            raise ValueError('No eligible outer folds after excluding the most recent fold')
        chosen = closest_to_median(eligible) if metric == 'median' else idx_min(eligible)
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def copy_representative_models(df, example_path, include_outer_fold=False):
    """Copy info/model/trials to <output>/<subject>/<architecture>/<model_id>_*.

    Validate the complete copy plan before writing; verify every copied file.
    Existing notebook callers can continue using this function directly.
    """
    example_path = Path(example_path)
    plan = []
    seen = set()
    for _, row in df.iterrows():
        for suffix in SUFFIXES:
            src = Path(row.info_path).parent/f'{row.model_id}{suffix}'
            folder = example_path/str(row.subject_id)/str(row.model_type)
            if include_outer_fold:
                folder = folder/f'outer_fold_{int(row.outer_loop_n)}'
            dst = folder/src.name
            if not src.is_file():
                raise FileNotFoundError(src)
            if dst in seen:
                raise ValueError(f'Duplicate example destination: {dst}')
            seen.add(dst)
            plan.append((src, dst))
    inventory = []
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        expected = sha256(src)
        if sha256(dst) != expected:
            raise OSError(f'Copied artifact hash mismatch: {dst}')
        inventory.append(dict(source=str(src.resolve()), destination=str(dst.resolve()),
                              bytes=dst.stat().st_size, sha256=expected))
    print(f'Copied and verified {len(inventory)} files for {len(df)} representative models.')
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected_models', type=Path, help='Verified outer-winner table from the analysis workflow; requires --all_outer_folds and matching selection metadata')
    parser.add_argument('--model_id', action='append', help='Filter model ID; repeat for multiple models. Default: all.')
    parser.add_argument('--folder_name', help='Source run under data/rnns. Omit for the legacy job preset.')
    parser.add_argument('--all_outer_folds', action='store_true',
                        help='Export all ten outer winners in separate outer_fold_N directories; no representative filtering.')
    parser.add_argument('--metric', choices=['median', 'min'], default='median')
    parser.add_argument('--selection', choices=['block_reset', 'trainval', 'validation'], default='block_reset')
    parser.add_argument('--holdout_recent', action=argparse.BooleanOptionalAction, default=True,
                        help='Exclude highest outer fold, as in the original exporter.')
    parser.add_argument('--output_dir', type=Path, default=REPO/'data/rnns/subject_examples')
    args = parser.parse_args(argv)
    # Resolve caller-supplied output before using legacy package-relative loaders.
    output = args.output_dir.resolve()
    winners_path = args.selected_models.resolve() if args.selected_models else None
    os.chdir(REPO.parent)
    if winners_path:
        if not args.all_outer_folds or not args.folder_name:
            parser.error('--selected_models requires --all_outer_folds and --folder_name')
        selected = pd.read_csv(winners_path, sep='\t')
        if args.selection != 'block_reset' or 'selection_method' not in selected or not selected.selection_method.eq('block_reset').all():
            raise ValueError('Winner table must record the requested block_reset selection')
        source_root=(REPO/'data/rnns'/args.folder_name).resolve()
        if not all(Path(p).resolve().is_relative_to(source_root) for p in selected.info_path):
            raise ValueError('Winner table source paths differ from --folder_name')
        if args.model_id:
            if set(args.model_id)-set(selected.model_id):raise ValueError('Requested models absent from winner table')
            selected=selected[selected.model_id.isin(args.model_id)].copy()
        if selected.duplicated(['subject_id','model_id','outer_loop_n']).any():
            raise ValueError('Duplicate outer winners')
        if not selected.groupby(['subject_id','model_id']).outer_loop_n.apply(lambda f:sorted(f)==list(range(1,11))).all():
            raise ValueError('Each subject/model must have all ten outer winners')
        selected=selected.sort_values(['subject_id','model_id','outer_loop_n']).reset_index(drop=True)
    else:
        experiment_df = (experiments.get_DA_info_df(folder_name=args.folder_name)
                         if args.folder_name else experiments.get_job_info_df())
        paths = analysis.get_analysis_df(experiment_df, n_jobs=1)
        scores = performance.get_performance_df(paths, n_jobs=1)
        if args.model_id:
            missing = set(args.model_id)-set(scores.model_id)
            if missing:
                raise ValueError(f'Unknown model IDs: {sorted(missing)}')
            scores = scores[scores.model_id.isin(args.model_id)]
        selected = (select_outer_winners(scores, args.selection) if args.all_outer_folds else
                    select_representatives(scores, args.metric, args.holdout_recent, args.selection))
    inventory = copy_representative_models(selected, output, include_outer_fold=args.all_outer_folds)
    # Preserve source paths and add explicit local example paths for easy loading.
    for field, suffix in [('example_info_path','_info.json'), ('example_model_pickle_path','_model.pickle'),
                          ('example_trials_data_path','_trials_data.htsv')]:
        selected[field] = [str(output/r.subject_id/r.model_type/
            (f'outer_fold_{int(r.outer_loop_n)}' if args.all_outer_folds else '')/f'{r.model_id}{suffix}')
            for r in selected.itertuples()]
    dataframe_name = 'outer_fold_winners.htsv' if args.all_outer_folds else 'representative_models.htsv'
    selected.to_csv(output/dataframe_name, sep='\t', index=False)
    metadata = dict(source_run=args.folder_name, subjects=sorted(selected.subject_id.unique()),
        selection={'block_reset':'performance.select_best_outer: minimum block-reset train+validation free-choice NLL per outer fold',
                   'trainval':'performance.select_best_outer: minimum continuous train+validation NLL per outer fold',
                   'validation':'minimum saved validation NLL per outer fold'}[args.selection],
        n_models=len(selected),
        n_representative_models=len(selected), n_model_configurations=int(selected.model_id.nunique()),
        inner_selection=args.selection, outer_selection='all' if args.all_outer_folds else args.metric,
        exclude_highest_outer_fold=False if args.all_outer_folds else args.holdout_recent,
        outer_score='eval_CE: saved block-reset outer-test NLL',
        trial_data='Unmodified saved continuous-subject trial export from this training run',
        note=('All outer winners; no selection using outer-test performance.' if args.all_outer_folds else
              'Representative examples use outer-test NLL for illustration; not a new unbiased performance estimate.'),
        source_code_sha256=sha256(__file__),
        selected_models_source=str(winners_path) if winners_path else None,
        selected_models_sha256=sha256(winners_path) if winners_path else None, artifacts=inventory)
    (output/'export_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved example dataframe and provenance to {output}')


if __name__ == '__main__':
    main()
