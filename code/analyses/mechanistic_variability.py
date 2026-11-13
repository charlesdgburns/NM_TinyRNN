"""Mechanistic variability: compute selected-model similarities, statistics, then plots.

From the repository root:
    .venv/bin/python code/analyses/mechanistic_variability.py
    .venv/bin/python code/analyses/mechanistic_variability.py --stage plots

This post-training study does not launch fits or AWS jobs.
"""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from NM_TinyRNN.code.models import experiments
from NM_TinyRNN.code.measures import analysis, performance
from NM_TinyRNN.code.measures import mechanistic_variability as measures

DEFAULT_RUN = 'nested_DA_last10_P100_S32_B8'
SELECTION = 'performance.select_best_outer: minimum combined train_val_CE'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def paths_for(folder):
    if Path(folder).name != folder or folder in ['', '.', '..']:
        raise ValueError('--folder must name one run under data/rnns')
    tables = REPO/'data/analysis/mechanistic_variability'
    figures = REPO/'results/figures'/f'appendix_mechanistic_variability_{folder}'
    cache = tables/'pairwise_similarities.htsv'
    return tables, figures, cache


def publish_notebook_cache(pairs_path, cache, tables, computation):
    """Expose the study pair table directly to notebooks, without a duplicate cache."""
    if cache.exists() and measures.sha256(cache) != measures.sha256(pairs_path):
        backup = REPO/'data/analysis/.local/mechanistic_variability'/f'previous_notebook_cache_{measures.sha256(cache)[:12]}.htsv'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(cache, backup)
    if not cache.exists() or measures.sha256(cache) != measures.sha256(pairs_path):
        temporary = cache.with_suffix('.htsv.tmp')
        shutil.copy2(pairs_path, temporary)
        temporary.replace(cache)
    write_json(tables/'notebook_cache.json', dict(computation,
        study='code/analyses/mechanistic_variability.py', tables=str(tables)))


def compute(folder, tables, cache, manifest, jobs=3, recompute=False):
    """Select exact notebook winners and compare responses on subject B own trials."""
    config_path = REPO/'data/rnns'/folder/'_aws/run-config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Frozen run settings are required: {config_path}')
    config = json.loads(config_path.read_text())
    experiment_df = experiments.get_DA_info_df(folder_name=folder,
        sequence_length=config['sequence_length'], batch_size=config['trainer']['batch_size'])
    analysis_df = analysis.get_analysis_df(experiment_df, n_jobs=1)
    performance_df = performance.get_performance_df(analysis_df, n_jobs=jobs)
    keys = ['subject_id', 'model_id', 'outer_loop_n', 'inner_loop_idx']
    if set(analysis_df[keys].itertuples(index=False, name=None)) != set(performance_df[keys].itertuples(index=False, name=None)):
        raise ValueError('Performance cache does not match the discovered inner models')
    selected = performance.select_best_outer(performance_df).sort_values(
        ['model_id', 'subject_id', 'outer_loop_n']).reset_index(drop=True)
    inputs, references = measures.build_subject_trial_inputs(selected)
    # Model bytes, selected endpoints, probe bytes, settings and metric code identify
    # reuse; run-name-only caching was the source of the original selection mismatch.
    model_hashes = {str(path): measures.sha256(path) for path in selected.model_pickle_path}
    fingerprint_data = dict(selection=SELECTION, selected=selected[keys].to_dict('records'),
        model_sha256=model_hashes, reference_protocol='subject_B_continuous_trials',
        inputs_sha256={subject: hashlib.sha256(value.numpy().tobytes()).hexdigest() for subject,value in inputs.items()},
        config_sha256=measures.sha256(config_path), measures_sha256=measures.sha256(measures.__file__))
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    pair_path = tables/'pairwise_similarities.htsv'
    previous = manifest.get('computation', {})
    reuse = (not recompute and previous.get('fingerprint') == fingerprint and pair_path.is_file()
             and previous.get('cache_sha256') == measures.sha256(pair_path))
    for frame, name in [(experiment_df, 'experiment_df'), (selected, 'selected_models'),
                        (references, 'reference_subjects')]:
        frame.to_csv(tables/f'{name}.htsv', sep='\t', index=False)
    np.savez_compressed(tables/'reference_inputs.npz', **{subject:value.numpy() for subject,value in inputs.items()})
    if reuse:
        print('Compute: reusing similarities with matching models, probes, settings and code.', flush=True)
        pairs = pd.read_csv(pair_path, sep='\t')
        validation = previous['validation']
    else:
        print(f'Compute: {len(selected)} selected models across {selected.model_id.nunique()} configurations.', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            validation = Parallel(n_jobs=jobs)(delayed(measures.compute_reference_group)(group.copy(), inputs, str(tables))
                for _, group in selected.groupby('model_id'))
        pairs = pd.concat([pd.read_csv(tables/'groups'/f'{key}.htsv', sep='\t')
                           for key in sorted(selected.model_id.unique())], ignore_index=True)
        measures.validate_similarity_cache_models(pairs, selected)
        temporary = pair_path.with_suffix('.htsv.tmp')
        pairs.to_csv(temporary, sep='\t', index=False)
        measures.validate_similarity_cache_models(pd.read_csv(temporary, sep='\t'), selected)
        temporary.replace(pair_path)
        manifest.pop('statistics', None)
        manifest.pop('plots', None)
    measures.validate_similarity_cache_models(pairs, selected)
    computation = dict(fingerprint=fingerprint, selection=SELECTION, n_models=len(selected),
        n_variants=int(selected.model_id.nunique()), n_pairs=len(pairs), input_shape={subject:list(value.shape) for subject,value in inputs.items()},
        input_protocol='Subject B complete saved trial sequence; within subject uses own data; continuous state as in original notebook',
        pair_direction='Original sorted subject/outer ordering; B supplies trials, without averaging the reverse direction',
        inputs_sha256=fingerprint_data['inputs_sha256'], cache_sha256=measures.sha256(pair_path),
        selected_sha256=measures.sha256(tables/'selected_models.htsv'), validation=validation,
        metrics='Canonical parameter alignment/cosine; hidden per-unit mean Pearson; flattened gate/readout Pearson',
        source_code_sha256=measures.sha256(measures.__file__))
    manifest['computation'] = computation
    write_json(tables/'computation_inputs.json', fingerprint_data)
    publish_notebook_cache(pair_path, cache, tables, computation)
    return selected, pairs


def load_computation(tables, manifest):
    """Later stages consume the exact frozen output of the computation stage."""
    meta = manifest.get('computation')
    if not meta:
        raise FileNotFoundError('Run the compute stage first (or omit --stage for the full workflow).')
    for name, key in [('pairwise_similarities.htsv', 'cache_sha256'), ('selected_models.htsv', 'selected_sha256')]:
        path = tables/name
        if not path.is_file() or measures.sha256(path) != meta[key]:
            raise ValueError(f'Computation output changed or is missing: {path}; rerun --stage compute')
    selected = pd.read_csv(tables/'selected_models.htsv', sep='\t')
    pairs = pd.read_csv(tables/'pairwise_similarities.htsv', sep='\t')
    measures.validate_similarity_cache_models(pairs, selected)
    return selected, pairs


def statistics(selected, pairs, tables, manifest):
    subject_pairs, summary = measures.summarize_reference_pairs(pairs)
    subject_pairs.to_csv(tables/'subject_pair_similarity.htsv', sep='\t', index=False)
    summary.to_csv(tables/'similarity_summary.htsv', sep='\t', index=False)
    two_unit_ids = selected.loc[selected.hidden_size.eq(2), 'model_id'].unique()
    contrasts = measures.compare_similarity_to_reference(subject_pairs[subject_pairs.model_id.isin(two_unit_ids)])
    contrasts.to_csv(tables/'similarity_vs_standard_gru.htsv', sep='\t', index=False)
    manifest['statistics'] = dict(reference=measures.STANDARD_GRU_REFERENCE,
        n_tests=int(contrasts.p_holm.notna().sum()), n_significant=int(contrasts.p_holm.lt(.05).sum()),
        inference='Paired subject-level descriptive dz; leave-subject-out jackknife SE, approximate t(N-1); global Holm correction',
        caveat='Exploratory inference with 11 subjects; across subject summaries share dyads. Pointwise CIs are unadjusted.',
        pair_cache_sha256=manifest['computation']['cache_sha256'],
        output_sha256={name: measures.sha256(tables/name) for name in
            ['subject_pair_similarity.htsv', 'similarity_summary.htsv', 'similarity_vs_standard_gru.htsv']})
    print(f'Statistics: {len(contrasts)} cells, {manifest["statistics"]["n_tests"]} tests, '
          f'{manifest["statistics"]["n_significant"]} Holm-corrected differences.', flush=True)
    return subject_pairs, contrasts


def plots(selected, pairs, tables, figures, manifest):
    from NM_TinyRNN.code.figures import mechanistic_variability as plotting
    meta = manifest.get('statistics')
    if not meta or meta['pair_cache_sha256'] != manifest['computation']['cache_sha256']:
        raise ValueError('Run the statistics stage before plots (or run the full workflow).')
    for name, expected in meta['output_sha256'].items():
        if measures.sha256(tables/name) != expected:
            raise ValueError(f'Statistics output changed: {name}; rerun --stage statistics')
    subject_pairs = pd.read_csv(tables/'subject_pair_similarity.htsv', sep='\t')
    contrasts = pd.read_csv(tables/'similarity_vs_standard_gru.htsv', sep='\t')
    two_unit = selected[selected.hidden_size.eq(2)]
    ids = two_unit.model_id.unique()
    plotting.similarity_figures(pairs[pairs.model_id.isin(ids)], two_unit,
        subject_pairs[subject_pairs.model_id.isin(ids)], figures,
        selection_label='Minimum train + validation NLL')
    plotting.reference_tables(contrasts, figures, selection_label='Minimum train + validation NLL')
    main_output = REPO/'results/main'
    plotting.main_gru_monogru(pairs, selected, contrasts, main_output)
    appendix_path = plotting.appendix_all_models(pairs, selected, REPO/'results/appendix')
    effect_path = plotting.appendix_cohens_d(contrasts, selected, REPO/'results/appendix')
    manifest['appendix_outputs'] = dict(figure=str(appendix_path), sha256=measures.sha256(appendix_path),
        cohens_d_figure=str(effect_path), cohens_d_sha256=measures.sha256(effect_path),
        cohens_d_table_sha256=measures.sha256(effect_path.parent/'table_mech_var_cohens_d.csv'))
    manifest['main_outputs'] = dict(directory=str(main_output),
        output_sha256={p.name: measures.sha256(p) for p in sorted(main_output.glob('*GRU_monoGRU_mech_var.*'))})
    manifest['plots'] = dict(directory=str(figures), source_code_sha256=measures.sha256(plotting.__file__),
        output_sha256={p.name: measures.sha256(p) for p in sorted(figures.iterdir()) if p.is_file()})
    print(f'Plots: saved to {figures}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=DEFAULT_RUN)
    parser.add_argument('--stage', choices=['all', 'compute', 'statistics', 'plots'], default='all')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--recompute', action='store_true', help='Recompute similarities even if the verified cache matches.')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.recompute and args.stage not in ['all', 'compute']:
        parser.error('--recompute requires --stage all or compute')
    torch.set_num_threads(1)
    os.chdir(REPO.parent)
    tables, figures, cache = paths_for(args.folder)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest_path = tables/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if 'Shared subject-balanced' in manifest.get('computation', {}).get('input_protocol', ''):
        archived=REPO/'data/analysis/.local/mechanistic_variability/shared_reference_previous'
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            raise FileExistsError(f'Previous-method archive already exists: {archived}')
        tables.rename(archived);tables.mkdir(parents=True)
        if figures.exists() and any(figures.iterdir()):
            figures.rename(figures.with_name(figures.name+'_shared_reference_previous'))
            figures.mkdir(parents=True)
        manifest={}
        print(f'Preserved superseded shared-reference outputs in {archived}',flush=True)
    if args.stage not in ['all', 'compute'] and manifest.get('folder', args.folder) != args.folder:
        raise ValueError('Cached analysis uses a different training run; rerun --stage compute for --folder.')
    manifest.update(study='mechanistic_variability', folder=args.folder)
    if args.stage in ['all', 'compute']:
        selected, pairs = compute(args.folder, tables, cache, manifest, args.jobs, args.recompute)
        write_json(manifest_path, manifest)
    else:
        selected, pairs = load_computation(tables, manifest)
    if args.stage in ['all', 'statistics']:
        statistics(selected, pairs, tables, manifest)
        write_json(manifest_path, manifest)
    if args.stage in ['all', 'plots']:
        plots(selected, pairs, tables, figures, manifest)
        write_json(manifest_path, manifest)
    print(f'Dataframes and provenance: {tables}', flush=True)


if __name__ == '__main__':
    main()
