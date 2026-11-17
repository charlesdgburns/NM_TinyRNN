"""Mechanistic variability: compute selected-model similarities, statistics, then plots.

From the repository root:
    .venv/bin/python code/analyses/mechanistic_variability_main.py
    .venv/bin/python code/analyses/mechanistic_variability_main.py --stage plots

Main-paper entry point. Appendix reports: mechanistic_variability_appendix.py.
Default report stage consumes verified cached similarities; --stage all computes them.
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

DEFAULT_RUN = 'nested_DA_S32_B8'
SELECTION = 'performance.select_best_outer: minimum block-reset train+validation free-choice CE'

# Authoritative reporting policy. Metric code and plotting remain reusable helpers.
METHODS = dict(
    selection='Minimum free-choice train+validation NLL on original saved blocks; reset to learned initial state per block; no test targets or penalties',
    evaluation='Continuous complete subject A saved inputs, including forced trials; B aligned to A; no block/session resets during similarity evaluation',
    alignment='Tanh permutation/sign search; ReLU grid winner then continuous positive-scale refinement; main-parameter Euclidean objective',
    scores='All-trainable-parameter cosine; mean valid unitwise hidden Pearson; flattened update/reset gates and logit-difference Pearson',
    aggregation='Within: 45 fold pairs -> one mean per subject (11); across: 100 fold pairs -> one mean per dyad (55, sharing 11 subjects)',
    within_test='Exact two-sided sign test of direction/median of independent subject paired differences; not a mean test',
    across_test='Mean dyad difference; delete each subject and all incident dyads; jackknife SE; approximate two-sided t(10)',
    corrections='BH across 8 focused GRU/monoGRU tests; separate BH family of 94 exploratory all-configuration contrasts; no reference/absent-gate tests',
    effect_size='Supplementary Cohen dz: paired mean/SD of 11 subject differences; across uses 11 overlapping incident-dyad means and is descriptive only',
    deletion_ranges='Within median and across mean deletion ranges, plus explicit across median deletion range; sensitivity ranges, not confidence intervals',
    highlighting='Within stars annotate median direction difference; across stars annotate mean similarity, not median differences',
    limitations='11 subjects; shared outer-fold models; across inference approximate; exploratory comparisons; no normality pretest is used to select tests')



def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def paths_for(folder):
    if Path(folder).name != folder or folder in ['', '.', '..']:
        raise ValueError('--folder must name one run under data/rnns')
    tables = REPO/'data/analyses/mechanistic_variability'
    figures = REPO/'results/main'
    cache = tables/'pairwise_similarities.htsv'
    return tables, figures, cache


def publish_notebook_cache(pairs_path, cache, tables, computation):
    """Expose the study pair table directly to notebooks, without a duplicate cache."""
    if cache.exists() and measures.sha256(cache) != measures.sha256(pairs_path):
        backup = REPO/'data/analyses/.local/mechanistic_variability'/f'previous_notebook_cache_{measures.sha256(cache)[:12]}.htsv'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(cache, backup)
    if not cache.exists() or measures.sha256(cache) != measures.sha256(pairs_path):
        temporary = cache.with_suffix('.htsv.tmp')
        shutil.copy2(pairs_path, temporary)
        temporary.replace(cache)
    write_json(tables/'notebook_cache.json', dict(computation,
        study='code/analyses/mechanistic_variability_main.py', tables=str(tables)))


def compute(folder, tables, cache, manifest, jobs=3, recompute=False):
    """Select exact notebook winners and compare responses on subject A own trials."""
    config_path = REPO/'data/rnns'/folder/'_aws/run-config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    else:
        # Native SLURM exports carry their actual settings in every info file.
        infos = sorted((REPO/'data/rnns'/folder).rglob('*_info.json'))
        if not infos:
            raise FileNotFoundError(f'No saved model settings found in {folder}')
        settings = set()
        hashes = {}
        for path in infos:
            info = json.loads(path.read_text())
            settings.add((info['sequence_length'],info['hparam_ranges']['batch_size'],
                          info['hparam_ranges']['early_stop']))
            hashes[str(path.relative_to(REPO))] = measures.sha256(path)
        if len(settings) != 1:
            raise ValueError(f'Mixed native run settings: {settings}')
        length,batch,patience = settings.pop()
        config = dict(source_run=folder,sequence_length=length,
                      trainer=dict(batch_size=batch,early_stop=patience),
                      provenance='Derived from all saved info files; RNG settings are not inferred',
                      info_sha256=hashes)
        config_path = tables/'native_run_settings.json'
        write_json(config_path,config)
    experiment_df = experiments.get_DA_info_df(folder_name=folder,
        sequence_length=config['sequence_length'], batch_size=config['trainer']['batch_size'])
    analysis_df = analysis.get_analysis_df(experiment_df, n_jobs=1, use_cache=not recompute)
    performance_df = performance.get_performance_df(analysis_df, n_jobs=jobs, use_cache=not recompute)
    keys = ['subject_id', 'model_id', 'outer_loop_n', 'inner_loop_idx']
    if set(analysis_df[keys].itertuples(index=False, name=None)) != set(performance_df[keys].itertuples(index=False, name=None)):
        raise ValueError('Performance cache does not match the discovered inner models')
    selected = performance.select_best_outer(performance_df).sort_values(
        ['model_id', 'subject_id', 'outer_loop_n']).reset_index(drop=True)
    performance_tables = REPO/'data/analyses/performance'
    selected.to_csv(performance_tables/'best_model_df.htsv',sep='\t',index=False)
    subject_performance = performance.compute_outer_mean(selected)
    subject_performance.to_csv(performance_tables/'subject_level_df.htsv',sep='\t',index=False)
    inputs, references = measures.build_subject_trial_inputs(selected)
    # Model bytes, selected endpoints, probe bytes, settings and metric code identify
    # reuse; run-name-only caching was the source of the original selection mismatch.
    model_hashes = {str(path): measures.sha256(path) for path in selected.model_pickle_path}
    fingerprint_data = dict(selection=SELECTION, selected=selected[keys].to_dict('records'),
        model_sha256=model_hashes, reference_protocol='subject_A_continuous_trials', alignment_protocol=measures.DEFAULT_ALIGNMENT_PROTOCOL,
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
    computation = dict(fingerprint=fingerprint, alignment_protocol=measures.DEFAULT_ALIGNMENT_PROTOCOL, selection=SELECTION, n_models=len(selected),
        n_variants=int(selected.model_id.nunique()), n_pairs=len(pairs), input_shape={subject:list(value.shape) for subject,value in inputs.items()},
        input_protocol='Subject A complete saved trial sequence; within subject uses own data; continuous state as in original notebook',
        pair_direction='Original sorted subject/outer ordering; A supplies trials, without averaging the reverse direction',
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
        raise FileNotFoundError('Run --stage compute or --stage all first.')
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
    # Keep the previous all-jackknife/Holm analysis as a labelled sensitivity table.
    contrasts.to_csv(tables/'similarity_vs_standard_gru_jackknife_sensitivity.htsv',sep='\t',index=False)
    from scipy.stats import binomtest, false_discovery_control
    contrasts['p_raw_jackknife']=contrasts['p_raw']
    contrasts['test_method']='Subject-deletion jackknife t'
    for index,row in contrasts[contrasts.within_subject & contrasts.available].iterrows():
        group=subject_pairs[subject_pairs.within_subject & subject_pairs.metric.eq(row.metric)]
        a=group[group.model_id.eq(measures.STANDARD_GRU_REFERENCE)][['subject_A','subject_B','mean_similarity']]
        b=group[group.model_id.eq(row.model_id)][['subject_A','subject_B','mean_similarity']]
        paired=a.merge(b,on=['subject_A','subject_B'],suffixes=('_reference','_model'),validate='one_to_one')
        delta=(paired.mean_similarity_model-paired.mean_similarity_reference).dropna()
        nonzero=delta[delta!=0]
        contrasts.loc[index,'p_raw']=binomtest(int((nonzero>0).sum()),len(nonzero),.5).pvalue if len(nonzero) else 1.
        contrasts.loc[index,'test_method']='Exact sign test'
    tested=contrasts.available & contrasts.model_id.ne(measures.STANDARD_GRU_REFERENCE)
    contrasts['p_bh']=np.nan
    contrasts.loc[tested,'p_bh']=false_discovery_control(contrasts.loc[tested,'p_raw'].to_numpy(),method='bh')
    contrasts['p_holm']=np.nan
    contrasts.loc[tested,'p_holm']=measures.holm(contrasts.loc[tested,'p_raw'].to_numpy())
    contrasts['significance']=''
    for cutoff,mark in [(.05,'*'),(.01,'**'),(.001,'***')]:contrasts.loc[contrasts.p_bh.lt(cutoff),'significance']=mark
    contrasts.to_csv(tables/'similarity_vs_standard_gru.htsv', sep='\t', index=False)
    manifest['statistics'] = dict(reference=measures.STANDARD_GRU_REFERENCE,
        n_tests=int(contrasts.p_bh.notna().sum()), n_significant=int(contrasts.p_bh.lt(.05).sum()),
        inference='Exact sign within; subject-deletion jackknife/t across; descriptive dz; BH across exploratory all-model family',
        caveat='Exploratory inference with 11 subjects; across subject summaries share dyads. Pointwise CIs are unadjusted.',
        pair_cache_sha256=manifest['computation']['cache_sha256'],
        output_sha256={name: measures.sha256(tables/name) for name in
            ['subject_pair_similarity.htsv', 'similarity_summary.htsv', 'similarity_vs_standard_gru.htsv']})
    print(f'Statistics: {len(contrasts)} cells, {manifest["statistics"]["n_tests"]} tests, '
          f'{manifest["statistics"]["n_significant"]} BH-corrected differences.', flush=True)
    return subject_pairs, contrasts


def plots(selected, pairs, tables, figures, manifest):
    """Main-paper outputs only; appendix reports have their own entry point."""
    from NM_TinyRNN.code.figures import mechanistic_variability as plotting
    from NM_TinyRNN.code.measures.gru_monogru_report import generate_report
    meta=manifest.get('statistics')
    if not meta or meta['pair_cache_sha256']!=manifest['computation']['cache_sha256']:
        raise ValueError('Run --stage statistics before --stage plots, or use --stage report')
    for name,expected in meta['output_sha256'].items():
        if measures.sha256(tables/name)!=expected:
            raise ValueError(f'Statistics changed: {name}; rerun --stage statistics')
    contrasts=pd.read_csv(tables/'similarity_vs_standard_gru.htsv',sep='\t')
    output=REPO/'results/main';output.mkdir(parents=True,exist_ok=True)
    generate_report(pairs,root=REPO,include_appendix=False)
    plotting.main_gru_monogru(pairs,selected,contrasts,output)
    plotting.main_gru_monogru(pairs,selected,contrasts,output,include_gate=True)
    manifest['main_outputs']=dict(directory=str(output),output_sha256={p.name:measures.sha256(p) for p in output.glob('*') if p.is_file() and (p.name.startswith('table_1.') or 'GRU_monoGRU_mech_var' in p.name)})
    manifest['plots']=dict(scope='main paper',source_code_sha256=measures.sha256(plotting.__file__),directory=str(output))
    write_json(tables/'main_methods.json',METHODS)
    print(f'Main-paper figures and Table 1: {output}',flush=True)


def exported_examples(args):
    """Compute from portable winner exports without the original training tree."""
    from NM_TinyRNN.code.figures import mechanistic_variability as plotting
    root = Path(args.examples).resolve()
    export_path = root/'export_manifest.json'
    winner_path = root/'outer_fold_winners.htsv'
    export = json.loads(export_path.read_text())
    selected = pd.read_csv(winner_path, sep='\t')
    ids = ['2_unit_GRU_tanh_unipolar_forced', '2_unit_monoGRU_relu_unipolar_forced_ndb']
    selected = selected[selected.model_id.isin(ids)].copy().sort_values(['model_id','subject_id','outer_loop_n']).reset_index(drop=True)
    if selected.duplicated(['subject_id','model_id','outer_loop_n']).any():
        raise ValueError('Duplicate exported outer winners')
    counts = selected.groupby(['model_id','subject_id']).outer_loop_n.agg(list)
    if len(selected) != 220 or any(sorted(values) != list(range(1,11)) for values in counts):
        raise ValueError('Expected 11 subjects, two configurations and all ten outer winners')
    expected = {str(Path(*Path(item['destination']).parts[-4:])):item['sha256'] for item in export['artifacts']}
    artifacts = {}; patience = set(); trial_columns = ['session_folder_name','forced_choice','choice','outcome']
    references = {}
    for index, row in selected.iterrows():
        folder = root/row.subject_id/row.model_type/f'outer_fold_{int(row.outer_loop_n)}'
        for column, suffix in [('info_path','_info.json'),('model_pickle_path','_model.pickle'),('trials_data_path','_trials_data.htsv')]:
            path = folder/f'{row.model_id}{suffix}'
            relative = str(path.relative_to(root)); digest = measures.sha256(path)
            if expected.get(relative) != digest:
                raise ValueError(f'Export checksum mismatch: {path}')
            selected.loc[index, 'source_'+column] = row[column]
            selected.loc[index, column] = str(path)
            artifacts[relative] = digest
        info = json.loads(Path(selected.loc[index,'info_path']).read_text())
        if int(info['outer_loop_number']) != int(row.outer_loop_n) or int(info['inner_fold_idx']) != int(row.inner_loop_idx):
            raise ValueError('Winner table and model info fold identities differ')
        patience.add(info['hparam_ranges']['early_stop'])
        trials = pd.read_csv(selected.loc[index,'trials_data_path'], sep='\t')[trial_columns]
        if row.subject_id not in references:
            references[row.subject_id] = trials
        else:
            pd.testing.assert_frame_equal(references[row.subject_id], trials)
    tables = REPO/'data/analyses/mechanistic_variability/exported_examples'
    output = REPO/'results/main/mechanistic_variability_examples'
    tables.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    inputs, reference_rows = measures.build_subject_trial_inputs(selected)
    definition = dict(source_export=str(root), source_run=export['source_run'], selection=export['selection'],
        export_sha256=measures.sha256(export_path), winner_table_sha256=measures.sha256(winner_path),
        artifact_sha256=artifacts, alignment_protocol=measures.DEFAULT_ALIGNMENT_PROTOCOL,
        measures_sha256=measures.sha256(measures.__file__), patience=sorted(patience),
        inputs_sha256={s:hashlib.sha256(x.numpy().tobytes()).hexdigest() for s,x in inputs.items()})
    fingerprint = hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    manifest_path = tables/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pair_path = tables/'pairwise_similarities.htsv'
    reuse = (not args.recompute and manifest.get('fingerprint') == fingerprint and pair_path.exists()
             and measures.sha256(pair_path) == manifest.get('computation',{}).get('cache_sha256'))
    if args.stage in ['plots','statistics'] and not reuse:
        raise ValueError('Export or scoring inputs changed; run --stage all or compute first')
    selected.to_csv(tables/'selected_models.htsv', sep='\t', index=False)
    reference_rows.to_csv(tables/'reference_subjects.htsv', sep='\t', index=False)
    if not reuse:
        print(f'Exported examples: {len(selected)} verified models; patience={sorted(patience)}; computing fresh similarities.',flush=True)
        with parallel_config(backend='loky',inner_max_num_threads=1):
            validation = Parallel(n_jobs=min(args.jobs,2))(delayed(measures.compute_reference_group)(group.copy(), inputs, str(tables))
                for _,group in selected.groupby('model_id'))
        pairs = pd.concat([pd.read_csv(tables/'groups'/f'{key}.htsv',sep='\t') for key in ids],ignore_index=True)
        measures.validate_similarity_cache_models(pairs, selected)
        pairs.to_csv(pair_path,sep='\t',index=False)
        manifest = dict(definition, fingerprint=fingerprint, computation=dict(validation=validation,
            cache_sha256=measures.sha256(pair_path), selected_sha256=measures.sha256(tables/'selected_models.htsv')))
    else:
        pairs = pd.read_csv(pair_path,sep='\t')
        measures.validate_similarity_cache_models(pairs, selected)
    if args.stage in ['all','statistics']:
        statistics(selected,pairs,tables,manifest)
    if args.stage in ['all','plots']:
        contrasts = pd.read_csv(tables/'similarity_vs_standard_gru.htsv',sep='\t')
        plotting.main_gru_monogru(pairs, selected, contrasts, output)
        plotting.main_gru_monogru(pairs, selected, contrasts, output, include_gate=True)
        manifest['plots'] = dict(directory=str(output), plotting_sha256=measures.sha256(plotting.__file__),
            files_sha256={p.name:measures.sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(manifest_path, manifest)
    (output/'README.md').write_text(f'# Exported-example mechanistic similarities\n\nSource: {root}\n\nTraining patience: {sorted(patience)} epochs without improvement. '
        f'{len(selected)} saved outer-fold winners; selection: {export["selection"]}.\n\n'
        f'The saved reference trials span {reference_rows.n_sessions.min()}–{reference_rows.n_sessions.max()} sessions per subject.\n\n'
        'Within-subject pairs use that subject’s saved inputs. Across-subject pairs use subject A’s inputs for both models. '
        'B is aligned to A using the current continuous-refinement protocol; state is continuous across the saved trial sequence. '
        'Hidden similarity averages unitwise correlations; logits and gates use flattened Pearson correlations. '
        'The with_gate figure adds scalar monoGRU/vector GRU update-gate similarity to the standard three-component plot.\n\n'
        'Reproduce from the repository root:\n```bash\n.venv/bin/python code/analyses/mechanistic_variability_main.py '
        f'--examples {root.relative_to(REPO)} --jobs 2\n```\n')
    print(f'Exported-example outputs: {output}',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=DEFAULT_RUN)
    parser.add_argument('--examples', type=Path, help='Use copied outer winners in an export directory; keep full-run caches and figures separate')
    parser.add_argument('--stage', choices=['report', 'all', 'compute', 'statistics', 'plots'], default='report', help='report (default) verifies cached similarities, updates statistics and main figures; all also computes similarities')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--recompute', action='store_true', help='Recompute similarities even if the verified cache matches.')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.recompute and args.stage not in ['all', 'compute']:
        parser.error('--recompute requires --stage all or compute')
    torch.set_num_threads(1)
    if args.examples:
        exported_examples(args)
        return
    os.chdir(REPO.parent)
    tables, figures, cache = paths_for(args.folder)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest_path = tables/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if 'Shared subject-balanced' in manifest.get('computation', {}).get('input_protocol', ''):
        archived=REPO/'data/analyses/.local/mechanistic_variability/shared_reference_previous'
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
    manifest.update(study='mechanistic_variability', folder=args.folder, methods=METHODS)
    write_json(tables/'methods.json',METHODS)
    if args.stage in ['all', 'compute']:
        selected, pairs = compute(args.folder, tables, cache, manifest, args.jobs, args.recompute)
        write_json(manifest_path, manifest)
    else:
        selected, pairs = load_computation(tables, manifest)
    if args.stage in ['report', 'all', 'statistics']:
        statistics(selected, pairs, tables, manifest)
        write_json(manifest_path, manifest)
    if args.stage in ['report', 'all', 'plots']:
        plots(selected, pairs, tables, figures, manifest)
        write_json(manifest_path, manifest)
    print(f'Dataframes and provenance: {tables}', flush=True)


if __name__ == '__main__':
    main()
