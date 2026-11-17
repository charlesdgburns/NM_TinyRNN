"""Validate GRU/monoGRU mechanistic variability in a saved subject cohort.

From the repository root:
    .venv/bin/python code/analyses/cohort_validation.py --jobs 3
    .venv/bin/python code/analyses/cohort_validation.py --train-missing --training-jobs 6

Consumes existing fits, optionally completing missing native fits locally first.
Isolated, input-verified caches and main results:
    data/analyses/cohort_validation/<folder>/
    results/main/<folder>/
Defaults to the serotonin cohort, including WS18 when its fits are available.
Absent subjects are recorded; partial subjects raise rather than bias selection.
"""
import os
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from scipy.stats import binomtest, false_discovery_control
from NM_TinyRNN.code.models import experiments, datasets, nested_cv, rnns
from NM_TinyRNN.code.measures import analysis, performance, block_performance
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.figures import mechanistic_variability as figures

SUBJECTS = [f'WS{i:02d}' for i in
            [3, 4, 6, 7, 11, 12, 15, 17, 18, 21, 23, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36]]
MODELS = [mv.STANDARD_GRU_REFERENCE, '2_unit_monoGRU_relu_unipolar_forced_ndb']
METRICS = [('logit_value_similarity', 'Logit'), ('weight_similarity', 'Weight'),
           ('hidden_similarity', 'Hidden state'), ('gate_update_similarity', 'Update gate')]


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def train_missing(folder, subjects, jobs):
    """Complete native CPU fits with the same settings as serotonin SLURM jobs.

    Retains complete outer folds, and checks all three required artifacts before
    deciding a fold is complete. Each inner fit uses the unchanged full search.
    """
    if torch.cuda.is_available():
        raise ValueError('Local CPU fitting requires CUDA_VISIBLE_DEVICES=""')
    root = REPO/'data/rnns'/folder
    existing = sorted(root.rglob('*_info.json'))
    if not existing:
        raise ValueError('An existing cohort run is required to verify training settings')
    settings = set()
    for path in existing:
        info = json.loads(path.read_text())
        settings.add((info['sequence_length'], info['hparam_ranges']['batch_size']))
    if len(settings) != 1:
        raise ValueError(f'Mixed source training settings: {settings}')
    length, batch = settings.pop()
    frame = experiments.get_job_info_df(
        processed_data_path=REPO/'data/AB_behaviour', save_path=REPO/'data/rnns',
        folder_name=folder, sequence_length=length, batch_size=batch)
    frame = frame[frame.subject_id.isin(subjects) & frame.model_id.isin(MODELS)]
    if set(subjects) != set(frame.subject_id):
        raise ValueError(f'Missing subject data: {sorted(set(subjects)-set(frame.subject_id))}')
    def complete(row):
        outer = Path(row.save_path)/f'outer_fold_{row.outer_loop_n}'
        return all((outer/f'inner_fold_{inner}'/f'{row.model_id}_{suffix}').is_file()
                   for inner in range(9) for suffix in ['model.pickle', 'info.json', 'trials_data.htsv'])
    for (subject, model_id), rows in frame.groupby(['subject_id', 'model_id'], sort=True):
        pending = [row for row in rows.itertuples(index=False) if not complete(row)]
        if not pending:
            continue
        row = pending[0]
        print(f'Local training: {subject} {model_id}, {len(pending)} outer folds, {jobs} CPU workers.', flush=True)
        data = datasets.AB_Dataset(row.data_path, sequence_length=length,
                                  input_encoding=row.input_encoding,
                                  input_forced_choice=bool(row.input_forced_choice))
        options = rnns.OPTIONS_DICT.copy()
        options.update(rnn_type=row.model_type, hidden_size=int(row.hidden_size),
                       nm_size=int(row.nm_size), nm_dim=int(row.nm_dim), nm_mode=row.nm_mode,
                       input_encoding=row.input_encoding, input_forced_choice=bool(row.input_forced_choice),
                       nonlinearity=row.nonlinearity, decoder_bias=bool(row.decoder_bias),
                       input_size=data.inputs.shape[-1])
        model = rnns.TinyRNN(**options)
        kwargs = dict(batch_size=batch)
        if row.constraint == 'sparsity':
            kwargs['energy_lambdas'] = [0]
        from NM_TinyRNN.code.models.training import Trainer
        expected = Trainer(**kwargs)
        # Match every recorded search/trainer parameter, not only S/B/patience.
        for reference_path in existing:
            if reference_path.name != f'{model_id}_info.json':
                continue
            reference = json.loads(reference_path.read_text())['hparam_ranges']
            for key, value in reference.items():
                if not hasattr(expected, key) or getattr(expected, key) != value:
                    raise ValueError(f'Trainer differs from existing fits: {model_id} {key}')
            break
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for row in pending:
                nested_cv.run_outer_fold(model, data, int(row.outer_loop_n),
                    n_outer_loops=10, trainer_kwargs=kwargs, save_path=Path(row.save_path), n_jobs=jobs)
                if not complete(row):
                    raise ValueError(f'Incomplete exports after training {subject} {model_id}, outer {row.outer_loop_n}')
        gc.collect()
    print('Local training: all requested subject/model fits are complete.', flush=True)


def discover(folder, subjects):
    root = REPO/'data/rnns'/folder
    present = [s for s in subjects if any((root/s).rglob('*_info.json'))]
    missing = sorted(set(subjects)-set(present))
    if len(present) < 3:
        raise ValueError('At least three subjects with saved fits are required')
    infos = sorted(p for s in present for p in (root/s).rglob('*_info.json')
                   if any(p.name == f'{model}_info.json' for model in MODELS))
    settings = set()
    for path in infos:
        info = json.loads(path.read_text())
        settings.add((int(info['sequence_length']), int(info['hparam_ranges']['batch_size']),
                      int(info['hparam_ranges']['early_stop'])))
    if len(settings) != 1:
        raise ValueError(f'Expected uniform sequence/batch/patience settings: {settings}')
    length, batch, patience = settings.pop()
    frame = experiments.get_job_info_df(
        processed_data_path=REPO/'data/AB_behaviour', save_path=REPO/'data/rnns',
        folder_name=folder, sequence_length=length, batch_size=batch)
    frame = frame[frame.subject_id.isin(present) & frame.model_id.isin(MODELS)].copy()
    expected = {(s, m, o) for s in present for m in MODELS for o in range(1, 11)}
    if set(frame[['subject_id', 'model_id', 'outer_loop_n']].itertuples(index=False, name=None)) != expected:
        raise ValueError('Cohort manifest is missing subject/model/outer-fold combinations')
    if not frame.completed.all():
        raise ValueError('Partially trained cohort; all 10 outer × 9 inner fits are required per model/subject')
    paths = pd.DataFrame([item for row in frame.itertuples(index=False)
                          for item in analysis.get_model_paths(row)])
    for _, group in paths.groupby(['subject_id', 'model_id', 'outer_loop_n']):
        if sorted(group.inner_loop_idx) != list(range(9)):
            raise ValueError('Expected all nine unique inner fits in each outer fold')
    if len(paths) != len(present)*2*90:
        raise ValueError('Unexpected native artifact count')
    artifact_hashes = {}
    for row in paths.itertuples():
        for field in ['info_path', 'model_pickle_path', 'trials_data_path']:
            path = Path(getattr(row, field))
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Missing/empty native artifact: {path}')
            artifact_hashes[str(path.relative_to(REPO))] = mv.sha256(path)
    metadata = dict(subjects=sorted(present), unavailable_subjects=missing,
                    sequence_length=length, batch_size=batch, early_stop=patience,
                    artifact_sha256=artifact_hashes)
    return frame, paths, metadata


def select_models(paths, tables, fingerprint, manifest, jobs, recompute):
    cache = tables/'performance_df.htsv'
    prior = manifest.get('performance', {})
    if (not recompute and prior.get('fingerprint') == fingerprint and cache.is_file()
            and prior.get('sha256') == mv.sha256(cache)):
        print('Performance: reusing verified block-reset scores.', flush=True)
        scores = pd.read_csv(cache, sep='\t')
        if not analysis.cache_matches_models(scores, paths):
            raise ValueError('Performance cache has different model paths')
    else:
        print(f'Performance: scoring {len(paths)} saved inner fits with {jobs} workers.', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            metrics = Parallel(n_jobs=jobs, verbose=5)(
                delayed(performance.get_model_performance)(row) for row in paths.itertuples())
        scores = pd.concat([paths.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
        scores = performance.add_block_scores(scores, n_jobs=jobs)
        scores.to_csv(cache, sep='\t', index=False)
        manifest['performance'] = dict(fingerprint=fingerprint, sha256=mv.sha256(cache))
    selected = performance.select_best_outer(scores).sort_values(
        ['model_id', 'subject_id', 'outer_loop_n']).reset_index(drop=True)
    selected.to_csv(tables/'selected_models.htsv', sep='\t', index=False)
    performance.compute_outer_mean(selected).to_csv(tables/'subject_performance.htsv', sep='\t', index=False)
    return selected


def similarities(selected, tables, manifest, recompute):
    inputs, references = mv.build_subject_trial_inputs(selected)
    references.to_csv(tables/'reference_subjects.htsv', sep='\t', index=False)
    np.savez_compressed(tables/'reference_inputs.npz', **{s: x.numpy() for s, x in inputs.items()})
    fingerprint = digest(dict(
        selected_sha256=mv.sha256(tables/'selected_models.htsv'),
        inputs={s: hashlib.sha256(x.numpy().tobytes()).hexdigest() for s, x in inputs.items()},
        source=manifest['performance']['fingerprint'], alignment=mv.DEFAULT_ALIGNMENT_PROTOCOL))
    groups = manifest.setdefault('groups', {})
    for model, rows in selected.groupby('model_id', sort=True):
        path = tables/'groups'/f'{model}.htsv'
        old = groups.get(model, {})
        if (not recompute and old.get('fingerprint') == fingerprint and path.is_file()
                and old.get('sha256') == mv.sha256(path)):
            print(f'Similarities: reusing verified {model} pairs.', flush=True)
        else:
            print(f'Similarities: computing {len(rows)*(len(rows)-1)//2} {model} pairs.', flush=True)
            # Sequential configurations bound memory when batching all models/subject sequences.
            validation = mv.compute_reference_group(rows.copy(), inputs, str(tables))
            groups[model] = dict(fingerprint=fingerprint, sha256=mv.sha256(path), validation=validation)
            write_json(tables/'manifest.json', manifest)
            gc.collect()
        mv.validate_similarity_cache_models(pd.read_csv(path, sep='\t'), rows)
    pairs = pd.concat([pd.read_csv(tables/'groups'/f'{m}.htsv', sep='\t') for m in MODELS], ignore_index=True)
    mv.validate_similarity_cache_models(pairs, selected)
    pairs.to_csv(tables/'pairwise_similarities.htsv', sep='\t', index=False)
    manifest['pairwise_sha256'] = mv.sha256(tables/'pairwise_similarities.htsv')
    return pairs


def statistics(pairs, n_subjects):
    subjects, means = mv.summarize_reference_pairs(pairs)
    records, paired = [], []
    for metric, label in METRICS:
        for scope in [True, False]:
            data = subjects[subjects.metric.eq(metric) & subjects.within_subject.eq(scope)]
            keys = ['subject_A', 'subject_B']
            a = data[data.model_id.eq(MODELS[0])][keys+['mean_similarity']].rename(columns={'mean_similarity': 'gru'})
            b = data[data.model_id.eq(MODELS[1])][keys+['mean_similarity']].rename(columns={'mean_similarity': 'mono'})
            d = a.merge(b, on=keys, validate='one_to_one')
            expected = n_subjects if scope else n_subjects*(n_subjects-1)//2
            if len(d) != expected or not np.isfinite(d[['gru', 'mono']]).all().all():
                raise ValueError(f'Missing/nonfinite subject summaries for {metric}, within={scope}')
            d['difference'] = d.mono-d.gru
            d['metric'], d['component'], d['within_subject'] = metric, label, scope
            paired.append(d)
            c = mv.contrast(d.rename(columns={'gru': 'reference', 'mono': 'model'}))
            names = sorted(set(d.subject_A) | set(d.subject_B))
            estimator = np.median if scope else np.mean
            loo = [estimator(d.loc[~(d.subject_A.eq(s) | d.subject_B.eq(s)), 'difference']) for s in names]
            median_loo = [d.loc[~(d.subject_A.eq(s) | d.subject_B.eq(s)), 'difference'].median() for s in names]
            positive, nonzero = int(d.difference.gt(0).sum()), int(d.difference.ne(0).sum())
            p = binomtest(positive, nonzero, .5).pvalue if scope and nonzero else (1. if scope else c['p_raw'])
            records.append(dict(metric=metric, component=label, within_subject=scope,
                n_subjects=n_subjects, n_units=len(d), gru_mean=d.gru.mean(), gru_sd=d.gru.std(ddof=1),
                mono_mean=d.mono.mean(), mono_sd=d.mono.std(ddof=1),
                median_difference=d.difference.median(), mean_difference=d.difference.mean(),
                median_deletion_min=min(median_loo), median_deletion_max=max(median_loo),
                deletion_min=min(loo), deletion_max=max(loo), cohens_dz=c['cohens_dz'],
                n_positive=positive, n_nonzero=nonzero, jackknife_se=c['jackknife_se'],
                test_method='Exact sign test' if scope else f'Subject-deletion jackknife t({n_subjects-1})',
                p_main=p))
    summary = pd.DataFrame(records)
    summary['q_main_bh8'] = false_discovery_control(summary.p_main.to_numpy(), method='bh')
    return subjects, means, summary, pd.concat(paired, ignore_index=True)


def stars(q):
    return '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''


def main_table(summary, path, folder):
    """Minimal publication layout with grouped headers and cohort-aware notes."""
    n = int(summary.n_subjects.iloc[0])
    with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                         'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.axis('off')
        ax.set(xlim=(0, 1), ylim=(0, 1))
        positions = [.025, .235, .355, .475, .675, .795, .915]
        ax.text(.5, .97, 'Table 1. Similarity of GRU and monoGRU models',
                ha='center', va='top', fontsize=16)
        for x, heading in [(.355, 'Within subject'), (.795, 'Across subjects')]:
            ax.text(x, .78, heading, ha='center', fontweight='bold', fontsize=14)
        headers = ['Component', 'GRU', 'monoGRU', 'Median Δ', 'GRU', 'monoGRU', 'Median Δ']
        for i, (x, heading) in enumerate(zip(positions, headers)):
            ax.text(x, .68, heading, ha='left' if i == 0 else 'center',
                    fontweight='bold', fontsize=12)
        ax.plot([.015, .99], [.63, .63], color='black', lw=.8)
        for i, (metric, label) in enumerate(METRICS):
            group = summary[summary.metric.eq(metric)]
            w = group[group.within_subject].iloc[0]
            a = group[~group.within_subject].iloc[0]
            ws, acs = stars(w.q_main_bh8), stars(a.q_main_bh8)
            cells = [label, f'{w.gru_mean:.3f}', f'{w.mono_mean:.3f}',
                     f'{w.median_difference:+.4f}' + (r'$^{' + ws + '}$' if ws else ''),
                     f'{a.gru_mean:.3f}', f'{a.mono_mean:.3f}' + (r'$^{' + acs + '}$' if acs else ''),
                     f'{a.median_difference:+.4f}']
            for j, (x, text) in enumerate(zip(positions, cells)):
                bold = (j == 3 and bool(ws)) or (j == 5 and bool(acs))
                ax.text(x, .55-i*.10, text, ha='left' if j == 0 else 'center',
                        va='center', fontsize=12, fontweight='bold' if bold else 'normal')
        ax.plot([.015, .99], [.18, .18], color='black', lw=.8)
        footnote = (
            f'GRU/monoGRU columns: mean similarity over {n} subject summaries (within) or {n*(n-1)//2} subject-pair summaries (across). '
            'Median Δ = median paired monoGRU − GRU difference.\n'
            f'Within stars: exact two-sided sign test on {n} independent subject differences. '
            f'Across stars: mean-difference test using subject-deletion jackknife SE and approximate t({n-1}).\n'
            'BH across all eight comparisons (* q < .05, ** < .01, *** < .001). Across stars annotate the mean, '
            f'not the median. Block-reset winners; {folder}.')
        ax.text(.015, .135, footnote, ha='left', va='top', fontsize=8.2, linespacing=1.4)
        fig.subplots_adjust(left=.025, right=.98, top=.98, bottom=.02)
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def report(pairs, selected, tables, output, folder):
    n = selected.subject_id.nunique()
    subjects, means, summary, paired = statistics(pairs, n)
    for frame, name in [(subjects, 'subject_pair_similarity'), (means, 'similarity_summary'),
                        (summary, 'summary'), (paired, 'paired_subject_data')]:
        frame.to_csv(tables/f'{name}.csv', index=False)
    summary.to_csv(output/'table_1_statistics.csv', index=False)
    records = []
    for metric, label in METRICS:
        w = summary[summary.metric.eq(metric) & summary.within_subject].iloc[0]
        a = summary[summary.metric.eq(metric) & ~summary.within_subject].iloc[0]
        row = dict(component=label)
        for r, scope in [(w, 'within'), (a, 'across')]:
            row.update({f'{scope}_{key}': r[key] for key in
                        ['gru_mean', 'mono_mean', 'mean_difference', 'median_difference', 'p_main', 'q_main_bh8']})
        records.append(row)
    pd.DataFrame(records).to_csv(output/'table_1.csv', index=False)
    main_table(summary, output/'table_1.svg', folder)
    # The main violin helper is cohort-size independent; inference uses our eight-test family.
    contrasts = mv.compare_similarity_to_reference(subjects)
    contrasts = contrasts.drop(columns=['p_raw', 'p_bh', 'significance'], errors='ignore').merge(
        summary[['metric', 'within_subject', 'p_main', 'q_main_bh8']].assign(model_id=MODELS[1]),
        on=['model_id', 'metric', 'within_subject'], how='left', validate='one_to_one')
    contrasts['p_raw'], contrasts['p_bh'] = contrasts.p_main, contrasts.q_main_bh8
    contrasts['significance'] = contrasts.q_main_bh8.map(lambda q: stars(q) if pd.notna(q) else '')
    for include_gate in [False, True]:
        figures.main_gru_monogru(pairs, selected, contrasts, output, include_gate=include_gate)
    for metric, label in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'width_ratios': [1, 1.5]})
        for index, scope in enumerate([True, False]):
            d = paired[paired.metric.eq(metric) & paired.within_subject.eq(scope)].sort_values(['subject_A', 'subject_B'])
            s = summary[summary.metric.eq(metric) & summary.within_subject.eq(scope)].iloc[0]
            ax = axes[index, 0]
            for r in d.itertuples():
                ax.plot([0, 1], [r.gru, r.mono], color='#888888', alpha=.6 if scope else .15, lw=.8, marker='o', markersize=2)
            ax.set(xticks=[0, 1], xticklabels=['GRU', 'monoGRU'], ylim=(-1, 1), ylabel='Similarity',
                   title=f'{"Within: subjects" if scope else "Across: subject pairs"} (n={len(d)})')
            ax = axes[index, 1]
            y = np.arange(len(d))
            ax.scatter(d.difference, y, s=20 if scope else 7, alpha=.8 if scope else .5)
            ax.axvline(0, color='.5', lw=1)
            ax.axvline(s.median_difference, color='#e58130', ls='--')
            ax.set(yticks=y if scope else [], yticklabels=d.subject_A if scope else [],
                   xlabel='Paired difference: monoGRU − GRU',
                   title=f'{s.test_method} · BH q={s.q_main_bh8:.4g}')
            ax.invert_yaxis()
        fig.suptitle(f'{label} similarity · {folder}')
        fig.text(.5, .02, 'Within: 45 fold pairs per subject; across: 100 per dyad. BH spans eight focused tests.\n'
                 'Orange line: descriptive median. Across inference tests the mean and accounts for shared subjects.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.12, hspace=.4, wspace=.32)
        fig.savefig(output/f'figure_paired_{metric}.svg', bbox_inches='tight', facecolor='white')
        plt.close(fig)
    text = (f'# Cohort validation: {folder}\n\nIncluded subjects: {", ".join(sorted(selected.subject_id.unique()))}.\n\n'
            'Selection: minimum free-choice block-reset train + validation NLL; no test targets or penalties. '
            'Each configuration has ten outer winners per subject. Evaluation uses complete continuous saved inputs '
            'of sorted reference subject A, including forced trials; model B is aligned to A. '
            'Tanh uses permutation/sign alignment; ReLU uses the canonical grid plus continuous positive-scale refinement. '
            'Metrics and inferential tests match the maintained main analysis. '
            'Within: independent subject means, exact two-sided sign tests. Across: dyad means, '
            f'subject-deletion jackknife SE and approximate t({n-1}); dyads are dependent. '
            'BH correction spans the four metrics × two scopes. Cohen dz is descriptive; deletion ranges are influence diagnostics.\n\n'
            f'Intermediate data: {tables}\n\nReproduce:\n```sh\n.venv/bin/python code/analyses/cohort_validation.py --folder {folder}\n```\n')
    (output/'README.md').write_text(text)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='nested_5HT_S32_B8')
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--jobs', type=int, default=3, help='Performance workers; similarity configurations run sequentially to bound RAM')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--train-missing', action='store_true', help='Complete missing native CPU fits before running the analysis')
    parser.add_argument('--training-jobs', type=int, default=6, help='Parallel CPU inner-fold fits for local training')
    args = parser.parse_args(argv)
    if Path(args.folder).name != args.folder or args.folder in ['', '.', '..'] or min(args.jobs, args.training_jobs) < 1:
        parser.error('folder must name one run under data/rnns; jobs must be positive')
    if any(Path(s).name != s or s in ['', '.', '..'] for s in args.subjects):
        parser.error('subjects must be directory names')
    torch.set_num_threads(1)
    os.chdir(REPO.parent)
    if args.train_missing:
        train_missing(args.folder, args.subjects, args.training_jobs)
    tables = REPO/'data/analyses/cohort_validation'/args.folder
    output = REPO/'results/main'/args.folder
    tables.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = tables/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    frame, paths, cohort = discover(args.folder, args.subjects)
    sources = {str(p.relative_to(REPO)): mv.sha256(p) for p in
               [Path(__file__), Path(mv.__file__), Path(performance.__file__),
                Path(block_performance.__file__), REPO/'code/models/rnns.py',
                REPO/'code/models/datasets.py', Path(analysis.__file__)]}
    fingerprint = digest(dict(cohort=cohort, sources=sources))
    manifest.update(folder=args.folder, cohort=cohort, source_sha256=sources,
                    selection='block_reset', alignment_protocol=mv.DEFAULT_ALIGNMENT_PROTOCOL)
    print(f'Cohort: {len(cohort["subjects"])} subjects; unavailable: {cohort["unavailable_subjects"]}', flush=True)
    frame.to_csv(tables/'experiment_df.htsv', sep='\t', index=False)
    paths.to_csv(tables/'analysis_df.htsv', sep='\t', index=False)
    selected = select_models(paths, tables, fingerprint, manifest, args.jobs, args.recompute)
    write_json(manifest_path, manifest)
    pairs = similarities(selected, tables, manifest, args.recompute)
    summary = report(pairs, selected, tables, output, args.folder)
    manifest['outputs_sha256'] = {str(p.relative_to(REPO)): mv.sha256(p) for p in output.iterdir() if p.is_file()}
    manifest['statistics'] = dict(n_tests=8, n_subjects=len(cohort['subjects']),
                                n_significant=int(summary.q_main_bh8.lt(.05).sum()))
    write_json(manifest_path, manifest)
    print(summary[['component', 'within_subject', 'gru_mean', 'mono_mean', 'p_main', 'q_main_bh8']].to_string(index=False), flush=True)
    print(f'Intermediate data: {tables}\nResults: {output}', flush=True)


if __name__ == '__main__':
    main()
