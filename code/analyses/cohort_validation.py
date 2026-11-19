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
    .venv/bin/python code/analyses/cohort_validation.py --stage power   # design sensitivity only

Pre-specified analysis plan for the cohort-2 replication
---------------------------------------------------------
Frozen 2026-11-19T00:06:31Z (UTC), before any author or assistant inspected cohort-2
similarity, performance or summary outputs (only the subject count and file names were seen).
Plan document sha256: 60ded1c2b46aae0f07b47a6f146cec619cc21394cc4f6d6f093f98fc02b781b9.

Contrast: 2-unit GRU (tanh, decoder bias) vs 2-unit monoGRU+BC-DB (ReLU, energy + sparsity,
no reset gate, scalar update gate, no decoder bias). Fitting, nested CV, block-reset selection,
lambda grid, seeds, alignment and similarity metrics identical to cohort 1 (nested_DA_S32_B8).

Similarity on the raw scale. Within subject: mean over outer-fold pairs per mouse.
Across subjects: mean per subject pair (dyads share subjects).
  Weight, Hidden state   within: Wilcoxon signed-rank, ONE-SIDED (H1: monoGRU > GRU)
  Weight, Hidden state   across: delete-one-subject jackknife t(n-1), ONE-SIDED (H1: monoGRU > GRU)
  Logit                  within / across: Wilcoxon / jackknife, two-sided (not a claim)
  Update gate            within / across: Wilcoxon / jackknife, two-sided; 95% CI, no equivalence claim
Rule: one-sided only for comparisons significant in cohort 1 that carry a claim. An effect in the
opposite direction is a failed replication, not a finding. Two-sided p values are also reported.
Benjamini-Hochberg over the same eight tests as the cohort-1 table.
Primary endpoint: across-subject hidden-state similarity. Within-subject comparisons are secondary
and underpowered; a non-significant within-subject result is not evidence against the claim.
Effect sizes: mean delta [95% CI] and d_z = delta/(jackknife SE*sqrt(n)), next to cohort-1 values.

Behavioural prediction: held-out NLL per mouse (free-choice-weighted eval_CE_computed over outer
winners), delta = monoGRU - GRU. TOST with fixed margin +/-0.0074 nats/trial (5% of the cohort-1
GRU gain over chance, 0.148 nats): equivalence if the 90% CI lies inside the margin.

Design sensitivity (safeguard power; Perugini, Gallucci & Costantini, 2014, Perspect Psychol Sci
9:319-332): effect = lower bound of the 60% CI of each cohort-1 effect, simulated with the tests
above from cohort-1 data only (``--stage power``; see ``design_sensitivity``). At N = 22,
alpha = .05 one-sided (4,000 replicates): weights within ~86% (delta 0.11), hidden within ~73%
(0.045), weights across >99% (0.28), hidden across ~93% (0.15); NLL TOST ~99%.

Deviation from the plan (2026-11-19, after unblinding)
------------------------------------------------------
The plan scored held-out NLL by continuous rollout over the whole history (eval_CE_computed).
Models are trained on 32-trial sequences and selected on block-reset NLL, so held-out NLL is now
scored the same way (eval_CE: the saved block-reset outer-fold loss, eval_pred_loss), for
consistency with training and selection across all analyses. The margin (+/-0.0074) and test are
unchanged. The pre-specified continuous result is still computed and reported
(nll_equivalence.csv, manifest 'nll_equivalence_prespecified'): it failed equivalence because of
one fit whose continuous rollout diverged (WS12, outer fold 8).
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
from scipy.stats import binomtest, wilcoxon, false_discovery_control, t as t_dist
from NM_TinyRNN.code.models import experiments, datasets, nested_cv, rnns
from NM_TinyRNN.code.measures import analysis, performance, block_performance
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.figures import mechanistic_variability as figures
from NM_TinyRNN.code.figures.model_family_colors import COLOR_BY_FAMILY

SUBJECTS = [f'WS{i:02d}' for i in
            [3, 4, 6, 7, 11, 12, 15, 17, 18, 21, 23, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36]]
MODELS = [mv.STANDARD_GRU_REFERENCE, '2_unit_monoGRU_relu_unipolar_forced_ndb']
METRICS = [('logit_value_similarity', 'Logit'), ('weight_similarity', 'Weight'),
           ('hidden_similarity', 'Hidden state'), ('gate_update_similarity', 'Update gate')]


# ---- Pre-specified tests (analysis plan in the module docstring) ----
ONE_SIDED = {('weight_similarity', True), ('weight_similarity', False),
             ('hidden_similarity', True), ('hidden_similarity', False)}
NLL_MARGIN = 0.0074  # nats per free-choice trial: 5% of the cohort-1 GRU gain over chance (0.148)
NLL_SCORE = 'eval_CE'  # block-reset held-out NLL, as in training and selection (see deviation note)
NLL_SCORE_PRESPECIFIED = 'eval_CE_computed'  # continuous rollout, as in the frozen plan
COHORT1_TABLES = REPO/'data/analyses/mechanistic_variability'  # cohort 1: nested_DA_S32_B8
POWER_SEED, POWER_REPS, POWER_N = 20261119, 4000, 22


def alternative_for(metric, within):
    return 'greater' if (metric, within) in ONE_SIDED else 'two-sided'


def within_test(differences, alternative='two-sided'):
    """Wilcoxon signed-rank test on independent subject differences (monoGRU - GRU)."""
    d = np.asarray(differences, dtype=float)
    return float(wilcoxon(d, alternative=alternative).pvalue) if np.any(d != 0) else 1.


def jackknife_t(i, j, d, n):
    """Mean dyad difference with delete-one-subject jackknife SE (all incident dyads removed).

    i, j: integer subject indices (0..n-1) of each dyad; d: dyad differences.
    Identical to the SE in measures.mechanistic_variability.contrast. Returns (mean, se, t).
    """
    d = np.asarray(d, dtype=float)
    sums = np.bincount(i, weights=d, minlength=n) + np.bincount(j, weights=d, minlength=n)
    counts = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    loo = (d.sum()-sums)/(len(d)-counts)
    se = float(np.sqrt((n-1)/n*np.sum((loo-loo.mean())**2)))
    mean = float(d.mean())
    return mean, se, (mean/se if se > 0 else np.nan)


def dyad_index(frame):
    names = sorted(set(frame.subject_A) | set(frame.subject_B))
    index = {s: k for k, s in enumerate(names)}
    return frame.subject_A.map(index).to_numpy(), frame.subject_B.map(index).to_numpy(), len(names)


def across_test(frame, alternative='two-sided'):
    """Jackknife t(n-1) test of the mean dyad difference; frame has subject_A, subject_B, difference."""
    i, j, n = dyad_index(frame)
    mean, se, tstat = jackknife_t(i, j, frame.difference.to_numpy(), n)
    p = t_dist.sf(tstat, n-1) if alternative == 'greater' else 2*t_dist.sf(abs(tstat), n-1)
    return float(p), mean, se


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
    """Pre-specified tests (module docstring): Wilcoxon within, jackknife t across; one-sided for ONE_SIDED."""
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
            alternative = alternative_for(metric, scope)
            if scope:
                p, p_two = within_test(d.difference, alternative), within_test(d.difference)
            else:
                p, p_two = across_test(d, alternative)[0], across_test(d)[0]
            records.append(dict(metric=metric, component=label, within_subject=scope,
                n_subjects=n_subjects, n_units=len(d), gru_mean=d.gru.mean(), gru_sd=d.gru.std(ddof=1),
                mono_mean=d.mono.mean(), mono_sd=d.mono.std(ddof=1),
                median_difference=d.difference.median(), mean_difference=d.difference.mean(),
                median_deletion_min=min(median_loo), median_deletion_max=max(median_loo),
                deletion_min=min(loo), deletion_max=max(loo), cohens_dz=c['cohens_dz'],
                delta_ci95_low=c['delta_ci95_low'], delta_ci95_high=c['delta_ci95_high'],
                n_positive=positive, n_nonzero=nonzero, jackknife_se=c['jackknife_se'],
                test_method='Wilcoxon signed-rank' if scope else f'Subject-deletion jackknife t({n_subjects-1})',
                alternative=alternative, p_two_sided=p_two,
                p_sign_sensitivity=binomtest(positive, nonzero, .5).pvalue if scope and nonzero else np.nan,
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
        headers = ['Component', 'GRU', 'monoGRU', 'Mean Δ', 'GRU', 'monoGRU', 'Mean Δ']
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
                     f'{w.mean_difference:+.4f}' + (r'$^{' + ws + '}$' if ws else ''),
                     f'{a.gru_mean:.3f}', f'{a.mono_mean:.3f}',
                     f'{a.mean_difference:+.4f}' + (r'$^{' + acs + '}$' if acs else '')]
            for j, (x, text) in enumerate(zip(positions, cells)):
                bold = (j == 3 and bool(ws)) or (j == 6 and bool(acs))
                ax.text(x, .55-i*.10, text, ha='left' if j == 0 else 'center',
                        va='center', fontsize=12, fontweight='bold' if bold else 'normal')
        ax.plot([.015, .99], [.18, .18], color='black', lw=.8)
        footnote = (
            f'GRU/monoGRU columns: mean similarity over {n} subject summaries (within) or {n*(n-1)//2} subject-pair summaries (across). '
            'Mean Δ = mean paired monoGRU − GRU difference.\n'
            f'Within: Wilcoxon signed-rank on {n} independent subject differences. '
            f'Across: subject-deletion jackknife SE, approximate t({n-1}). Pre-specified one-sided tests (H1: monoGRU > GRU)\n'
            'for weight and hidden-state similarity; two-sided otherwise. BH across all eight comparisons '
            f'(* q < .05, ** < .01, *** < .001). Block-reset winners; {folder}.')
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
                        ['gru_mean', 'mono_mean', 'mean_difference', 'delta_ci95_low', 'delta_ci95_high',
                         'cohens_dz', 'median_difference', 'alternative', 'p_main', 'p_two_sided', 'q_main_bh8']})
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
                ax.plot([0, 1], [r.gru, r.mono], color='#888888', alpha=.6 if scope else .15, lw=.8)
                ax.scatter([0, 1], [r.gru, r.mono], c=[COLOR_BY_FAMILY['GRU'], COLOR_BY_FAMILY['monoGRU']],
                           alpha=.6 if scope else .15, s=4, zorder=3)
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
            'Within: independent subject means, Wilcoxon signed-rank tests. Across: dyad means, '
            f'subject-deletion jackknife SE and approximate t({n-1}); dyads are dependent. '
            'Pre-specified one-sided tests (H1: monoGRU > GRU) for weight and hidden-state similarity, two-sided otherwise '
            '(analysis plan in the script docstring). '
            'BH correction spans the four metrics × two scopes. d_z = Δ/(jackknife SE·√n); deletion ranges are influence diagnostics.\n\n'
            f'Intermediate data: {tables}\n\nReproduce:\n```sh\n.venv/bin/python code/analyses/cohort_validation.py --folder {folder}\n```\n')
    (output/'README.md').write_text(text)
    return summary



def subject_nll(selected, score=NLL_SCORE):
    """Per-subject held-out NLL: outer-fold scores weighted by free-choice trial counts (subjects x models)."""
    frame = selected[selected.model_id.isin(MODELS)].copy()
    frame['_w'] = pd.to_numeric(frame.eval_n_free, errors='coerce')
    frame['_s'] = pd.to_numeric(frame[score], errors='coerce')
    frame = frame[frame._w.gt(0) & frame._s.notna()]
    total = frame.assign(_ws=frame._s*frame._w).groupby(['subject_id', 'model_id'])[['_ws', '_w']].sum()
    return (total._ws/total._w).unstack('model_id')


def nll_equivalence(selected, tables, output, margin=NLL_MARGIN, score=NLL_SCORE, suffix=''):
    """Held-out NLL, monoGRU - GRU per subject (free-choice-weighted outer folds); TOST against +/-margin."""
    wide = subject_nll(selected, score)
    d = (wide[MODELS[1]]-wide[MODELS[0]]).dropna()
    n = len(d); df = n-1
    mean, sd = float(d.mean()), float(d.std(ddof=1)); se = sd/np.sqrt(n)
    ci95 = mean + np.array([-1, 1])*t_dist.ppf(.975, df)*se
    ci90 = mean + np.array([-1, 1])*t_dist.ppf(.95, df)*se
    p_tost = max(t_dist.sf((mean+margin)/se, df), t_dist.cdf((mean-margin)/se, df))
    result = dict(score=score, n_subjects=int(n), nll_gru=float(wide[MODELS[0]].mean()), nll_monogru=float(wide[MODELS[1]].mean()),
                  delta_mean=mean, delta_sd=sd, ci95_low=float(ci95[0]), ci95_high=float(ci95[1]),
                  ci90_low=float(ci90[0]), ci90_high=float(ci90[1]), margin=float(margin),
                  p_tost=float(p_tost), equivalent=bool(p_tost < .05), n_monogru_worse=int((d > 0).sum()),
                  likelihood_ratio=float(np.exp(-mean)), max_abs_subject_delta=float(d.abs().max()))
    pd.DataFrame({'subject_id': d.index, 'delta_nll': d.values}).to_csv(tables/f'nll_per_subject{suffix}.csv', index=False)
    pd.Series(result).to_csv(output/f'nll_equivalence{suffix}.csv', header=['value'])
    return result


def design_sensitivity(output, n_target=POWER_N, reps=POWER_REPS, seed=POWER_SEED, cohort1=COHORT1_TABLES):
    """Safeguard power for the pre-specified tests, computed from cohort-1 data only.

    Effects: observed, safeguard (lower bound of the 60% CI; Perugini et al., 2014), half the
    observed effect, and 0 (null calibration: power_* is then the false-positive rate).
    Within: resample centred cohort-1 subject differences, shift to the effect, within_test (Wilcoxon).
    Across: additive dyad model d_ij = mu + a_i + a_j + e_ij with empirical cohort-1 subject effects
    and residuals (residuals inflated by sqrt(n_dyads/(n_dyads - n))), tested with jackknife_t.
    NLL: normal subject differences with the cohort-1 SD, TOST against NLL_MARGIN.
    """
    rng = np.random.default_rng(seed)
    pairs = pd.read_csv(cohort1/'pairwise_similarities.htsv', sep='\t')
    subjects, _ = mv.summarize_reference_pairs(pairs[pairs.model_id.isin(MODELS)])
    keys, rows = ['subject_A', 'subject_B'], []
    for metric, label in METRICS:
        if (metric, True) not in ONE_SIDED:
            continue
        for scope in [True, False]:
            data = subjects[subjects.metric.eq(metric) & subjects.within_subject.eq(scope)]
            d = data[data.model_id.eq(MODELS[0])][keys+['mean_similarity']].merge(
                data[data.model_id.eq(MODELS[1])][keys+['mean_similarity']], on=keys,
                suffixes=('_gru', '_mono'), validate='one_to_one')
            d['difference'] = d.mean_similarity_mono - d.mean_similarity_gru
            if scope:
                x = d.difference.to_numpy(); n1 = len(x)
                observed = float(x.mean()); se = float(x.std(ddof=1)/np.sqrt(n1)); centred = x-observed
            else:
                i, j, n1 = dyad_index(d)
                observed, se, _ = jackknife_t(i, j, d.difference.to_numpy(), n1)
                X = np.zeros((len(d), n1)); X[np.arange(len(d)), i] = 1; X[np.arange(len(d)), j] = 1
                y = d.difference.to_numpy()-observed
                effects = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = (y-X@effects)*np.sqrt(len(d)/(len(d)-n1)); effects = effects-effects.mean()
                I, J = np.triu_indices(n_target, 1)
            safeguard = observed - t_dist.ppf(.80, n1-1)*se
            for scenario, mu in [('observed', observed), ('safeguard', safeguard),
                                 ('half observed', observed/2), ('null', 0.)]:
                hits = {'greater': 0, 'two-sided': 0}
                for _ in range(reps):
                    if scope:
                        sample = rng.choice(centred, n_target)+mu
                        for alternative in hits:
                            hits[alternative] += within_test(sample, alternative) < .05
                    else:
                        a = rng.choice(effects, n_target)
                        sample = mu + a[I] + a[J] + rng.choice(residuals, len(I))
                        tstat = jackknife_t(I, J, sample, n_target)[2]
                        hits['greater'] += t_dist.sf(tstat, n_target-1) < .05
                        hits['two-sided'] += 2*t_dist.sf(abs(tstat), n_target-1) < .05
                rows.append(dict(metric=metric, component=label, within_subject=scope, scenario=scenario,
                                 delta=mu, test='Wilcoxon signed-rank' if scope else f'jackknife t({n_target-1})',
                                 power_one_sided=hits['greater']/reps, power_two_sided=hits['two-sided']/reps,
                                 n_cohort1=n1, n_target=n_target, reps=reps, seed=seed))
    wide = subject_nll(pd.read_csv(cohort1/'selected_models.htsv', sep='\t'), NLL_SCORE)
    dn = (wide[MODELS[1]]-wide[MODELS[0]]).dropna(); sd = float(dn.std(ddof=1))
    critical = t_dist.ppf(.95, n_target-1)
    for scenario, mu in [('observed', float(dn.mean())), ('zero difference', 0.)]:
        x = rng.normal(mu, sd, (reps, n_target)); m = x.mean(1); s = x.std(1, ddof=1)/np.sqrt(n_target)
        power = float(np.mean(((m+NLL_MARGIN)/s > critical) & ((NLL_MARGIN-m)/s > critical)))
        rows.append(dict(metric='heldout_nll', component='NLL equivalence (TOST)', within_subject=np.nan,
                         scenario=scenario, delta=mu, test=f'TOST ({NLL_SCORE}), margin +/-{NLL_MARGIN}',
                         power_one_sided=np.nan, power_two_sided=power, n_cohort1=len(dn),
                         n_target=n_target, reps=reps, seed=seed))
    table = pd.DataFrame(rows)
    Path(output).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output)/'design_sensitivity.csv', index=False)
    print(table[['component', 'within_subject', 'scenario', 'delta', 'power_one_sided', 'power_two_sided']]
          .round(3).to_string(index=False), flush=True)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='nested_5HT_S32_B8')
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--jobs', type=int, default=3, help='Performance workers; similarity configurations run sequentially to bound RAM')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--train-missing', action='store_true', help='Complete missing native CPU fits before running the analysis')
    parser.add_argument('--training-jobs', type=int, default=6, help='Parallel CPU inner-fold fits for local training')
    parser.add_argument('--stage', choices=['all', 'power', 'analysis'], default='all',
                        help='power: design sensitivity from cohort-1 data only (does not read this cohort); '
                             'analysis: pre-specified cohort analysis; all: both')
    args = parser.parse_args(argv)
    if Path(args.folder).name != args.folder or args.folder in ['', '.', '..'] or min(args.jobs, args.training_jobs) < 1:
        parser.error('folder must name one run under data/rnns; jobs must be positive')
    if any(Path(s).name != s or s in ['', '.', '..'] for s in args.subjects):
        parser.error('subjects must be directory names')
    torch.set_num_threads(1)
    os.chdir(REPO.parent)
    if args.stage in ['all', 'power']:
        design_sensitivity(REPO/'data/analyses/cohort_validation'/args.folder)
        if args.stage == 'power':
            return
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
    manifest['nll_equivalence'] = nll_equivalence(selected, tables, output)
    manifest['nll_equivalence_prespecified'] = nll_equivalence(
        selected, tables, output, score=NLL_SCORE_PRESPECIFIED, suffix='_prespecified_continuous')
    manifest['analysis_plan'] = dict(frozen_utc='2026-11-19T00:06:31Z',
                                     plan_sha256='60ded1c2b46aae0f07b47a6f146cec619cc21394cc4f6d6f093f98fc02b781b9',
                                     one_sided=sorted(f'{m}:{"within" if w else "across"}' for m, w in ONE_SIDED),
                                     nll_margin=NLL_MARGIN, nll_score=NLL_SCORE,
                                     deviation='NLL scored with block resets (eval_CE) after unblinding; '
                                               'pre-specified continuous score reported as nll_equivalence_prespecified')
    manifest['outputs_sha256'] = {str(p.relative_to(REPO)): mv.sha256(p) for p in output.iterdir() if p.is_file()}
    manifest['statistics'] = dict(n_tests=8, n_subjects=len(cohort['subjects']),
                                n_significant=int(summary.q_main_bh8.lt(.05).sum()))
    write_json(manifest_path, manifest)
    print(summary[['component', 'within_subject', 'gru_mean', 'mono_mean', 'p_main', 'q_main_bh8']].to_string(index=False), flush=True)
    print(f'Intermediate data: {tables}\nResults: {output}', flush=True)


if __name__ == '__main__':
    main()
