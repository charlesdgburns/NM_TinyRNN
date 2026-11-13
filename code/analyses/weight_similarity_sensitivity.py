"""Reproduce pairwise weight-scaling sensitivity tables and empirical figures.

    .venv/bin/python code/analyses/weight_similarity_sensitivity.py
    .venv/bin/python code/analyses/weight_similarity_sensitivity.py --stage plots

Uses saved fits; does not train models, balance weights, or resample subjects.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import least_squares
from NM_TinyRNN.code.measures import mechanistic_variability as mv, performance
from NM_TinyRNN.code.measures.mechanistic_variability import CPUUnpickler

DEFAULT_RUN = 'nested_DA_last10_P100_S32_B8'
ENDPOINTS = ['subject_id', 'outer_loop_n', 'inner_loop_idx']
BLOCK_KEYS = ['model_id', 'architecture', 'activation', 'hidden_size', 'decoder_bias',
              'comparison', 'subject_A', 'subject_B']
SETTINGS = dict(selection='minimum combined train_val_CE; lower inner index breaks ties',
    grid=[.5, .875, 1., 1.25, 1.625, 2.], objective='core Euclidean distance',
    score='cosine over all trainable parameters',
    relu='grid winner then analytical-Jacobian TRF refinement; winning permutation fixed',
    tanh='permutations/signs without scaling versus continuous positive scaling in every sign/permutation branch, starting at unit scales',
    log_scale_bounds=[-20., 20.], ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300,
    orientation='A fixed, B aligned; endpoints sorted by subject then outer fold',
    no_balancing=True, no_bootstrap=True)


def _parameters(model):
    return {k: v.detach().cpu().numpy().astype(float).copy()
            for k, v in model.named_parameters() if v.requires_grad}

def _transform_parameters(params, perm, scales):
    """Match align_rnns and transform_gate_weights, including row-vector recurrence."""
    out = {k: v.copy() for k, v in params.items()}
    h = len(perm)
    p = np.eye(h)[list(perm)]
    pt = np.diag(scales) @ p
    inv = p.T @ np.diag(1 / np.asarray(scales))
    w = params['rnn.W_ih']
    out['rnn.W_ih'] = pt @ w if w.shape[0] == h else (pt @ w.T).T
    for key in ('rnn.bias_h', 'rnn.hidden_0'):
        if key in params:
            out[key] = pt @ params[key]
    out['rnn.W_hh'] = inv.T @ params['rnn.W_hh'] @ pt.T
    out['decoder.weight'] = params['decoder.weight'] @ inv
    for gate in ('z', 'r', 'i'):
        key = f'rnn.W_h{gate}'
        if key in params:
            w = params[key]
            if w.shape == (h, h):
                out[key] = inv.T @ w @ p.T
            elif w.shape == (h, 1):
                out[key] = inv.T @ w
        key = f'rnn.W_i{gate}'
        if key in params and params[key].shape[1] == h:
            out[key] = params[key] @ p.T
        key = f'rnn.bias_{gate}'
        if key in params and params[key].shape == (h,):
            out[key] = p @ params[key]
    return out

def _core_parameters(params, h):
    w = params['rnn.W_ih']
    if w.shape[0] != h:
        w = w.T
    return np.concatenate((w.ravel(), params.get('rnn.bias_h', np.zeros(h)).ravel(),
                           params['rnn.W_hh'].ravel(), params['decoder.weight'].ravel()))

def _scale_exponents(params, h):
    """Log-scale exponents for each core parameter; supplies exact Jacobian."""
    w = params['rnn.W_ih']
    inputs = w.shape[1] if w.shape[0] == h else w.shape[0]
    eye = np.eye(h)
    return np.concatenate((np.repeat(eye, inputs, axis=0), eye,
        np.array([eye[j]-eye[i] for i in range(h) for j in range(h)]),
        np.tile(-eye, (params['decoder.weight'].shape[0], 1))), axis=0)

def _parameter_cosine(a, b):
    va = np.concatenate([a[k].ravel() for k in a])
    vb = np.concatenate([b[k].ravel() for k in a])
    return float(va @ vb / (np.linalg.norm(va)*np.linalg.norm(vb)+1e-8))


def _tanh_scaling_pair(a, b):
    """Hypothetical tanh scaling in each permutation/sign branch; not a symmetry."""
    h = a['rnn.W_hh'].shape[0]
    target = _core_parameters(a, h)
    powers = _scale_exponents(b, h)
    best_base = (np.inf, None, None)
    best_scaled = (np.inf, None, None)
    failures = 0
    for perm in itertools.permutations(range(h)):
        for signs in itertools.product([1., -1.], repeat=h):
            signs = np.asarray(signs)
            coeff = _core_parameters(_transform_parameters(b, perm, signs), h)
            baseline_distance = float(np.sum((coeff - target)**2))
            if baseline_distance < best_base[0]:
                best_base = (baseline_distance, perm, signs)
            def residual(x):
                return coeff * np.exp(powers @ x) - target
            def jacobian(x):
                return (coeff * np.exp(powers @ x))[:, None] * powers
            result = least_squares(residual, np.zeros(h), method='trf', jac=jacobian,
                bounds=(-20., 20.), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
            failures += int(not result.success)
            distance = float(result.fun @ result.fun)
            scales = signs * np.exp(result.x)
            if distance > baseline_distance:
                distance, scales = baseline_distance, signs
            if distance < best_scaled[0]:
                best_scaled = (distance, perm, scales)
    baseline = _transform_parameters(b, best_base[1], best_base[2])
    scaled = _transform_parameters(b, best_scaled[1], best_scaled[2])
    return dict(baseline_similarity=_parameter_cosine(a, baseline),
        continuous_similarity=_parameter_cosine(a, scaled),
        baseline_distance=np.sqrt(best_base[0]), continuous_distance=np.sqrt(best_scaled[0]),
        optimizer_failed_branches=failures,
        scale_bound_hit=bool(np.any(np.abs(np.log(np.abs(best_scaled[2]))) > 19.99)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def write_table(frame, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, sep='\t', index=False)
    temp.replace(path)


def paths_for(folder, hidden_size):
    if Path(folder).name != folder or folder in ('', '.', '..'):
        raise ValueError('--folder must name one run under data/rnns')
    tables = REPO/'data/analysis/weight_similarity_sensitivity'
    figures = REPO/'results/figures'/f'weight_similarity_sensitivity_{folder}_h{hidden_size}'
    return tables, figures


def model_path(value, folder):
    path = Path(value)
    if not path.is_absolute():
        path = (REPO.parent/path) if path.parts[0] == REPO.name else REPO/path
    path = path.resolve()
    root = (REPO/'data/rnns'/folder).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Fit does not belong to requested run: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def select_models(frame, hidden_size, folds=10, subjects=None):
    required = {'model_id', 'model_type', 'hidden_size', 'nonlinearity', 'decoder_bias',
                'model_pickle_path', 'train_val_CE', *ENDPOINTS}
    if missing := required.difference(frame.columns):
        raise ValueError(f'Performance table missing {sorted(missing)}')
    frame = frame[frame.hidden_size.eq(hidden_size) & frame.nonlinearity.isin(['relu', 'tanh'])].copy()
    if frame.empty:
        raise ValueError('No matching ReLU/tanh fits')
    if frame.duplicated(['model_id', *ENDPOINTS]).any():
        raise ValueError('Duplicate inner-fit endpoints in performance table')
    if not np.isfinite(frame.train_val_CE).all():
        raise ValueError('Selection requires finite combined train+validation NLL values')
    frame = frame.sort_values(['model_id', 'subject_id', 'outer_loop_n', 'inner_loop_idx'])
    selected = performance.select_best_outer(frame).copy()
    selected['train_val_CE'] = frame.loc[selected.index, 'train_val_CE']
    selected = selected.sort_values(['model_id', 'subject_id', 'outer_loop_n']).reset_index(drop=True)
    for key, group in selected.groupby('model_id'):
        if subjects is not None and set(group.subject_id) != set(subjects):
            raise ValueError(f'Subject coverage differs from frozen run settings: {key}')
        for subject, g in group.groupby('subject_id'):
            if len(g) != folds or set(g.outer_loop_n) != set(range(1, folds + 1)):
                raise ValueError(f'Expected {folds} outer-fold winners for {key}, {subject}')
    configurations = selected[['model_type', 'decoder_bias', 'nonlinearity', 'model_id']].drop_duplicates()
    if configurations.duplicated(['model_type', 'decoder_bias', 'nonlinearity']).any():
        raise ValueError('Ambiguous matched variants: restrict the input to one configuration per architecture/activation/bias')
    if not configurations.groupby(['model_type', 'decoder_bias']).nonlinearity.nunique().eq(2).all():
        raise ValueError('Every architecture/bias condition needs both ReLU and tanh')
    return selected


def validate_pairs(pairs, selected):
    expected_ids = set(selected.model_id)
    if set(pairs.model_id) != expected_ids:
        raise ValueError('Pair cache model variants do not match selection')
    for key, group in selected.groupby('model_id'):
        group = group.sort_values(['subject_id', 'outer_loop_n']).reset_index(drop=True)
        p = pairs[pairs.model_id.eq(key)]
        expected = set(itertools.combinations(range(len(group)), 2))
        actual = list(p[['index_A', 'index_B']].itertuples(index=False, name=None))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f'Incomplete or duplicate fit pairs: {key}')
        for side in ['A', 'B']:
            index = p[f'index_{side}'].to_numpy(dtype=int)
            for source, target in [('subject_id', f'subject_{side}'), ('outer_loop_n', f'outer_{side}'), ('inner_loop_idx', f'inner_{side}')]:
                if not np.array_equal(p[target].to_numpy(), group.iloc[index][source].to_numpy()):
                    raise ValueError(f'Cached endpoint mismatch: {key}, {target}')
    if not np.isfinite(pairs[['baseline_similarity', 'scaled_similarity', 'baseline_distance', 'scaled_distance']]).all().all():
        raise ValueError('Nonfinite pair scores')
    if (pairs[['baseline_similarity', 'scaled_similarity']].abs() > 1 + 1e-6).any().any():
        raise ValueError('Cosine outside valid range')
    if (pairs.scaled_distance > pairs.baseline_distance + 1e-5 * np.maximum(pairs.baseline_distance, 1)).any():
        raise ValueError('Refinement increased alignment distance')


def compute_group(rows, tables, identity, recompute=False):
    torch.set_num_threads(1)
    rows = rows.sort_values(['subject_id', 'outer_loop_n']).reset_index(drop=True)
    key = rows.model_id.iloc[0]
    path, meta_path = tables/'groups'/f'{key}.htsv', tables/'groups'/f'{key}.json'
    if not recompute and path.is_file() and meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        if previous.get('fingerprint') == identity and previous.get('sha256') == sha256(path):
            validate_pairs(pd.read_csv(path, sep='\t'), rows)
            print(f'Reuse {key}', flush=True)
            return path
    models = []
    for row in rows.itertuples():
        with Path(row.model_pickle_path).open('rb') as handle:
            models.append(CPUUnpickler(handle).load().cpu().eval())
    activation = rows.nonlinearity.iloc[0]
    arrays = [_parameters(m) for m in models] if activation == 'tanh' else None
    records = []
    for count, (ia, ib) in enumerate(itertools.combinations(range(len(rows)), 2)):
        a, b = models[ia], models[ib]
        if activation == 'relu':
            grid, gi = mv.align_rnns(a, b, refine_scaling=False)
            refined, ci = mv.align_rnns(a, b, refine_scaling=True)
            baseline, scaled = mv.compute_weight_similarity(a, grid), mv.compute_weight_similarity(a, refined)
            bd, cd = gi['aligned_distance'], ci['aligned_distance']
            failed = int(not ci['refinement_success'])
            nfev = ci['refinement_nfev']
            bound = bool(np.any(np.abs(np.log(ci['transform_vec'])) > 19.99))
        else:
            result = _tanh_scaling_pair(arrays[ia], arrays[ib])
            baseline, scaled = result['baseline_similarity'], result['continuous_similarity']
            bd, cd = result['baseline_distance'], result['continuous_distance']
            failed, nfev, bound = result['optimizer_failed_branches'], np.nan, result['scale_bound_hit']
            if count < 2:
                aligned, info = mv.align_rnns(a, b)
                np.testing.assert_allclose(baseline, mv.compute_weight_similarity(a, aligned), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(bd, info['aligned_distance'], rtol=2e-5, atol=2e-6)
        ra, rb = rows.iloc[ia], rows.iloc[ib]
        records.append(dict(model_id=key, architecture=ra.model_type, activation=activation,
            hidden_size=int(ra.hidden_size), decoder_bias=bool(ra.decoder_bias), index_A=ia, index_B=ib,
            subject_A=ra.subject_id, subject_B=rb.subject_id, outer_A=int(ra.outer_loop_n), outer_B=int(rb.outer_loop_n),
            inner_A=int(ra.inner_loop_idx), inner_B=int(rb.inner_loop_idx),
            comparison='within_subject' if ra.subject_id == rb.subject_id else 'across_subject',
            baseline_similarity=baseline, scaled_similarity=scaled, delta_similarity=scaled-baseline,
            baseline_distance=bd, scaled_distance=cd, failed_branches=failed, refinement_nfev=nfev, scale_bound_hit=bound))
    result = pd.DataFrame(records)
    validate_pairs(result, rows)
    write_table(result, path)
    write_json(meta_path, dict(fingerprint=identity, sha256=sha256(path), pairs=len(result),
        failed_branches=int(result.failed_branches.sum()), scale_bound_hits=int(result.scale_bound_hit.sum())))
    print(f'Computed {key}: {len(result):,} pairs', flush=True)
    return path


def compute(folder, hidden_size, tables, jobs=6, recompute=False, performance_table=None):
    config_path = REPO/'data/rnns'/folder/'_aws/run-config.json'
    config = json.loads(config_path.read_text())
    source = Path(performance_table).resolve() if performance_table else REPO/'data/analysis/performance/performance_df.htsv'
    if not source.is_file():
        raise FileNotFoundError(f'Performance table missing: {source}. Build it with analysis.get_analysis_df and performance.get_performance_df, or supply --performance-table.')
    frame = pd.read_csv(source, sep='\t')
    selected = select_models(frame, hidden_size, config.get('n_folds', 10), config.get('subjects'))
    selected['source_model_pickle_path'] = selected.model_pickle_path
    selected['model_pickle_path'] = [str(model_path(p, folder)) for p in selected.model_pickle_path]
    selected['model_sha256'] = [sha256(p) for p in selected.model_pickle_path]
    write_table(selected, tables/'selected_models.htsv')
    code_hashes = {str(Path(p).resolve()): sha256(p) for p in [__file__, mv.__file__]}
    groups = list(selected.groupby('model_id'))
    identities = {key: fingerprint(dict(settings=SETTINGS, code=code_hashes,
        fits=g[['model_id', *ENDPOINTS, 'model_pickle_path', 'model_sha256']].to_dict('records'))) for key, g in groups}
    with parallel_config(backend='loky', inner_max_num_threads=1):
        paths = Parallel(n_jobs=jobs)(delayed(compute_group)(g.copy(), tables, identities[key], recompute) for key, g in groups)
    pairs = pd.concat([pd.read_csv(path, sep='\t') for path in paths], ignore_index=True)
    validate_pairs(pairs, selected)
    write_table(pairs, tables/'pairwise_scaling.htsv')
    manifest = dict(study='weight_similarity_sensitivity', folder=folder, hidden_size=hidden_size, settings=SETTINGS,
        source_performance=str(source), performance_sha256=sha256(source), config_sha256=sha256(config_path),
        selected_sha256=sha256(tables/'selected_models.htsv'), pairs_sha256=sha256(tables/'pairwise_scaling.htsv'),
        code_sha256=code_hashes, group_fingerprints=identities, fits=len(selected), pairs=len(pairs),
        failed_branches=int(pairs.failed_branches.sum()), scale_bound_hits=int(pairs.scale_bound_hit.sum()))
    write_json(tables/'manifest.json', manifest)
    return selected, pairs, manifest


def load_computation(tables):
    manifest = json.loads((tables/'manifest.json').read_text())
    for name, field in [('selected_models.htsv', 'selected_sha256'), ('pairwise_scaling.htsv', 'pairs_sha256')]:
        if sha256(tables/name) != manifest[field]:
            raise ValueError(f'Cached computation was changed: {name}. Rerun --stage compute.')
    selected, pairs = (pd.read_csv(tables/name, sep='\t') for name in ['selected_models.htsv', 'pairwise_scaling.htsv'])
    validate_pairs(pairs, selected)
    return selected, pairs, manifest


def aggregate(pairs):
    scores = ['baseline_similarity', 'scaled_similarity', 'delta_similarity']
    blocks = pairs.groupby(BLOCK_KEYS, as_index=False)[scores].mean()
    blocks['fit_pairs'] = pairs.groupby(BLOCK_KEYS).size().to_numpy()
    blocks['percent_change'] = 100 * blocks.delta_similarity / blocks.baseline_similarity.abs().where(blocks.baseline_similarity.abs() > 1e-8)
    meta = ['model_id', 'architecture', 'activation', 'hidden_size', 'decoder_bias', 'comparison']
    summary = blocks.groupby(meta, as_index=False)[scores].mean()
    summary['percent_change'] = 100 * summary.delta_similarity / summary.baseline_similarity.abs().where(summary.baseline_similarity.abs() > 1e-8)
    ranges = summary.groupby(['activation', 'comparison'], as_index=False).percent_change.agg(min_percent_change='min', max_percent_change='max')
    contrasts = {}
    keys = ['architecture', 'hidden_size', 'decoder_bias', 'comparison', 'subject_A', 'subject_B']
    for mode in ['discrete', 'continuous']:
        relu_column = 'baseline_similarity' if mode == 'discrete' else 'scaled_similarity'
        relu = blocks[blocks.activation.eq('relu')][keys + [relu_column]].rename(columns={relu_column: 'relu_similarity'})
        tanh = blocks[blocks.activation.eq('tanh')][keys + ['baseline_similarity']].rename(columns={'baseline_similarity': 'tanh_similarity'})
        matched = relu.merge(tanh, on=keys, validate='one_to_one', how='outer', indicator=True)
        if not matched['_merge'].eq('both').all():
            raise ValueError('Unmatched subject/dyad blocks between activations')
        matched = matched.drop(columns='_merge')
        matched['difference'] = matched.relu_similarity - matched.tanh_similarity
        contrasts[mode] = matched
    return blocks, summary, ranges, contrasts


def report(pairs, tables, manifest):
    blocks, summary, ranges, contrasts = aggregate(pairs)
    for frame, name in [(blocks, 'subject_pair_scaling'), (summary, 'per_variant_summary'), (ranges, 'scaling_ranges')]:
        write_table(frame, tables/f'{name}.htsv')
    for mode, frame in contrasts.items():
        write_table(frame, tables/f'activation_contrasts_{mode}.htsv')
    lines = ['# Weight similarity sensitivity to pairwise scaling', '',
        'All comparisons use selected raw fitted weights. No minimum-norm balancing or subject bootstrap is applied.', '',
        '| Comparison | Within subjects | Across subjects |', '|---|---:|---:|']
    for activation, label in [('relu', 'ReLU: discrete → continuous scaling'), ('tanh', 'tanh: unscaled → hypothetical continuous scaling')]:
        values = []
        for scope in ['within_subject', 'across_subject']:
            r = ranges[ranges.activation.eq(activation) & ranges.comparison.eq(scope)].iloc[0]
            values.append(f'{r.min_percent_change:+.2f}–{r.max_percent_change:+.2f}%')
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    lines += ['', 'Ranges describe percentage changes in variant means, not uncertainty intervals. Each variant mean equally averages subjects or unordered subject dyads. Percent change = 100 × (scaled mean − baseline mean) / |baseline mean|. Individual near-zero baseline scores make percentage changes unstable; figures show absolute cosine differences.', '',
        'ReLU: exhaustive permutation/grid search using {0.5, 0.875, 1, 1.25, 1.625, 2}, then TRF least-squares refinement from the winning scales with its permutation fixed. Tanh: exhaustive permutation/sign baseline; the hypothetical scaled condition refines positive scales in every sign/permutation branch, starting at unit scales. Tanh scaling changes function. Both use the core Euclidean alignment objective; cosine includes all trainable parameters, so scaling can lower the reported cosine.', '',
        'Both continuous procedures use analytical Jacobians, log-scale bounds [-20,20], ftol/xtol/gtol 1e-10 and at most 300 residual evaluations. A successful ReLU refinement is retained only if it improves the grid objective. The hypothetical tanh helper retains the baseline if a candidate does not improve the distance. Neither local search proves a global continuous optimum.', '',
        'The combined discrete figure reproduces the previous standard-method contrast. The continuous figure instead uses grid-refined ReLU scores in its lower panels. Both keep tanh unscaled in the activation contrast; tanh scaling appears only in the sensitivity panels.', '',
        f"Selected fits: {manifest['fits']:,}; fit pairs: {manifest['pairs']:,}; failed optimisation branches: {manifest['failed_branches']}; selected scales at numerical bounds: {manifest['scale_bound_hits']}.", '',
        '## Per-variant changes', '', '| Variant | Scope | Baseline | With continuous scaling | Δ cosine | % change |', '|---|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        label = r.architecture + ('+BC' if r.activation == 'relu' else '') + ('-DB' if not r.decoder_bias else '')
        lines.append(f'| {label} | {r.comparison} | {r.baseline_similarity:.5f} | {r.scaled_similarity:.5f} | {r.delta_similarity:+.5f} | {r.percent_change:+.2f}% |')
    reports = REPO/'results/appendix/weight_similarity_sensitivity'
    reports.mkdir(parents=True, exist_ok=True)
    (reports/'REPORT.md').write_text('\n'.join(lines) + '\n')
    return blocks, contrasts


def plots(pairs, figures, hidden_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    blocks, _, _, contrasts = aggregate(pairs)
    configs = sorted(set(zip(blocks.architecture, blocks.decoder_bias)), key=lambda x: (['GRU', 'lightGRU', 'monoGRU'].index(x[0]) if x[0] in ['GRU', 'lightGRU', 'monoGRU'] else 99, x[0], not x[1]))
    labels = [a + ('-DB' if not b else '') for a, b in configs]
    rng = np.random.default_rng(42)
    scopes = [('within_subject', 'Within subject'), ('across_subject', 'Across subjects')]
    def panel(ax, data, column, scope, title, color, limit=None):
        for y, (architecture, bias) in enumerate(configs):
            g = data[data.architecture.eq(architecture) & data.decoder_bias.eq(bias) & data.comparison.eq(scope)]
            if g.empty:
                raise ValueError('Empty empirical panel')
            ax.scatter(g[column], y + rng.uniform(-.18, .18, len(g)), s=22, c=color, alpha=.6, edgecolors='none', zorder=3)
            ax.scatter(g[column].mean(), y, marker='D', s=55, c='black', edgecolors='white', linewidths=.6, zorder=4)
        for y in np.arange(1.5, len(configs)-1, 2):
            ax.axhline(y, c='#e1e1e1', lw=.7)
        ax.axvline(0, c='#555555', lw=.9)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_yticks(range(len(configs)), labels)
        ax.set_ylim(len(configs)-.5, -.5)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', labelbottom=True)
        if limit is not None:
            ax.set_xlim(-limit, limit)
            ax.set_xticks(np.arange(-np.floor(limit/.2)*.2, limit + 1e-8, .2))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5))
    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='#777777', markeredgecolor='none', label='Subject / subject-pair difference'),
               Line2D([], [], marker='D', linestyle='none', c='black', label='Mean difference')]
    def save(fig, stem):
        for ext in ['png', 'svg']:
            fig.savefig(figures/f'{stem}.{ext}', dpi=220)
        plt.close(fig)
    tanh = blocks[blocks.activation.eq('tanh')]
    for activation in ['relu', 'tanh']:
        data = blocks[blocks.activation.eq(activation)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
        for ax, (scope, title) in zip(axes, scopes):
            panel(ax, data, 'delta_similarity', scope, title, '#477b99')
            ax.set_xlabel('Change in weight cosine similarity')
        fig.suptitle('Effect of adding scaling to tanh weight similarity' if activation == 'tanh' else 'ReLU: discrete → continuous pairwise scaling', fontsize=14, y=.99)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .92), ncol=2, frameon=False)
        fig.text(.55, .03, f'{hidden_size}-unit models; -DB = no decoder bias. Dots: empirical subject/dyad means; diamonds: means.\nTanh scaling changes function; both conditions retain permutations and tanh sign flips.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .1, 1, .83])
        save(fig, f'{activation}_scaling_sensitivity')
    for mode, contrast in contrasts.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
        for ax, (scope, title) in zip(axes, scopes):
            panel(ax, contrast, 'difference', scope, title, '#775a91')
            ax.set_xlabel('Cosine difference: ReLU − tanh')
        fig.suptitle(f'ReLU (+BC) versus tanh · {mode} ReLU scaling', fontsize=14, y=.99)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .92), ncol=2, frameon=False)
        fig.text(.55, .03, 'Original pairwise Euclidean alignment; no balancing. Tanh unscaled. -DB = no decoder bias.\nAcross-subject points share subjects and are not independent.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .1, 1, .83])
        save(fig, f'activation_contrasts_{mode}')
        values = np.r_[tanh.delta_similarity.to_numpy(), contrast.difference.to_numpy()]
        limit = max(.2, np.ceil((np.max(np.abs(values)) + .035)*10)/10)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True,
            gridspec_kw=dict(left=.16, right=.985, bottom=.15, top=.87, hspace=.72, wspace=.1))
        for row, (data, column, color) in enumerate([(tanh, 'delta_similarity', '#477b99'), (contrast, 'difference', '#775a91')]):
            for col, (scope, title) in enumerate(scopes):
                ax = axes[row, col]
                panel(ax, data, column, scope, title, color, limit)
                ax.set_xlabel('Cosine difference: scaling − no scaling' if row == 0 else 'Cosine difference: ReLU − tanh')
                ax.text(-.065, 1.08, 'ABCD'[row*2+col], transform=ax.transAxes, fontweight='bold', fontsize=12)
        fig.text(.57, .975, 'Effect of adding scaling to tanh weight similarity', ha='center', va='top', fontweight='bold', fontsize=16)
        fig.text(.57, .49, 'Weight similarity: ReLU versus tanh', ha='center', fontweight='bold', fontsize=16)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .948), ncol=2, frameon=False)
        fig.text(.57, .035, f'Top: tanh without versus with continuous scaling. Bottom: {mode} ReLU scaling versus unscaled tanh.\n{hidden_size}-unit models; -DB = no decoder bias. All panels share x limits and ticks. No balancing or bootstrap.\nDots: subject/dyad means; diamonds: overall means. Subject pairs are dependent.', ha='center', fontsize=9, va='bottom')
        save(fig, f'combined_{mode}')
    print(f'Figures: {figures}', flush=True)


def export_appendix(pairs, figures, include_figure=True):
    """Publish the default study's requested manuscript filenames."""
    appendix = REPO/'results/appendix'
    appendix.mkdir(parents=True, exist_ok=True)
    _, _, ranges, _ = aggregate(pairs)
    rows = []
    for activation, label in [('relu', 'ReLU: discrete → continuous scaling'),
                              ('tanh', 'tanh: unscaled → hypothetical continuous scaling')]:
        row = [label]
        for scope in ['within_subject', 'across_subject']:
            result = ranges[ranges.activation.eq(activation) & ranges.comparison.eq(scope)].iloc[0]
            row.append(f'{result.min_percent_change:+.2f}–{result.max_percent_change:+.2f}%')
        rows.append(row)
    table = appendix/'table_weight_sensitivity.csv'
    temporary = table.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Comparison', 'Within subjects', 'Across subjects'])
        writer.writerows(rows)
    temporary.replace(table)
    if include_figure:
        figure = appendix/'figure_weight_sensitivity.png'
        temporary = figure.with_suffix('.png.tmp')
        shutil.copyfile(figures/'combined_continuous.png', temporary)
        temporary.replace(figure)
    print(f'Appendix exports: {appendix}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=DEFAULT_RUN)
    parser.add_argument('--hidden-size', type=int, choices=[1, 2], default=2)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--stage', choices=['all', 'compute', 'report', 'plots'], default='all')
    parser.add_argument('--performance-table', type=Path)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.recompute and args.stage not in ['all', 'compute']:
        parser.error('--recompute requires --stage all or compute')
    torch.set_num_threads(1)
    tables, figures = paths_for(args.folder, args.hidden_size)
    tables.mkdir(parents=True, exist_ok=True)
    (tables/'groups').mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if args.stage in ['all', 'compute']:
        _, pairs, manifest = compute(args.folder, args.hidden_size, tables, args.jobs, args.recompute, args.performance_table)
    else:
        _, pairs, manifest = load_computation(tables)
        if manifest.get('folder', args.folder) != args.folder or manifest.get('hidden_size', args.hidden_size) != args.hidden_size:
            raise ValueError('Cached analysis uses a different training run or hidden size; rerun --stage compute.')
    if args.stage in ['all', 'report']:
        report(pairs, tables, manifest)
    if args.stage in ['all', 'plots']:
        plots(pairs, figures, args.hidden_size)
    if args.folder == DEFAULT_RUN and args.hidden_size == 2 and args.stage in ['all', 'report', 'plots']:
        export_appendix(pairs, figures, include_figure=args.stage in ['all', 'plots'])
    print(f'Tables and provenance: {tables}', flush=True)


if __name__ == '__main__':
    main()
