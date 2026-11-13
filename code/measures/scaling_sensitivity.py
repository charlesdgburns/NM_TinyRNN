"""Paired scaling sensitivity of train+validation-selected outer-fold fits.

Run from any directory with the repository's Python environment. Existing
alignment code and source fit files are left unchanged.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import torch
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.mechanistic_variability import CPUUnpickler


def parameters(model):
    return {k: v.detach().cpu().numpy().astype(float).copy()
            for k, v in model.named_parameters() if v.requires_grad}


def transform(params, perm, scales):
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


def core(params, h):
    w = params['rnn.W_ih']
    if w.shape[0] != h:
        w = w.T
    return np.concatenate((w.ravel(), params.get('rnn.bias_h', np.zeros(h)).ravel(),
                           params['rnn.W_hh'].ravel(), params['decoder.weight'].ravel()))


def exponents(params, h):
    """Log-scale exponents for each core parameter; supplies exact Jacobian."""
    w = params['rnn.W_ih']
    inputs = w.shape[1] if w.shape[0] == h else w.shape[0]
    eye = np.eye(h)
    return np.concatenate((np.repeat(eye, inputs, axis=0), eye,
        np.array([eye[j]-eye[i] for i in range(h) for j in range(h)]),
        np.tile(-eye, (params['decoder.weight'].shape[0], 1))), axis=0)


def cosine(a, b):
    va = np.concatenate([a[k].ravel() for k in a])
    vb = np.concatenate([b[k].ravel() for k in a])
    return float(va @ vb / (np.linalg.norm(va)*np.linalg.norm(vb)+1e-8))


def fit_pair(a, b, nonlinearity):
    h = a['rnn.W_hh'].shape[0]
    target = core(a, h)
    powers = exponents(b, h)
    grid = np.unique(np.r_[np.linspace(.5, 2., 5), 1.])
    transforms = list(itertools.product(grid if nonlinearity == 'relu' else [1., -1.], repeat=h))
    best_base = (np.inf, None, None)
    best_cont = (np.inf, None, None, None)
    failures = 0
    # Refine the best discrete candidate within EVERY permutation/sign branch.
    for perm in itertools.permutations(range(h)):
        signs_list = [np.ones(h)] if nonlinearity == 'relu' else transforms
        for signs in signs_list:
            signs = np.asarray(signs)
            coeff = core(transform(b, perm, signs), h)
            starts = np.array(transforms) if nonlinearity == 'relu' else np.ones((1, h))
            vals = coeff[None, :] * np.exp(np.log(starts) @ powers.T)
            distances = np.sum((vals-target)**2, axis=1)
            idx = int(np.argmin(distances))
            initial = starts[idx]
            dist = float(distances[idx])
            scale = signs * initial
            if dist < best_base[0]:
                best_base = (dist, perm, scale)
            def residual(x):
                return coeff*np.exp(powers @ x)-target
            def jacobian(x):
                return (coeff*np.exp(powers @ x))[:, None]*powers
            result = least_squares(residual, np.log(initial), jac=jacobian,
                bounds=(-20., 20.), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)
            failures += int(not result.success)
            refined_dist = float(result.fun @ result.fun)
            # Retain the baseline if numerical optimization fails to improve it.
            if refined_dist > dist:
                refined_dist, refined_scale = dist, scale
            else:
                refined_scale = signs*np.exp(result.x)
            if refined_dist < best_cont[0]:
                best_cont = (refined_dist, perm, refined_scale, result)
    base = transform(b, best_base[1], best_base[2])
    cont = transform(b, best_cont[1], best_cont[2])
    sb, sc = cosine(a, base), cosine(a, cont)
    return dict(baseline_similarity=sb, continuous_similarity=sc, delta_similarity=sc-sb,
        percent_change=100*(sc-sb)/abs(sb) if abs(sb) > 1e-8 else np.nan,
        baseline_distance=np.sqrt(best_base[0]), continuous_distance=np.sqrt(best_cont[0]),
        distance_reduction_pct=100*(1-np.sqrt(best_cont[0]/best_base[0])) if best_base[0] > 0 else 0.,
        baseline_perm=json.dumps(list(best_base[1])), continuous_perm=json.dumps(list(best_cont[1])),
        baseline_scales=json.dumps(best_base[2].tolist()), continuous_scales=json.dumps(best_cont[2].tolist()),
        optimizer_failed_branches=failures,
        scale_bound_hit=bool(np.any(np.abs(np.log(np.abs(best_cont[2]))) > 19.99)))


def group_analysis(rows, out):
    torch.set_num_threads(1)
    model_id = rows.model_id.iloc[0]
    rows = rows.sort_values(['subject_id', 'outer_loop_n']).reset_index(drop=True)
    models = []
    for row in rows.itertuples():
        path = Path(row.model_pickle_path)
        if not path.is_absolute():
            path = REPO.parent/path
        with path.open('rb') as f:
            models.append(CPUUnpickler(f).load().cpu().eval())
    arrays = [parameters(m) for m in models]
    records = []
    for ia, ib in itertools.combinations(range(len(rows)), 2):
        ra, rb = rows.iloc[ia], rows.iloc[ib]
        result = fit_pair(arrays[ia], arrays[ib], ra.nonlinearity)
        # Validate the accelerated baseline against the existing implementation.
        if len(records) < 2:
            aligned, info = mv.align_rnns(models[ia], models[ib])
            np.testing.assert_allclose(result['baseline_distance'], info['aligned_distance'], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(result['baseline_similarity'], mv.compute_weight_similarity(models[ia], aligned), rtol=2e-5, atol=2e-6)
        records.append(dict(model_id=model_id, nonlinearity=ra.nonlinearity, hidden_size=int(ra.hidden_size),
            subject_A=ra.subject_id, outer_A=int(ra.outer_loop_n), inner_A=int(ra.inner_loop_idx),
            subject_B=rb.subject_id, outer_B=int(rb.outer_loop_n), inner_B=int(rb.inner_loop_idx),
            comparison='within_subject' if ra.subject_id == rb.subject_id else 'across_subject', **result))
    frame = pd.DataFrame(records)
    frame.to_csv(out/'groups'/f'{model_id}.tsv', sep='\t', index=False)
    print(f'Completed {model_id}: {len(frame)} pairs', flush=True)
    return frame


def summarize(pairs):
    results = []
    for (model_id, comparison), g in pairs.groupby(['model_id', 'comparison']):
        # Subject or unordered subject-pair averages avoid treating folds as independent subjects.
        blocks = g.groupby(['subject_A', 'subject_B'])[['baseline_similarity', 'continuous_similarity', 'delta_similarity']].mean()
        delta = blocks.delta_similarity
        base, cont = blocks.baseline_similarity.mean(), blocks.continuous_similarity.mean()
        results.append(dict(model_id=model_id, comparison=comparison, n_pairs=len(g), n_subject_blocks=len(blocks),
            baseline_similarity=base, continuous_similarity=cont, delta_similarity=cont-base,
            percent_change=100*(cont-base)/abs(base) if abs(base)>1e-8 else np.nan,
            paired_dz_subject_blocks=delta.mean()/delta.std(ddof=1) if delta.std(ddof=1)>0 else np.nan,
            median_pair_delta=g.delta_similarity.median(), min_pair_delta=g.delta_similarity.min(),
            max_pair_delta=g.delta_similarity.max(), fraction_pairs_increased=float((g.delta_similarity>1e-6).mean()),
            mean_distance_reduction_pct=g.distance_reduction_pct.mean(),
            optimizer_failed_branches=int(g.optimizer_failed_branches.sum()), scale_bound_hits=int(g.scale_bound_hit.sum())))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--model-id', default=None)
    args = parser.parse_args()
    out = REPO/'data/analysis/scaling_sensitivity_nested_DA_last10_P100_S32_B8'
    (out/'groups').mkdir(parents=True, exist_ok=True)
    source = REPO/'data/analysis/performance_df_nested_DA_last10_P100_S32_B8_final.htsv'
    frame = pd.read_csv(source, sep='\t')
    keys = ['model_id', 'subject_id', 'outer_loop_n']
    # Same criterion as performance.select_best_outer, with explicit stable tie-breaking.
    selected = frame.sort_values(keys+['train_val_CE', 'inner_loop_idx']).groupby(keys, sort=False).head(1).copy()
    assert selected.groupby(['subject_id', 'model_id']).size().eq(10).all()
    assert selected.groupby(['subject_id', 'model_id']).outer_loop_n.nunique().eq(10).all()
    selected.to_csv(out/'selected_models.tsv', sep='\t', index=False)
    if args.model_id:
        selected = selected[selected.model_id == args.model_id]
    groups = list(selected.groupby('model_id'))
    started = time.time()
    results = Parallel(n_jobs=args.jobs)(delayed(group_analysis)(g, out) for _, g in groups)
    pairs = pd.concat(results, ignore_index=True)
    pairs.to_csv(out/'pairwise_results.tsv', sep='\t', index=False)
    summary = summarize(pairs)
    summary.to_csv(out/'summary.tsv', sep='\t', index=False)
    manifest = dict(source=str(source), criterion='minimum combined training+validation NLL (train_val_CE); inner index breaks ties',
        selected_fits=len(selected), pairs=len(pairs), elapsed_seconds=time.time()-started,
        baseline='existing five-step ReLU grid plus 1; tanh signs; all permutations',
        continuous='positive log-scale least squares, refining every permutation/sign branch from its best baseline candidate',
        objective='existing core Euclidean parameter distance; cosine over all trainable parameters',
        scale_bounds=[float(np.exp(-20)), float(np.exp(20))],
        percent_change='100*(continuous-baseline)/abs(baseline); ratio of block-mean similarities in summary',
        effect_size='paired dz of subject means (within) or unordered subject-pair means (across); descriptive only, blocks across share subjects',
        orientation='subject then outer fold; model B aligned to A; not symmetrized',
        tanh='positive scaling added to existing sign flips and permutations; not function preserving')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
