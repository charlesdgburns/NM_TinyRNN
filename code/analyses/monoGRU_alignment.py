"""Balanced, semantically labeled monoGRU references and aligned model copies.

Default: the 110 block-reset train+validation-selected two-unit ReLU monoGRUs without decoder bias
from nested_DA_S32_B8. No training, behavior-based reference selection, or source
checkpoint edits. See MONOGRU_ALIGNMENT.md for the coordinate convention.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.monogru_balancing import canonicalize, power_matrix, parameters, transform, core
from NM_TinyRNN.code.models.datasets import input_encoder

LABEL_RULE = 'h1 has the larger decoder[1]-decoder[0] contrast; opponent models: h1 favors choice 1, h2 favors choice 0'


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT.parent/path


def load_model(path):
    with resolve(path).open('rb') as handle:
        model = mv.CPUUnpickler(handle).load().cpu().float().eval()
    validate_model(model)
    return model


def validate_model(model):
    if (model.rnn_type, model.H, model.O, model.nonlinearity) != ('monoGRU', 2, 2, 'relu'):
        raise ValueError('Only two-unit, two-output monoGRU ReLU models are supported')
    if model.decoder_bias or model.decoder.bias is not None:
        raise ValueError('monoGRU +BC-DB requires decoder bias off')
    if model.rnn.mode != 'default' or model.batch_norm:
        raise ValueError('Expected default scalar gate without batch normalization')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite source parameters')


def write_parameters(model, values):
    result = deepcopy(model)
    with torch.no_grad():
        for key, parameter in result.named_parameters():
            parameter.copy_(torch.as_tensor(values[key], dtype=parameter.dtype))
    return result


def balance_model(model):
    """Minimize full trainable-parameter squared norm along the ReLU scale orbit.

    Reuses the existing convex log-scale canonicalizer. Unanchored units have
    no guaranteed finite canonical scale: use a bounded approximation, flag it,
    and exclude the model from reference candidacy (but keep its cohort weight).
    """
    validate_model(model)
    original = parameters(model)
    try:
        values, info = canonicalize(original, 'relu')
        info['balance_status'] = 'finite_minimum'
    except ValueError as error:
        if 'Unanchored unit' not in str(error):
            raise
        powers = power_matrix({k: np.ones_like(v) for k, v in original.items()})
        vector = np.concatenate([v.ravel() for v in original.values()])
        denominator = max(float(vector@vector), 1e-30)
        def objective(x):
            square = vector**2*np.exp(2*powers@x)/denominator
            return float(square.sum()), 2*square@powers
        fit = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
                       bounds=[(-8., 8.)]*2, options=dict(ftol=1e-14, gtol=1e-10, maxiter=1000))
        if not fit.success:
            raise RuntimeError(f'Degenerate balancing failed: {fit.message}')
        values = transform(original, (0, 1), np.exp(fit.x))
        info = dict(log_scale_1=float(fit.x[0]), log_scale_2=float(fit.x[1]),
                    balance_status='unanchored_bounded', bound=8., anchored=False,
                    norm_before=float(np.linalg.norm(vector)),
                    norm_after=float(np.linalg.norm(np.concatenate([v.ravel() for v in values.values()]))))
    return write_parameters(model, values), info


def label_permutation(model):
    contrast = model.decoder.weight.detach().numpy()[1]-model.decoder.weight.detach().numpy()[0]
    return tuple(np.argsort(-contrast, kind='stable').tolist())


def align_balanced_model(model, reference):
    """Use the existing alignment engine with scale fixed and semantic order hard.

    With two labeled units there is exactly one permitted permutation. No
    reference-specific rescaling follows independent balancing.
    """
    return mv.align_rnns(reference, model, nonlinearity='relu', scale_values=np.array([1.]),
                         refine_scaling=False, allowed_permutations=[label_permutation(model)])


def align_model(model, reference):
    """Public notebook API: balance a new model then align it to a frozen reference."""
    balanced, balancing = balance_model(model)
    aligned, alignment = align_balanced_model(balanced, reference)
    return aligned, dict(balancing=balancing, alignment=alignment)


def probes(model):
    """Fixed synthetic histories; no observed trials, NLL, or outer-test targets."""
    generator = torch.Generator().manual_seed(20261117)
    raw = torch.randint(0, 2, (24, 64, 3), generator=generator).float()
    constants = torch.tensor([[f, a, o] for f in (0, 1) for a in (0, 1) for o in (0, 1)]).float()
    raw = torch.cat([raw, constants[:, None, :].repeat(1, 64, 1)])
    return input_encoder(raw, model.input_encoding, model.input_forced_choice)


def check_equivalence(original, aligned, inputs, perm, scales):
    with torch.no_grad():
        logits_a, hidden_a = original(inputs)
        logits_b, hidden_b = aligned(inputs)
        _, gates_a = original.rnn(inputs, return_gate_activations=True)
        _, gates_b = aligned.rnn(inputs, return_gate_activations=True)
    expected = hidden_a[..., list(perm)]*torch.tensor(scales, dtype=hidden_a.dtype)
    torch.testing.assert_close(logits_b, logits_a, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(hidden_b, expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(gates_b['update'], gates_a['update'], atol=2e-4, rtol=2e-4)
    return dict(max_logit_error=float((logits_b-logits_a).abs().max()),
                max_probability_error=float((logits_b.softmax(-1)-logits_a.softmax(-1)).abs().max()),
                max_gate_error=float((gates_b['update']-gates_a['update']).abs().max()),
                max_hidden_error=float((hidden_b-expected).abs().max()),
                min_unit_rms=float(hidden_b.square().mean((0, 1)).sqrt().min()))


def cohort_weights(rows):
    # Equal subject cells; equal folds within each cell (cohort is bias-off only).
    sizes = rows.groupby(['subject_id', 'decoder_bias']).subject_id.transform('size').to_numpy()
    weights = 1./sizes
    return weights/weights.sum()


def choose_medoid(vectors, rows, eligible):
    weights = cohort_weights(rows)
    distances = cdist(vectors, vectors)
    scores = distances@weights
    candidate_scores = np.where(eligible, scores, np.inf)
    if not np.isfinite(candidate_scores).any():
        raise ValueError('No finite balanced, opponent, two-active-unit reference candidates')
    return int(np.argmin(candidate_scores)), scores, distances


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='nested_DA_S32_B8')
    parser.add_argument('--selected-table', type=Path, default=ROOT/'data/analyses/mechanistic_variability/selected_models.htsv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = pd.read_csv(args.selected_table, sep='\t')
    rows = rows.query('model_type == "monoGRU" and nonlinearity == "relu" and hidden_size == 2 and decoder_bias == False').copy()
    if rows.empty or not rows.selection_method.eq('block_reset').all():
        raise ValueError('Provide block-reset train+validation-selected outer winners')
    if not rows.model_pickle_path.map(lambda p: args.folder in resolve(p).parts).all():
        raise ValueError('Source-run mismatch: --folder must match every selected model path')
    keys = ['subject_id', 'decoder_bias', 'outer_loop_n']
    if rows.duplicated(keys).any(): raise ValueError('Duplicate outer winners')
    if set(rows.decoder_bias) != {False}: raise ValueError('Only decoder-bias-off models are permitted')
    for _, group in rows.groupby(['subject_id', 'decoder_bias']):
        if set(group.outer_loop_n) != set(range(1, 11)):
            raise ValueError('Expected all ten outer folds in each subject/condition')
    rows = rows.sort_values(keys).reset_index(drop=True)
    out = args.output or ROOT/'data/analyses/monoGRU_alignment'/args.folder/'monoGRU_relu_ndb'
    out.mkdir(parents=True, exist_ok=True)
    hashes = {str(resolve(p)): mv.sha256(resolve(p)) for p in rows.model_pickle_path}
    if (out/'manifest.json').exists():
        previous = json.loads((out/'manifest.json').read_text())
        if (previous['selected_sha256'] != mv.sha256(args.selected_table)
                or previous['source_model_sha256'] != hashes):
            raise ValueError('Frozen cohort changed: choose a new --output directory to preserve the previous reference')
    models, balanced, audit, hashes = [], [], [], {}
    for index, row in rows.iterrows():
        path = resolve(row.model_pickle_path)
        model = load_model(path)
        if model.decoder_bias != row.decoder_bias: raise ValueError('Decoder bias metadata mismatch')
        hashes[str(path)] = mv.sha256(path)
        b, info = balance_model(model)
        # Self-reference only supplies shapes: hard semantic permutation controls labels.
        labeled, transform_info = align_balanced_model(b, b)
        perm = transform_info['perm']
        scales = np.exp([info['log_scale_1'], info['log_scale_2']])[perm]
        check = check_equivalence(model, labeled, probes(model), perm, scales)
        contrast = labeled.decoder.weight.detach().numpy()[1]-labeled.decoder.weight.detach().numpy()[0]
        opponent = bool(contrast[0]>1e-7 and contrast[1]<-1e-7)
        eligible = info['balance_status']=='finite_minimum' and opponent and check['min_unit_rms']>1e-6
        audit.append(dict(index=index, **info, **check, perm=perm.tolist(), scales=scales.tolist(),
                          contrast_1=float(contrast[0]), contrast_2=float(contrast[1]),
                          opponent=opponent, reference_eligible=eligible))
        models.append(model); balanced.append(labeled)
    vectors = np.stack([core(parameters(m), 2) for m in balanced])
    eligible = np.array([a['reference_eligible'] for a in audit])
    winner, scores, distances = choose_medoid(vectors, rows, eligible)
    reference = balanced[winner]
    with (out/'reference_model.pickle').open('wb') as handle: pickle.dump(reference, handle)
    # Distances use the same core blocks as align_rnns; decoder bias is excluded.
    sensitivity = []
    for subject in rows.subject_id.unique():
        keep = rows.subject_id.ne(subject).to_numpy()
        indices = np.flatnonzero(keep)
        other, _, _ = choose_medoid(vectors[keep], rows.loc[keep], eligible[keep])
        chosen = int(indices[other])
        sensitivity.append(dict(omitted_subject=subject, medoid_index=chosen,
                                distance_to_full_reference=float(distances[winner, chosen]),
                                subject_id=rows.iloc[chosen].subject_id,
                                decoder_bias=bool(rows.iloc[chosen].decoder_bias),
                                outer_loop_n=int(rows.iloc[chosen].outer_loop_n)))
    for i, (model, b) in enumerate(zip(models, balanced)):
        aligned, info = align_balanced_model(b, reference)
        # Already labeled: further swaps/scaling must be unnecessary.
        if list(info['perm']) != [0, 1]: raise RuntimeError('Semantic labels were not stable')
        a = audit[i]
        a.update(check_equivalence(model, aligned, probes(model), a['perm'], a['scales']))
        a.update(population_medoid_objective=float(scores[i]), distance_to_reference=info['aligned_distance'])
        row = rows.iloc[i]
        relative = Path('aligned')/row.subject_id/row.model_id/f'outer_fold_{row.outer_loop_n}.pickle'
        (out/relative).parent.mkdir(parents=True, exist_ok=True)
        with (out/relative).open('wb') as handle: pickle.dump(aligned, handle)
        a['aligned_model_path'] = str((out/relative).resolve())
    table = pd.concat([rows, pd.DataFrame(audit).drop(columns='index')], axis=1)
    table.to_csv(out/'aligned_models.htsv', sep='\t', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'leave_subject_out.htsv', sep='\t', index=False)
    np.savez_compressed(out/'balanced_core_parameters.npz', parameters=vectors)
    plot_reference(reference, rows.iloc[winner], out)
    manifest = dict(folder=args.folder, model_type='monoGRU +BC-DB', decoder_bias=False, n_models=len(rows), n_subjects=rows.subject_id.nunique(),
        label_rule=LABEL_RULE, scale_rule='Independent minimum FULL trainable-parameter L2 norm on positive ReLU scaling orbit; unanchored units use flagged [-8,8] log-scale approximation',
        reference_rule='Actual balanced model minimizing subject-weighted mean Euclidean core distance to ALL cohort models, among finite opponent two-active-unit candidates; stable row ordering breaks exact ties',
        objective_blocks=['rnn.W_ih', 'rnn.bias_h', 'rnn.W_hh', 'decoder.weight'],
        reference_index=winner, reference_source=rows.iloc[winner].to_dict(),
        reference_candidates=int(eligible.sum()), unanchored_models=sum(a['balance_status']!='finite_minimum' for a in audit),
        nonopponent_models=sum(not a['opponent'] for a in audit),
        validation='32 fixed synthetic binary histories, length 64; logits, probabilities, hidden transforms, update gates; no held-out behavior used',
        scope='Descriptive reference learned from this entire selected cohort; not independent of this cohort and not for held-out mechanistic inference without refitting in training folds',
        limitations='Minimum norm is a coordinate convention, not a biological truth. Raw scaling-dependent hidden statistics change. No output relabeling, sign flips, or decoder common-mode gauge removal. Frozen reference should be reused for subsequent cohorts.',
        selected_table=str(args.selected_table.resolve()), selected_sha256=mv.sha256(args.selected_table),
        source_model_sha256=hashes,
        code_sha256={str(Path(p).resolve()):mv.sha256(Path(p)) for p in [__file__, mv.__file__, sys.modules[canonicalize.__module__].__file__]},
        reference_sha256=mv.sha256(out/'reference_model.pickle'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_value)+'\n')
    print(json.dumps(dict(output=str(out), reference=rows.iloc[winner][keys].to_dict(),
        reference_candidates=int(eligible.sum()), n_models=len(rows),
        nonopponent=manifest['nonopponent_models'], unanchored=manifest['unanchored_models'],
        max_logit_error=max(a['max_logit_error'] for a in audit)), indent=2, default=json_value), flush=True)


def plot_reference(model, row, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blocks=[('Candidate input',model.rnn.W_ih.T), ('Candidate recurrent',model.rnn.W_hh.T),
            ('Candidate bias',model.rnn.bias_h[:,None]), ('Gate input',model.rnn.W_iz.T),
            ('Gate recurrent',model.rnn.W_hz.T), ('Decoder',model.decoder.weight)]
    limit=max(float(t.detach().abs().max()) for _,t in blocks)
    fig, axes=plt.subplots(2,3,figsize=(10,5))
    for ax,(name,tensor) in zip(axes.flat,blocks):
        values=tensor.detach().numpy()
        im=ax.imshow(values,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto')
        for (i,j),value in np.ndenumerate(values):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9)
        columns = (['c', 'a', 'o'] if name in ('Candidate input','Gate input')
                   else ['h1','h2'] if name in ('Candidate recurrent','Gate recurrent','Decoder') else ['bias'])
        row_labels = (['h1','h2'] if name.startswith('Candidate')
                      else ['choice 0','choice 1'] if name=='Decoder' else ['gate'])
        ax.set(title=name,xticks=range(values.shape[1]),yticks=range(values.shape[0]),
               xticklabels=columns,yticklabels=row_labels)
    fig.suptitle(f'Balanced medoid · {row.subject_id} · outer {row.outer_loop_n} · decoder bias {row.decoder_bias}\nh1 favors choice 1; h2 favors choice 0')
    fig.tight_layout(rect=(0,0,.9,.9));cax=fig.add_axes([.92,.15,.015,.65]);fig.colorbar(im,cax=cax,label='Parameter value')
    fig.savefig(out/'reference_parameters.png',dpi=180,bbox_inches='tight');fig.savefig(out/'reference_parameters.svg',bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__': main()
