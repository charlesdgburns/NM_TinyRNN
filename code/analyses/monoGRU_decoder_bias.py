"""Two-unit monoGRU ReLU: what removing the decoder bias does.

Consolidates the decoder-bias appendix analysis: solution families of all 1,980 paired inner fits,
their held-out cost, the effect on outer-fold model selection and on across-mouse similarity, and the
six-panel appendix figure with its summary table. The plots stage also draws the decoder geometry of the first example
fit (hidden states -> softmax scores -> scores plus bias; plotting in code/figures/monoGRU_decoder_softmax.py).

From the repository root:
    .venv/bin/python code/analyses/monoGRU_decoder_bias.py                 # fits, statistics, tables, figure
    .venv/bin/python code/analyses/monoGRU_decoder_bias.py --stage report  # fits + statistics + tables, no figure
    .venv/bin/python code/analyses/monoGRU_decoder_bias.py --stage plots   # figure and tables from saved fits
    .venv/bin/python code/analyses/monoGRU_decoder_bias.py --recompute     # ignore the per-fit cache

Inputs (no refitting, no torch; the softmax figure reads one pickled model with a torch-free reader): data/analyses/performance/analysis_df.htsv (inner-fit paths), each fit's
*_trials_data.htsv and *_info.json under data/rnns/<run>, and, for selection and similarity,
data/analyses/mechanistic_variability/{selected_models,pairwise_similarities}.htsv.
Outputs: data/analyses/monogru_decoder_bias/ (tables, manifest) and results/appendix/monogru_decoder_bias/
(figure_decoder_bias, figure_decoder_softmax, tables).

Definitions
- Every fit's decoder (w1, w2, b) is recovered exactly by regressing its saved logit on the two hidden units; the unit-driven
  part of the logit is u = w . h.
- A fit is *collapsed* when u takes its less frequent sign on fewer than 5% of trials, so the units alone can signal one side only.
  Collapsed fits with a decoder bias are 'sign from bias'; without one they are 'one-sided output'. All other fits are 'sign in units'.
- Held-out NLL: the pipeline's block-reset outer-fold loss (eval_pred_loss), as in training and model selection. The
  continuous-state version (saved logits on the evaluation trials, free choices in the same session; the logit at trial t
  predicts the choice at t+1 with P(choice = 1) = sigmoid(-logit)) is kept alongside as heldout_nll_continuous.
- Participation ratio: uncentred, as in monogru_dimensionality.py.
- Inference: 11 mice are the replicates. Fits are averaged within mouse, then a one-sample t-test over mouse differences.
  Fits within a mouse and fold are dependent; the tests are descriptive.
"""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = 'nested_DA_S32_B8'
MODEL = dict(model_type='monoGRU', nonlinearity='relu', hidden_size=2)
TOL = 1e-6                    # numerical zero for activity and logit
COLLAPSE_MINORITY = 0.05      # collapsed: u takes its less frequent sign on < 5% of trials
LOGIT_BINS = 40
KEY = ['subject', 'outer', 'inner']
GREY, ORANGE, AXIS_GREY = '#4d4d4d', '#e08214', '#888888'   # with / without decoder bias; red-blue is reserved for the logit
FAMILIES = ['sign in units', 'sign from bias', 'one-sided output']
METHODS = dict(
    fits='All inner fits of the 2-unit monoGRU ReLU variants (11 mice x 10 outer x 9 inner folds, with and without decoder bias), read from saved trial tables',
    decoder='(w1, w2, b) recovered by least squares from the saved logit; maximum residual and the free intercept of no-bias fits are checked',
    collapse='Unit-driven logit u = w.h takes its less frequent sign on < 5% of trials (|u| > 1e-6)',
    heldout_nll='Block-reset outer-fold loss (eval_pred_loss), as in training and selection; continuous saved-logit NLL kept as heldout_nll_continuous',
    participation='Uncentred participation ratio of the two hidden units over the whole saved trajectory',
    mouse_level_inference='Fits averaged within mouse, one-sample t-test over 11 mouse differences; descriptive, fits within a fold are dependent',
    selection='Outer-fold winners from selected_models.htsv (block-reset train+validation CE); not recomputed here',
    similarity='Selected-model pairs from pairwise_similarities.htsv, across mice; pair family = family of both fits; points are mouse pairs (mean over fold pairs)',
    examples='Each pair closest to the group medians (z-scored): collapsed with bias -> still one-sided; collapsed with a silent unit -> sign in units')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def locations(run, tables_dir=None, figures_dir=None):
    tables = Path(tables_dir) if tables_dir else REPO / 'data/analyses/monogru_decoder_bias'
    figures = Path(figures_dir) if figures_dir else REPO / 'results/appendix/monogru_decoder_bias'
    inputs = dict(analysis_df=REPO / 'data/analyses/performance/analysis_df.htsv',
                  selected=REPO / 'data/analyses/mechanistic_variability/selected_models.htsv',
                  pairwise=REPO / 'data/analyses/mechanistic_variability/pairwise_similarities.htsv',
                  similarity_manifest=REPO / 'data/analyses/mechanistic_variability/manifest.json')
    return tables, figures, inputs


def resolve(path):
    """Saved paths are relative to the directory that contains the NM_TinyRNN folder."""
    return str(REPO.parent / path)


def load_fit_rows(run, inputs):
    frame = pd.read_csv(inputs['analysis_df'], sep='\t')
    frame = frame[(frame.model_type == MODEL['model_type']) & (frame.nonlinearity == MODEL['nonlinearity']) & (frame.hidden_size == MODEL['hidden_size'])].copy()
    if not frame.save_path.str.contains(f'/{run}/').all():
        raise ValueError(f'analysis_df.htsv is not from {run}; rebuild data/analyses/performance for this run')
    frame = frame.rename(columns={'subject_id': 'subject', 'outer_loop_n': 'outer', 'inner_loop_idx': 'inner'})
    assert not frame.duplicated(KEY + ['decoder_bias']).any()
    assert frame.subject.nunique() == 11 and len(frame) == 11 * 10 * 9 * 2, f'expected 1,980 fits, found {len(frame)}'
    assert frame.groupby(['subject', 'decoder_bias']).apply(lambda g: set(zip(g.outer, g.inner)) == {(o, i) for o in range(1, 11) for i in range(9)}, include_groups=False).all()
    manifest = json.loads(Path(inputs['similarity_manifest']).read_text())
    if manifest.get('folder') != run:
        raise ValueError(f"mechanistic_variability caches are from {manifest.get('folder')}, not {run}")
    return frame.reset_index(drop=True)


def characterise_fit(task):
    """One saved fit -> decoder, solution-family inputs, participation and held-out NLL."""
    subject, outer, inner, bias, trials_path, info_path = task
    d = pd.read_csv(trials_path, sep='\t', usecols=['hidden_1', 'hidden_2', 'logit_value', 'split', 'forced_choice', 'choice', 'session_folder_name'])
    info = json.loads(Path(info_path).read_text())
    H = d[['hidden_1', 'hidden_2']].to_numpy(float); L = d.logit_value.to_numpy(float); T = len(L)
    A = np.column_stack([H, np.ones(T)]); coef = np.linalg.lstsq(A, L, rcond=None)[0]
    w = coef[:2]; b = coef[2] if bias else 0.0
    u = H @ w
    ev = np.linalg.eigvalsh(H.T @ H); participation = ev.sum() ** 2 / (ev ** 2).sum()
    bins = np.argsort(np.argsort(L)) * LOGIT_BINS // T; r2 = []
    for k in range(2):
        x = H[:, k] - H[:, k].mean(); mu = np.bincount(bins, weights=x, minlength=LOGIT_BINS) / np.bincount(bins, minlength=LOGIT_BINS)
        r2.append(1 - ((x - mu[bins]) ** 2).mean() / max(x.var(), 1e-10))
    split = d.split.to_numpy(); sess = d.session_folder_name.to_numpy()
    ok = (split[1:] == 'eval') & (d.forced_choice.to_numpy()[1:] == 0) & (sess[1:] == sess[:-1])
    y = d.choice.to_numpy(float)[1:][ok]; z = -L[:-1][ok]
    nll = float(np.mean(np.logaddexp(0, z) - y * z))
    return dict(subject=subject, outer=outer, inner=inner, decoder_bias=bool(bias), n_trials=T, w1=w[0], w2=w[1], b=b, fit_free_intercept=coef[2],
                decoder_residual=float(np.abs(A @ coef - L).max()), logit_sd=L.std(),
                bias_decides=float(np.mean(((L > TOL) & (u <= TOL)) | ((L < -TOL) & (u >= -TOL)))),
                unit_part_minority_sign=float(min((u > TOL).mean(), (u < -TOL).mean())), participation=participation, logit_determined=float(np.mean(r2)),
                unit_silent=bool((H.std(0) < 1e-5).any()), heldout_nll_continuous=nll, n_heldout_choices=int(ok.sum()),
                pipeline_eval_pred_loss=info['eval_pred_loss'], pipeline_val_loss=info['val_loss'], trials_data_path=trials_path)


def compute_fits(rows, workers):
    tasks = [(r.subject, r.outer, r.inner, bool(r.decoder_bias), resolve(r.trials_data_path), resolve(r.info_path)) for r in rows.itertuples()]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            out = list(pool.map(characterise_fit, tasks, chunksize=20))
    else:
        out = [characterise_fit(t) for t in tasks]
    fits = pd.DataFrame(out)
    assert fits.decoder_residual.max() < 1e-4, 'saved logits are not a linear readout of the saved hidden states'
    assert fits[~fits.decoder_bias].fit_free_intercept.abs().max() < 1e-4, 'a fit without a decoder bias has a non-zero intercept'
    return fits


def load_or_compute_fits(run, rows, inputs, tables, recompute, workers):
    cache = tables / 'paired_fits.htsv'; meta = tables / 'paired_fits.json'
    stamp = dict(run=run, analysis_df_sha256=sha(inputs['analysis_df']), rows=len(rows), tol=TOL, bins=LOGIT_BINS)
    if cache.exists() and meta.exists() and not recompute:
        old = json.loads(meta.read_text())
        if all(old.get(k) == v for k, v in stamp.items()):
            print(f'Loaded per-fit table from cache: {cache}')
            return with_heldout_nll(pd.read_csv(cache, sep='\t'))
        print('Per-fit cache does not match the current inputs; recomputing')
    fits = compute_fits(rows, workers)
    tables.mkdir(parents=True, exist_ok=True); fits.to_csv(cache, sep='\t', index=False); write_json(meta, stamp)
    return with_heldout_nll(fits)


def with_heldout_nll(fits):
    """Reported held-out NLL = block-reset eval_pred_loss; continuous version kept (older caches name it heldout_nll)."""
    fits = fits.copy()
    if 'heldout_nll_continuous' not in fits.columns and 'heldout_nll' in fits.columns:
        fits = fits.rename(columns={'heldout_nll': 'heldout_nll_continuous'})
    fits['heldout_nll'] = fits['pipeline_eval_pred_loss']
    return fits


def add_families(fits):
    fits = fits.copy()
    fits['collapsed'] = fits.unit_part_minority_sign < COLLAPSE_MINORITY
    fits['family'] = np.where(~fits.collapsed, 'sign in units', np.where(fits.decoder_bias, 'sign from bias', 'one-sided output'))
    return fits


def median_pair(group, columns):
    z = (group[columns] - group[columns].median()) / (group[columns].std() + 1e-9)
    return group.loc[(z ** 2).sum(axis=1).idxmin()]


def mouse_test(differences, label):
    d = differences.dropna(); t = stats.ttest_1samp(d, 0)
    return dict(contrast=label, mean_difference=d.mean(), t=t.statistic, df=len(d) - 1, p=t.pvalue, n_mice_positive=int((d > 0).sum()), n_mice=len(d))


def statistics(fits, inputs):
    fits = add_families(fits)
    pairs = fits[fits.decoder_bias].set_index(KEY).join(fits[~fits.decoder_bias].set_index(KEY), lsuffix='_on', rsuffix='_off').reset_index()
    assert len(pairs) == 990
    pairs['nll_change'] = pairs.heldout_nll_off - pairs.heldout_nll_on
    out = {}
    out['family_counts'] = fits.groupby(['decoder_bias', 'family']).agg(n=('outer', 'size'), participation_median=('participation', 'median'), silent_unit=('unit_silent', 'sum'),
                                                                        logit_determined_median=('logit_determined', 'median'), heldout_nll_pooled=('heldout_nll', 'mean')).reset_index()
    out['transitions'] = pairs.groupby(['family_on', 'family_off']).agg(n=('nll_change', 'size'), participation_with_bias=('participation_on', 'median'), participation_without=('participation_off', 'median'),
                                                                         nll_with_bias=('heldout_nll_on', 'mean'), nll_without=('heldout_nll_off', 'mean')).reset_index()
    mouse = fits.groupby(['decoder_bias', 'family', 'subject']).heldout_nll.mean().unstack(['decoder_bias', 'family'])
    tests = [mouse_test(mouse[(True, 'sign from bias')] - mouse[(True, 'sign in units')], 'with bias: sign from bias minus sign in units'),
             mouse_test(mouse[(False, 'one-sided output')] - mouse[(False, 'sign in units')], 'without bias: one-sided output minus sign in units')]
    for fo, ff in [('sign in units', 'sign in units'), ('sign from bias', 'sign in units'), ('sign from bias', 'one-sided output')]:
        tests.append(mouse_test(pairs[(pairs.family_on == fo) & (pairs.family_off == ff)].groupby('subject').nll_change.mean(), f'paired folds {fo} -> {ff}: without minus with bias'))
    out['nll_tests'] = pd.DataFrame(tests)
    # --- selected outer-fold winners ---
    selected = pd.read_csv(inputs['selected'], sep='\t')
    assert selected.selection_method.eq('block_reset').all()
    selected = selected[(selected.model_type == MODEL['model_type']) & (selected.nonlinearity == MODEL['nonlinearity'])]
    winners = selected.rename(columns={'subject_id': 'subject', 'outer_loop_n': 'outer', 'inner_loop_idx': 'inner'})[KEY + ['decoder_bias', 'eval_CE', 'model_id']].merge(
        fits, on=KEY + ['decoder_bias'], validate='one_to_one')
    assert len(winners) == 220
    candidates = fits[fits.decoder_bias].groupby(['subject', 'outer']).collapsed.apply(lambda s: int((~s).sum())).rename('sign_in_units_candidates').reset_index()
    with_bias = winners[winners.decoder_bias].merge(candidates, on=['subject', 'outer']).join(
        winners[~winners.decoder_bias].set_index(['subject', 'outer'])[['eval_CE', 'family']].rename(columns={'eval_CE': 'eval_CE_without_bias', 'family': 'family_without_bias'}), on=['subject', 'outer'])
    without_bias = winners[~winners.decoder_bias]
    out['selected'] = pd.DataFrame([dict(collapsed_with_bias=int(with_bias.collapsed.sum()), outer_folds=len(with_bias), collapsed_without_bias=int(without_bias.collapsed.sum()),
                                         won_despite_available_alternative=int((with_bias.collapsed & (with_bias.sign_in_units_candidates > 0)).sum()),
                                         pipeline_eval_loss_collapsed_winners=with_bias[with_bias.collapsed].eval_CE.mean(),
                                         pipeline_eval_loss_same_folds_without_bias=with_bias[with_bias.collapsed].eval_CE_without_bias.mean(),
                                         per_mouse=json.dumps(with_bias.groupby('subject').collapsed.sum().astype(int).to_dict()))])
    # --- across-mouse similarity of selected models by solution family ---
    pw = pd.read_csv(inputs['pairwise'], sep='\t', usecols=['model_id', 'subject_A', 'model_A_outer_n', 'subject_B', 'model_B_outer_n', 'within_subject', 'hidden_similarity', 'weight_similarity', 'gate_update_similarity', 'logit_value_similarity'])
    pw = pw[pw.model_id.isin(winners.model_id.unique()) & ~pw.within_subject]
    family = winners[['model_id', 'subject', 'outer', 'family']]
    pw = pw.merge(family.rename(columns={'subject': 'subject_A', 'outer': 'model_A_outer_n', 'family': 'family_A'}), on=['model_id', 'subject_A', 'model_A_outer_n'], validate='many_to_one').merge(
        family.rename(columns={'subject': 'subject_B', 'outer': 'model_B_outer_n', 'family': 'family_B'}), on=['model_id', 'subject_B', 'model_B_outer_n'], validate='many_to_one')
    a, b = pw.family_A == 'sign in units', pw.family_B == 'sign in units'
    pw['pair_family'] = np.where(a & b, 'both sign in units', np.where(~a & ~b, 'both collapsed', 'one of each')); pw['decoder_bias'] = ~pw.model_id.str.endswith('_ndb')
    out['similarity'] = pw.groupby(['decoder_bias', 'pair_family']).agg(n_pairs=('hidden_similarity', 'size'), hidden=('hidden_similarity', 'mean'), weight=('weight_similarity', 'mean'),
                                                                       update_gate=('gate_update_similarity', 'mean'), logit=('logit_value_similarity', 'mean')).reset_index()
    out['dyads'] = pw.groupby(['decoder_bias', 'pair_family', 'subject_A', 'subject_B']).hidden_similarity.mean().reset_index()
    # --- summary table (one column per decoder-bias condition) ---
    counts = fits.groupby(['decoder_bias', 'family']).size(); tw, to = out['nll_tests'].iloc[0], out['nll_tests'].iloc[1]
    median_participation = fits[fits.collapsed].groupby('decoder_bias').participation.median()
    n_folds = len(with_bias)
    out['summary_numeric'] = pd.DataFrame([
        dict(row='Sign-in-units fits (of 990)', with_bias=counts[(True, 'sign in units')], without_bias=counts[(False, 'sign in units')]),
        dict(row='Collapsed fits (of 990)', with_bias=counts[(True, 'sign from bias')], without_bias=counts[(False, 'one-sided output')]),
        dict(row='Median participation of collapsed fits', with_bias=median_participation[True], without_bias=median_participation[False]),
        dict(row='Held-out NLL cost of collapse, within mice', with_bias=tw.mean_difference, without_bias=to.mean_difference),
        dict(row='  p value (t-test over mice)', with_bias=tw.p, without_bias=to.p),
        dict(row='  mice with higher NLL when collapsed', with_bias=tw.n_mice_positive, without_bias=to.n_mice_positive),
        dict(row=f'Selected outer-fold models that are collapsed (of {n_folds})', with_bias=int(with_bias.collapsed.sum()), without_bias=int(without_bias.collapsed.sum()))])
    out['summary'] = pd.DataFrame({'With decoder bias': [f"{counts[(True, 'sign in units')]}", f"{counts[(True, 'sign from bias')]} (sign from bias)", f"{median_participation[True]:.2f}",
                                                         f"{tw.mean_difference:+.3f} (p = {tw.p:.2f})", f"{int(with_bias.collapsed.sum())}"],
                                   'No decoder bias': [f"{counts[(False, 'sign in units')]}", f"{counts[(False, 'one-sided output')]} (one-sided output)", f"{median_participation[False]:.2f}",
                                                       f"{to.mean_difference:+.3f} ({to.n_mice_positive} of {to.n_mice} mice)", f"{int(without_bias.collapsed.sum())}"]},
                                  index=['Sign-in-units fits (of 990)', 'Collapsed fits (of 990)', 'Median participation of collapsed fits', 'Held-out NLL cost of collapse, within mice',
                                         f'Selected outer-fold models that are collapsed (of {n_folds})'])
    out['summary'].index.name = ''
    out['mouse_nll'] = mouse
    out['fits'] = fits; out['pairs'] = pairs
    # --- example pairs ---
    top = median_pair(pairs[(pairs.family_on == 'sign from bias') & (pairs.family_off == 'one-sided output')], ['participation_on', 'participation_off', 'heldout_nll_on', 'heldout_nll_off'])
    bottom = median_pair(pairs[(pairs.family_on == 'sign from bias') & (pairs.family_off == 'sign in units') & pairs.unit_silent_on], ['participation_off', 'heldout_nll_on', 'heldout_nll_off'])
    out['examples'] = pd.DataFrame([top, bottom], index=['collapse stays one-sided', 'collapse becomes sign in units'])[KEY + ['participation_on', 'participation_off', 'heldout_nll_on', 'heldout_nll_off', 'trials_data_path_on', 'trials_data_path_off']]
    return out


def save_tables(out, tables, figures):
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    for name in ['family_counts', 'transitions', 'nll_tests', 'selected', 'similarity', 'dyads', 'examples', 'summary_numeric']:
        out[name].to_csv(tables / f'{name}.htsv', sep='\t', index=False)
    out['summary'].to_csv(figures / 'table_decoder_bias_summary.csv')
    out['summary_numeric'].to_csv(figures / 'table_decoder_bias_summary_numeric.csv', index=False)
    pd.crosstab(out['pairs'].family_on, out['pairs'].family_off, margins=True, margins_name='Total').to_csv(figures / 'table_decoder_bias_family_transitions.csv')


def trial_states(path):
    d = pd.read_csv(resolve(path), sep='\t', usecols=['hidden_1', 'hidden_2', 'logit_value'])
    return d[['hidden_1', 'hidden_2']].to_numpy(float), d.logit_value.to_numpy(float)


def plot(out, figures):
    plt.rcParams.update({'font.size': 12.5, 'axes.titlesize': 12.5, 'axes.labelsize': 12.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'xtick.major.width': 1.1, 'ytick.major.width': 1.1, 'xtick.major.size': 4.5, 'ytick.major.size': 4.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 1.1, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    mouse, dyads, similarity, examples = out['mouse_nll'], out['dyads'], out['similarity'].set_index(['decoder_bias', 'pair_family']), out['examples']
    chance = np.log(2); vmax = 3.0
    fig = plt.figure(figsize=(11.5, 5.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], wspace=.55, hspace=1.15, left=.105, right=.875, top=.84, bottom=.17)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]); rng = np.random.default_rng(0)
    ax = axes[0, 0]
    for i, (bias, family, color) in enumerate([(True, 'sign in units', GREY), (True, 'sign from bias', GREY), (False, 'sign in units', ORANGE), (False, 'one-sided output', ORANGE)]):
        v = mouse[(bias, family)].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)), v, s=26, color=color, alpha=.6, lw=0); ax.hlines(v.mean(), i - .28, i + .28, color='#222222', lw=2.6, zorder=3)
    ax.axhline(chance, color=AXIS_GREY, lw=1.1, ls=':'); ax.text(-.4, chance + .003, 'chance', ha='left', va='bottom', fontsize=10, color='#555555'); ax.set_ylim(.46, .715)
    ax.set_xticks(range(4), ['sign in\nunits', 'sign from\nbias', 'sign in\nunits', 'one-sided\noutput']); ax.tick_params(axis='x', labelsize=10)
    ax.set_ylabel('Held-out NLL per choice'); ax.set_title('Collapse is costly\nwithout the bias', loc='left')
    ax.text(.25, -.56, 'decoder bias', transform=ax.transAxes, ha='center', fontsize=11, color=GREY); ax.text(.75, -.56, 'no decoder bias', transform=ax.transAxes, ha='center', fontsize=11, color=ORANGE)
    ax = axes[1, 0]
    for i, (bias, pair, color) in enumerate([(True, 'both sign in units', GREY), (True, 'one of each', GREY), (True, 'both collapsed', GREY), (False, 'both sign in units', ORANGE)]):
        v = dyads[(dyads.decoder_bias == bias) & (dyads.pair_family == pair)].hidden_similarity.to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.16, .16, len(v)), v, s=16, color=color, alpha=.45, lw=0); ax.hlines(similarity.loc[(bias, pair), 'hidden'], i - .28, i + .28, color='#222222', lw=2.6, zorder=3)
    ax.axhline(0, color=AXIS_GREY, lw=.9); ax.set_xticks(range(4), ['both sign\nin units', 'one of\neach', 'both sign\nfrom bias', 'all\npairs']); ax.tick_params(axis='x', labelsize=10)
    ax.set_ylabel('Across-mouse\nhidden-state similarity'); ax.set_title('Selected fits agree\nwithin a family', loc='left')
    ax.text(.375, -.56, 'decoder bias', transform=ax.transAxes, ha='center', fontsize=11, color=GREY); ax.text(.875, -.56, 'no bias', transform=ax.transAxes, ha='center', fontsize=11, color=ORANGE)
    for r, example in enumerate([examples.iloc[0], examples.iloc[1]]):
        states = [trial_states(example.trials_data_path_on), trial_states(example.trials_data_path_off)]; lim = 1.06 * max(H.max() for H, _ in states)
        for c, ((H, L), nll) in enumerate(zip(states, [example.heldout_nll_on, example.heldout_nll_off])):
            ax = axes[r, c + 1]; idx = rng.choice(len(L), min(3000, len(L)), replace=False)
            scatter = ax.scatter(H[idx, 0], H[idx, 1], c=L[idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax, s=7, lw=0, rasterized=True)
            ax.set_xlim(-.05 * lim, lim); ax.set_ylim(-.05 * lim, lim); ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('Hidden unit 1')
            if c == 0: ax.set_ylabel('Hidden unit 2')
            else: ax.tick_params(labelleft=False)
            ax.text(.12, .99, f'held-out NLL {nll:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color='black')
            ax.set_title(f'{example.subject}, outer fold {int(example.outer)}, inner fold {int(example.inner)}', loc='left', fontsize=10, pad=8)
    colorbar = fig.colorbar(scatter, cax=fig.add_axes([.94, .3, .011, .4]), extend='both'); colorbar.set_label('Logit', fontsize=12.5); colorbar.ax.tick_params(labelsize=10.5, width=1.1); colorbar.outline.set_linewidth(1.1)
    fig.canvas.draw(); first, second = axes[0, 1].get_position(), axes[0, 2].get_position(); y = .955
    left = fig.text(first.x0 + first.width / 2, y, 'With decoder bias', ha='center', va='center', fontsize=13.5, color=GREY, fontweight='bold')
    right = fig.text(second.x0 + second.width / 2, y, 'Without decoder bias', ha='center', va='center', fontsize=13.5, color=ORANGE, fontweight='bold')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); inverse = fig.transFigure.inverted()
    gap_left, gap_right = inverse.transform(left.get_window_extent(renderer))[1, 0], inverse.transform(right.get_window_extent(renderer))[0, 0]; mid = (gap_left + gap_right) / 2; half = min(.55 / 2 / fig.get_figwidth(), (gap_right - gap_left) / 2 - .01)
    fig.add_artist(FancyArrowPatch((mid - half, y), (mid + half, y),
                                   transform=fig.transFigure, arrowstyle='-|>', mutation_scale=15, color='#222222', lw=1.6))
    for r in range(2):
        a, b = axes[r, 1].get_position(), axes[r, 2].get_position(); yy = a.y0 + a.height / 2; mid = (a.x1 + b.x0) / 2; half = .55 / 2 / fig.get_figwidth()   # 0.55 in long, centred in the gap
        fig.add_artist(FancyArrowPatch((mid - half, yy), (mid + half, yy), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=15, color='#222222', lw=1.6))
    for ax, letter in zip(axes.ravel(), 'abcdef'):          # row-major: a-c top row, d-f bottom row; left of the y-axis label
        fig.canvas.draw(); x = inverse.transform(ax.get_tightbbox(renderer))[0, 0]
        fig.text(x - .004, ax.get_position().y1 + .05, letter, fontsize=16, fontweight='bold', va='bottom', ha='right')
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(figures / f'figure_decoder_bias.{ext}', dpi=300)
    plt.close(fig)


# --- decoder weights without torch --------------------------------------------------------------------------------
# The softmax figure needs the two decoder rows separately (the logit only fixes their difference), so the decoder is
# read from the fit's pickled model. The reader rebuilds tensors as numpy arrays and stubs every other class; the
# result is checked against the saved logits in decoder_softmax().
class _Stub:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state if isinstance(state, dict) else {})
    def __call__(self, *args, **kwargs): return self


class _Storage:
    def __init__(self, name):
        self.dtype = np.dtype({'Float': np.float32, 'Double': np.float64, 'Long': np.int64}.get(name.replace('Storage', ''), np.float32))


def _storage_from_bytes(data):
    """torch.storage._load_from_bytes for a torch.save-d storage (zip archive or legacy stream)."""
    import io, pickle, zipfile
    if data[:2] == b'PK':
        archive = zipfile.ZipFile(io.BytesIO(data)); pkl = [n for n in archive.namelist() if n.endswith('data.pkl')][0]; prefix = pkl[:-len('data.pkl')]
        class Reader(pickle.Unpickler):
            def find_class(self, module, name): return _Storage(name) if name.endswith('Storage') else _Stub
            def persistent_load(self, pid): return np.frombuffer(archive.read(prefix + 'data/' + pid[2]), dtype=pid[1].dtype)
        return Reader(io.BytesIO(archive.read(pkl))).load()
    stream = io.BytesIO(data); pickle.load(stream); pickle.load(stream); pickle.load(stream)   # magic number, protocol, system info
    dtypes = {}
    class Reader(pickle.Unpickler):
        def find_class(self, module, name): return _Storage(name) if name.endswith('Storage') else _Stub
        def persistent_load(self, pid): dtypes[pid[2]] = pid[1].dtype; return pid[2]
    key = Reader(stream).load(); arrays = {}
    for k in pickle.load(stream):
        n = int.from_bytes(stream.read(8), 'little'); dtype = np.dtype(dtypes[k]); arrays[k] = np.frombuffer(stream.read(n * dtype.itemsize), dtype=dtype)
    return arrays[key]


def _rebuild_tensor(storage, offset, size, stride, *rest):
    if len(size) == 0: return np.array(storage[offset], dtype=float)
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=tuple(st * storage.itemsize for st in stride)).astype(float)


def read_decoder(model_path):
    """Decoder weight (2 outputs x 2 units) and bias (zeros when absent) of a pickled TinyRNN, without importing torch."""
    import collections, pickle
    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if name == '_rebuild_tensor_v2': return _rebuild_tensor
            if name == '_rebuild_parameter': return lambda data, requires_grad, hooks: data
            if name == '_load_from_bytes': return _storage_from_bytes
            if module == 'collections' and name == 'OrderedDict': return collections.OrderedDict
            if module in ('builtins', 'copyreg', '__builtin__'): return super().find_class(module, name)
            return type(name, (_Stub,), {})
    with open(model_path, 'rb') as handle:
        decoder = Reader(handle).load()._modules['decoder']._parameters
    weight = np.asarray(decoder['weight'], float); bias = decoder.get('bias')
    return weight, (np.zeros(weight.shape[0]) if bias is None else np.asarray(bias, float))


def decoder_softmax(out, figures):
    """Decoder geometry of the first example fit with its bias (hidden states -> W h -> W h + b); code/figures/monoGRU_decoder_softmax.py."""
    import sys
    sys.path.insert(0, str(REPO.parent))
    from NM_TinyRNN.code.figures.monoGRU_decoder_softmax import decoder_softmax_figure
    example = out['examples'].iloc[0]; trials = resolve(example.trials_data_path_on)
    hidden, logit = trial_states(trials)
    weight, bias = read_decoder(str(trials).replace('_trials_data.htsv', '_model.pickle'))
    scores = hidden @ weight.T + bias
    error = float(np.abs(scores[:, 0] - scores[:, 1] - logit).max())
    assert error < 1e-4, f'decoder read from the model does not reproduce the saved logits (max error {error:.2e})'
    fig = decoder_softmax_figure(hidden, weight, bias, f'With decoder bias: {example.subject}, outer fold {int(example.outer)}, inner fold {int(example.inner)}')
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(figures / f'figure_decoder_softmax.{ext}', dpi=300)
    plt.close(fig)
    return dict(subject=example.subject, outer=int(example.outer), inner=int(example.inner), decoder_weight=weight.tolist(), decoder_bias=bias.tolist(), max_logit_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default=DEFAULT_RUN, help='run under data/rnns that the caches were built from')
    parser.add_argument('--stage', choices=['all', 'report', 'plots'], default='all', help='report: fits, statistics, tables; plots: figure and tables from cached fits')
    parser.add_argument('--recompute', action='store_true', help='ignore the per-fit cache')
    parser.add_argument('--workers', type=int, default=1, help='processes for the per-fit pass (1,980 trial tables)')
    parser.add_argument('--tables-dir', help='override data/analyses/monogru_decoder_bias')
    parser.add_argument('--figures-dir', help='override results/appendix/monogru_decoder_bias')
    args = parser.parse_args()
    tables, figures, inputs = locations(args.run, args.tables_dir, args.figures_dir)
    rows = load_fit_rows(args.run, inputs)
    fits = load_or_compute_fits(args.run, rows, inputs, tables, args.recompute, args.workers)
    out = statistics(fits, inputs)
    save_tables(out, tables, figures)
    softmax_example = None
    if args.stage in ['all', 'plots']:
        plot(out, figures)
        softmax_example = decoder_softmax(out, figures)
    manifest = dict(run=args.run, n_fits=len(fits), methods=METHODS, example_pairs=out['examples'][KEY].to_dict('records'),
                    max_decoder_residual=float(fits.decoder_residual.max()), max_free_intercept_without_bias=float(fits[~fits.decoder_bias].fit_free_intercept.abs().max()),
                    corr_heldout_nll_vs_pipeline_eval_loss=float(np.corrcoef(fits.heldout_nll, fits.pipeline_eval_pred_loss)[0, 1]),
                    softmax_figure_example=softmax_example, inputs_sha256={k: sha(v) for k, v in inputs.items() if Path(v).exists()}, script_sha256=sha(Path(__file__)))
    write_json(tables / 'manifest.json', manifest)
    print(out['summary'].to_string()); print(out['examples'][KEY].to_string())


if __name__ == '__main__':
    main()
