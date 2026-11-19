"""Consolidated mechanistic-variability appendix analyses.

Includes descriptive/hierarchical variance, optional inclusion evidence, factorial
contrasts and publication orchestration. Plotting lives in code/figures.
Default execution reuses saved posterior draws; --sections evidence explicitly
requests nested sampling. No neural training or model reselection occurs here.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
REPO=ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO.parent))
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtri
from scipy.linalg import helmert, cho_factor, cho_solve
from scipy.stats import f as f_dist, t as t_dist, binomtest, wilcoxon, false_discovery_control, ttest_1samp
from threadpoolctl import threadpool_limits
from NM_TinyRNN.code.analyses import mechanistic_variability_main as study
from NM_TinyRNN.code.measures import mechanistic_variability as measures

# Descriptive and hierarchical variance

MODELS = {'GRU': '2_unit_GRU_tanh_unipolar_forced',
          'monoGRU': '2_unit_monoGRU_relu_unipolar_forced_ndb'}
METRICS = {'weight_similarity': 'Weights', 'logit_value_similarity': 'Logits',
           'hidden_similarity': 'Hidden state', 'gate_update_similarity': 'Update gate'}
COMPONENTS = ['subject', 'subject_pair', 'fitted_common', 'fitted_role', 'pair_residual']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decomposition(values, groups):
    """Exact population-variance identity, with observation-weighted group means."""
    d = pd.DataFrame({'value': values, 'group': groups})
    assert d.value.notna().all() and len(d)
    g = d.groupby('group').value.agg(['mean', 'size', 'var'])
    total = float(np.var(d.value, ddof=0))
    between = float(np.average((g['mean'] - d.value.mean())**2, weights=g['size']))
    fitted = d.group.map(g['mean'])
    within = float(np.mean((d.value - fitted)**2))
    np.testing.assert_allclose(total, between + within, rtol=1e-10, atol=1e-14)
    return dict(n_pairs=len(d), n_groups=len(g), mean=d.value.mean(),
                total_variance=total, between_group_variance=between,
                within_group_variance=within,
                between_group_fraction=between / total if total else np.nan,
                within_group_fraction=within / total if total else np.nan)


def endpoint_decomposition(d, metric, within):
    """Orthogonal finite-sample model-endpoint/interaction decomposition.

    Within: complete undirected graph, least-squares y_ij=mean+u_i+u_j+e_ij.
    Across: complete bipartite matrix, y_ij=mean+row_i+column_j+interaction_ij.
    This is a descriptive projection, not an unbiased random-effects estimator.
    """
    y = d[metric].to_numpy()
    centered = y - y.mean()
    if within:
        nodes = sorted(set(d.model_A_outer_n) | set(d.model_B_outer_n))
        n = len(nodes)
        means = {i: d.loc[d.model_A_outer_n.eq(i) | d.model_B_outer_n.eq(i), metric].mean() for i in nodes}
        u = {i: (n - 1) / (n - 2) * (means[i] - y.mean()) for i in nodes}
        predicted = d.model_A_outer_n.map(u).to_numpy() + d.model_B_outer_n.map(u).to_numpy()
        reference = candidate = np.nan
    else:
        row = d.groupby('model_A_outer_n')[metric].mean() - y.mean()
        col = d.groupby('model_B_outer_n')[metric].mean() - y.mean()
        a = d.model_A_outer_n.map(row).to_numpy()
        b = d.model_B_outer_n.map(col).to_numpy()
        predicted = a + b
        reference, candidate = float(np.mean(a*a)), float(np.mean(b*b))
    endpoint = float(np.mean(predicted**2))
    interaction = float(np.mean((centered - predicted)**2))
    np.testing.assert_allclose(np.mean(centered**2), endpoint + interaction, rtol=1e-9, atol=1e-14)
    return dict(endpoint_variance=endpoint, interaction_variance=interaction,
                reference_endpoint_variance=reference, candidate_endpoint_variance=candidate)


def load_inputs(source):
    paths = [source / f for f in ['pairwise_similarities.htsv', 'selected_models.htsv', 'manifest.json']]
    hashes = {p.name: sha(p) for p in paths}
    manifest = json.loads(paths[2].read_text())
    if manifest['folder'] != 'nested_DA_S32_B8':
        raise ValueError('This analysis requires nested_DA_S32_B8')
    pairs = pd.read_csv(paths[0], sep='\t')
    selected = pd.read_csv(paths[1], sep='\t')
    selected = selected[selected.model_id.isin(MODELS.values())].copy()
    pairs = pairs[pairs.model_id.isin(MODELS.values())].copy()
    assert selected.selection_method.eq('block_reset').all()
    assert selected.save_path.str.contains('/nested_DA_S32_B8/').all()
    assert pairs.trial_subject.eq(pairs.subject_A).all()
    assert pairs[list(METRICS)].notna().all().all()
    assert np.isfinite(pairs[list(METRICS)].to_numpy()).all()
    assert (pairs[list(METRICS)].abs() <= 1 + 1e-10).all().all()
    assert len(selected) == 220 and len(pairs) == 11990
    for endpoint in ['A', 'B']:
        match = pairs.merge(selected[['model_id', 'subject_id', 'outer_loop_n', 'inner_loop_idx']],
                            left_on=['model_id', f'subject_{endpoint}', f'model_{endpoint}_outer_n'],
                            right_on=['model_id', 'subject_id', 'outer_loop_n'], validate='many_to_one')
        assert len(match) == len(pairs)
        assert match[f'model_{endpoint}_inner_n'].eq(match.inner_loop_idx).all()
    for _, g in pairs.groupby(['model_id', 'within_subject']):
        n = 45 if g.within_subject.iloc[0] else 100
        assert g.groupby(['subject_A', 'subject_B']).size().eq(n).all()
        assert not g.duplicated(['subject_A', 'subject_B', 'model_A_outer_n', 'model_B_outer_n']).any()
    for p in paths:
        assert sha(p) == hashes[p.name], 'Source changed while loading'
    return pairs, selected, dict(source=str(source), input_sha256=hashes,
                                source_manifest=manifest)


def descriptive(pairs, selected, out):
    rows, group_rows, fold_rows = [], [], []
    for name, model_id in MODELS.items():
        for within in [True, False]:
            d = pairs[pairs.model_id.eq(model_id) & pairs.within_subject.eq(within)]
            groups = d.subject_A + ':' + d.subject_B
            for metric in METRICS:
                row = dict(model=name, metric=metric, within_subject=within,
                           **decomposition(d[metric].to_numpy(), groups.to_numpy()))
                deletion = []
                for subject in sorted(set(d.subject_A) | set(d.subject_B)):
                    keep = ~d.subject_A.eq(subject) & ~d.subject_B.eq(subject)
                    deletion.append(decomposition(d.loc[keep, metric].to_numpy(), groups[keep].to_numpy())['between_group_fraction'])
                row.update(between_fraction_subject_deletion_min=min(deletion),
                           between_fraction_subject_deletion_max=max(deletion))
                rows.append(row)
                endpoint_parts = []
                for (a, b), g in d.groupby(['subject_A', 'subject_B']):
                    part = endpoint_decomposition(g, metric, within)
                    endpoint_parts.append((len(g), part))
                    group_rows.append(dict(model=name, metric=metric, within_subject=within,
                                           subject_A=a, subject_B=b, mean=g[metric].mean(),
                                           sd=g[metric].std(ddof=0), n=len(g), **part))
                for component in ['endpoint_variance', 'interaction_variance',
                                  'reference_endpoint_variance', 'candidate_endpoint_variance']:
                    row[component] = np.average([part[component] for _, part in endpoint_parts],
                                                weights=[n for n, _ in endpoint_parts])
                row['endpoint_fraction'] = row['endpoint_variance'] / row['total_variance']
                row['interaction_fraction'] = row['interaction_variance'] / row['total_variance']
                np.testing.assert_allclose(row['between_group_fraction'] + row['endpoint_fraction'] + row['interaction_fraction'], 1.)
                # Role-specific summaries are descriptive, not independent replicates.
                for role, other in [('A', 'B'), ('B', 'A')]:
                    for (subject, fold), g in d.groupby([f'subject_{role}', f'model_{role}_outer_n']):
                        fold_rows.append(dict(model=name, model_id=model_id, metric=metric,
                                              within_subject=within, role=role, subject_id=subject,
                                              outer_loop_n=fold, mean_similarity=g[metric].mean(), n=len(g)))
    pd.DataFrame(rows).to_csv(out / 'descriptive_variance.csv', index=False)
    pd.DataFrame(group_rows).to_csv(out / 'group_similarities.csv', index=False)
    pd.DataFrame(fold_rows).merge(selected[['model_id', 'subject_id', 'outer_loop_n', 'eval_CE',
                                           'block_train_val_CE', 'weight_seed', 'energy_lambda', 'sparsity_lambda']],
                               validate='many_to_one').to_csv(out / 'fitted_model_role_summaries.csv', index=False)
    return pd.DataFrame(rows)


def design(d):
    """Endpoint effects have the same identity whichever role a model occupies."""
    subjects = sorted(set(d.subject_A) | set(d.subject_B))
    fits = sorted(set(zip(d.subject_A, d.model_A_outer_n)) | set(zip(d.subject_B, d.model_B_outer_n)))
    dyads = sorted(set(zip(d.subject_A, d.subject_B)))
    si, fi, di = ({x: i for i, x in enumerate(v)} for v in [subjects, fits, dyads])
    return dict(sa=np.array([si[x] for x in d.subject_A]), sb=np.array([si[x] for x in d.subject_B]),
                fa=np.array([fi[x] for x in zip(d.subject_A, d.model_A_outer_n)]),
                fb=np.array([fi[x] for x in zip(d.subject_B, d.model_B_outer_n)]),
                dyad=np.array([di[x] for x in zip(d.subject_A, d.subject_B)]),
                subjects=subjects, fits=fits, dyads=dyads)


def reference_matched_contrasts(pairs,out):
    """Within-A minus B-to-A similarity, using only the saved A-reference direction."""
    assert pairs.trial_subject.eq(pairs.subject_A).all()
    d=pairs.copy()
    d['model']=d.model_id.map({value:key for key,value in MODELS.items()})
    assert d.model.notna().all()
    long=d.melt(id_vars=['model','subject_A','subject_B','within_subject'],
                value_vars=list(METRICS),var_name='metric',value_name='similarity')
    groups=long.groupby(['model','metric','subject_A','subject_B','within_subject']).similarity.agg(['mean','size']).reset_index()
    baseline=groups[groups.within_subject][['model','metric','subject_A','mean','size']].rename(columns={'mean':'within_similarity','size':'n_within_pairs'})
    across=groups[~groups.within_subject].rename(columns={'mean':'across_similarity','size':'n_across_pairs'})
    contrasts=across.merge(baseline,on=['model','metric','subject_A'],validate='many_to_one')
    assert len(contrasts)==len(across), 'A saved reference lacks its within-subject baseline'
    contrasts['difference']=contrasts.within_similarity-contrasts.across_similarity
    contrasts=contrasts.rename(columns={'subject_A':'reference_subject','subject_B':'comparison_subject'}).drop(columns='within_subject')
    contrasts.to_csv(out/'reference_matched_contrasts.csv',index=False)
    summaries=[];deletions=[]
    for (model,metric),g in contrasts.groupby(['model','metric']):
        subjects=sorted(set(g.reference_subject)|set(g.comparison_subject))
        values=[]
        for subject in subjects:
            kept=g[~g.reference_subject.eq(subject)&~g.comparison_subject.eq(subject)]
            if not len(kept):continue
            value=float(kept.difference.mean());values.append(value)
            deletions.append(dict(model=model,metric=metric,deleted_subject=subject,
                                  mean_difference=value,n_subject_pairs=len(kept)))
        summaries.append(dict(model=model,metric=metric,n_subject_pairs=len(g),n_subjects=len(subjects),
                              n_reference_subjects=g.reference_subject.nunique(),
                              within_similarity=g.within_similarity.mean(),across_similarity=g.across_similarity.mean(),
                              mean_difference=g.difference.mean(),median_difference=g.difference.median(),
                              n_positive=int(g.difference.gt(0).sum()),
                              deletion_min=min(values) if values else np.nan,
                              deletion_max=max(values) if values else np.nan))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'reference_matched_summary.csv',index=False)
    pd.DataFrame(deletions).to_csv(out/'reference_matched_subject_deletions.csv',index=False)
    (out/'reference_matched_methods.json').write_text(json.dumps(dict(
        contrast='mean within-A similarity minus mean B-to-A similarity; both evaluated on A saved inputs',
        direction='saved subject_A reference only; no reverse evaluation or symmetrisation',
        averaging='equal weight per saved subject pair; reference subjects are not equally represented',
        uncertainty='subject-deletion min/max, not confidence intervals; no independent-pair test',
        interpretation='difference in mean similarities, not an additional variance component'),indent=2))
    return contrasts,summary


def contribution_multipliers(within):
    # Variance for a randomly sampled score under independent population effects.
    # Within: one subject effect, not a_s+a_s; across: two distinct subjects.
    return np.array([1 if within else 2, 0 if within else 1, 2, 2, 1])


def group_variance_budget(groups):
    """Reconstruct the exact raw variance budget from complete group summaries."""
    w=groups.n.to_numpy();means=groups['mean'].to_numpy()
    between=float(np.average((means-np.average(means,weights=w))**2,weights=w))
    endpoint=float(np.average(groups.endpoint_variance,weights=w))
    interaction=float(np.average(groups.interaction_variance,weights=w))
    return dict(between_subjects=between,outer_fold=endpoint,model_pair=interaction,
                total=between+endpoint+interaction)


def robust_variance_analysis(pairs,out):
    """Subject-level median/MAD, direction-matched median gaps and deletion stability."""
    from scipy.stats import binomtest,false_discovery_control
    robust=out/'robust';robust.mkdir(exist_ok=True)
    d=pairs.copy();d['model']=d.model_id.map({value:key for key,value in MODELS.items()})
    long=d.melt(id_vars=['model','subject_A','subject_B','within_subject'],value_vars=list(METRICS),
                var_name='metric',value_name='similarity')
    subjects=[]
    for (model,metric,subject),g in long[long.within_subject].groupby(['model','metric','subject_A']):
        median=float(g.similarity.median())
        subjects.append(dict(model=model,metric=metric,subject_id=subject,median_similarity=median,
                             mad_similarity=float(np.median(np.abs(g.similarity-median))),n_pairs=len(g)))
    subjects=pd.DataFrame(subjects);subjects.to_csv(robust/'within_subject_summaries.csv',index=False)
    tests=[]
    for metric in METRICS:
        for statistic in ['median_similarity','mad_similarity']:
            wide=subjects[subjects.metric.eq(metric)].pivot(index='subject_id',columns='model',values=statistic)
            diff=(wide.monoGRU-wide.GRU).to_numpy();positive=int(np.sum(diff>1e-12));negative=int(np.sum(diff < -1e-12))
            n=positive+negative
            tests.append(dict(metric=metric,statistic=statistic,n_subjects=len(diff),n_positive=positive,
                              n_negative=negative,n_ties=len(diff)-n,median_difference=float(np.median(diff)),
                              gru_median=float(wide.GRU.median()),mono_median=float(wide.monoGRU.median()),
                              p_sign=float(binomtest(positive,n,p=.5).pvalue) if n else 1.))
    tests=pd.DataFrame(tests);tests['q_bh8']=false_discovery_control(tests.p_sign,method='bh')
    tests.to_csv(robust/'within_subject_sign_tests.csv',index=False)
    groups=long.groupby(['model','metric','subject_A','subject_B','within_subject']).similarity.agg(['median','mean']).reset_index()
    base=groups[groups.within_subject][['model','metric','subject_A','median','mean']].rename(columns={'median':'within_median','mean':'within_mean'})
    across=groups[~groups.within_subject].rename(columns={'median':'across_median','mean':'across_mean'})
    gaps=across.merge(base,on=['model','metric','subject_A'],validate='many_to_one').drop(columns='within_subject')
    gaps['median_based_gap']=gaps.within_median-gaps.across_median
    gaps['mean_based_gap']=gaps.within_mean-gaps.across_mean
    gaps.to_csv(robust/'subject_specificity_pairs.csv',index=False)
    gap_summary=[];gap_deletions=[]
    for (model,metric),g in gaps.groupby(['model','metric']):
        for estimator in ['median_based_gap','mean_based_gap']:
            values=[]
            for subject in sorted(set(g.subject_A)|set(g.subject_B)):
                kept=g[~g.subject_A.eq(subject)&~g.subject_B.eq(subject)]
                value=float(kept[estimator].median());values.append(value)
                gap_deletions.append(dict(model=model,metric=metric,estimator=estimator,deleted_subject=subject,
                                          median_gap=value,n_subject_pairs=len(kept)))
            gap_summary.append(dict(model=model,metric=metric,estimator=estimator,median_gap=float(g[estimator].median()),
                                    deletion_min=min(values),deletion_max=max(values),n_positive=int(g[estimator].gt(0).sum()),
                                    n_subject_pairs=len(g)))
    pd.DataFrame(gap_summary).to_csv(robust/'subject_specificity_summary.csv',index=False)
    pd.DataFrame(gap_deletions).to_csv(robust/'subject_specificity_deletions.csv',index=False)
    group_data=pd.read_csv(out/'group_similarities.csv');original=pd.read_csv(out/'descriptive_variance.csv')
    budget_ranges=[];budget_deletions=[]
    for (model,metric,within),g in group_data.groupby(['model','metric','within_subject']):
        full=group_variance_budget(g)
        expected=original[original.model.eq(model)&original.metric.eq(metric)&original.within_subject.eq(within)].iloc[0]
        np.testing.assert_allclose([full['between_subjects'],full['outer_fold'],full['model_pair']],
                                   [expected.between_group_variance,expected.endpoint_variance,expected.interaction_variance],rtol=1e-9,atol=1e-14)
        deleted=[]
        for subject in sorted(set(g.subject_A)|set(g.subject_B)):
            kept=g[~g.subject_A.eq(subject)&~g.subject_B.eq(subject)]
            estimate=group_variance_budget(kept);deleted.append(estimate)
            for component in ['between_subjects','outer_fold','model_pair']:
                budget_deletions.append(dict(model=model,metric=metric,within_subject=within,deleted_subject=subject,
                                             component=component,variance=estimate[component],fraction=estimate[component]/estimate['total']))
        for component in ['between_subjects','outer_fold','model_pair']:
            fractions=[x[component]/x['total'] for x in deleted];absolute=[x[component] for x in deleted]
            budget_ranges.append(dict(model=model,metric=metric,within_subject=within,component=component,
                                      variance=full[component],fraction=full[component]/full['total'],
                                      variance_min=min(absolute),variance_max=max(absolute),
                                      fraction_min=min(fractions),fraction_max=max(fractions)))
    pd.DataFrame(budget_ranges).to_csv(robust/'variance_deletion_ranges.csv',index=False)
    pd.DataFrame(budget_deletions).to_csv(robust/'variance_subject_deletions.csv',index=False)
    (robust/'methods.json').write_text(json.dumps(dict(
        analysis_code_sha256=sha(__file__),
        input_sha256=json.loads((out/'manifest.json').read_text()).get('input_sha256',{}),
        dataset='nested_DA_S32_B8; block-reset-selected outer winners; cached scores only',
        within='one median and raw unscaled MAD of 45 similarities per subject/model/measure',
        tests='exact two-sided sign tests of monoGRU-GRU subject differences; ties abs<=1e-12 removed; separate BH family of 8 (4 metrics x median/MAD)',
        specificity='primary median(within-A scores)-median(B-to-A scores), summarised by median over 55 saved dyads; mean-based gaps retained for sensitivity',
        across_inference='no independent-dyad p-values; no reversed comparisons; all incident dyads removed on subject deletion',
        variance='exact finite raw-scale squared-variation decomposition, not latent population variance estimates',
        ranges='min/max after deleting each subject; descriptive sensitivity ranges, not confidence intervals',
        limitations='11 subjects, overlapping fits, one direction and uneven references; low MAD alone does not establish high similarity'),indent=2))


def across_sufficient_statistics(d, metric):
    """Lossless orthogonal decomposition of balanced dyad score matrices.

    Integrating Gaussian effects reduces the across sampler to just five SDs.
    Grand means, endpoint contrasts and interactions are independent blocks.
    """
    x = design(d)
    subjects = x['subjects']
    n = len({fold for _, fold in x['fits']})
    q = helmert(n).T
    b = np.zeros((len(x['dyads']), len(subjects)))
    role = b.copy()
    endpoint = {s: [] for s in subjects}
    signs = {s: [] for s in subjects}
    mean = []
    ss = 0.
    for k, (a, c) in enumerate(x['dyads']):
        g = d[d.subject_A.eq(a) & d.subject_B.eq(c)]
        matrix = np.arctanh(np.clip(g.pivot(index='model_A_outer_n', columns='model_B_outer_n', values=metric).to_numpy(), -1+1e-7, 1-1e-7))
        assert matrix.shape == (n, n)
        row = np.sqrt(n) * q.T @ matrix.mean(axis=1)
        col = np.sqrt(n) * q.T @ matrix.mean(axis=0)
        mean.append(n * matrix.mean())
        interaction = q.T @ matrix @ q
        ss += float(np.sum(interaction**2))
        ia, ic = subjects.index(a), subjects.index(c)
        b[k, [ia, ic]] = 1
        role[k, ia], role[k, ic] = 1, -1
        endpoint[a].append(row)
        endpoint[c].append(col)
        signs[a].append(1.)
        signs[c].append(-1.)
    return dict(n=n, q=q, subjects=subjects, dyads=x['dyads'], b=b, role=role,
                mean=np.array(mean), endpoint=[np.array(endpoint[s]) for s in subjects],
                signs=[np.array(signs[s]) for s in subjects], interaction_ss=ss,
                interaction_df=len(x['dyads'])*(n-1)**2, design=x)


def collapsed_covariances(s, sd):
    n = s['n']
    a, c, u, v, e = sd**2
    m = n*n*((a+u/n)*(s['b']@s['b'].T) + v/n*(s['role']@s['role'].T) + 9*np.ones((len(s['mean']),len(s['mean']))))
    m += (n*n*c+e)*np.eye(len(s['mean']))
    endpoint = [n*(u*np.ones((len(sign),len(sign))) + v*np.outer(sign,sign)) + e*np.eye(len(sign)) for sign in s['signs']]
    return m, endpoint


def collapsed_logpdf(s, sd):
    """Reference implementation for checking the exact marginal likelihood."""
    m, endpoints = collapsed_covariances(s, sd)
    def lognormal(y, cov):
        cf = cho_factor(cov, lower=True)
        k = 1 if y.ndim == 1 else y.shape[1]
        return -.5*(k*len(y)*np.log(2*np.pi) + k*2*np.log(np.diag(cf[0])).sum() + np.sum(y*cho_solve(cf,y)))
    result = lognormal(s['mean'], m)
    for r, cov in zip(s['endpoint'], endpoints):
        result += lognormal(r, cov)
    result += -.5*(s['interaction_df']*np.log(2*np.pi*sd[4]**2) + s['interaction_ss']/sd[4]**2)
    return float(result)


def conditional_across_predictions(s, d, sd, rng):
    """Conditional Gaussian effects via exact normal conditioning; one draw/SD draw."""
    n, ns, nd = s['n'], len(s['subjects']), len(s['mean'])
    predictions, latent = [], []
    x = s['design']
    for sigma in sd:
        a, c, u, v, e = sigma
        # Mean-block effects: intercept, subject, average u/v, dyad.
        L = np.column_stack([np.ones(nd)*n, n*s['b'], n*s['b'], n*s['role'], n*np.eye(nd)])
        prior_sd = np.r_[3., np.repeat(a,ns), np.repeat(u/np.sqrt(n),ns), np.repeat(v/np.sqrt(n),ns), np.repeat(c,nd)]
        A = L * prior_sd
        cf = cho_factor(A@A.T + e*e*np.eye(nd), lower=True)
        z = rng.normal(size=len(prior_sd))
        z += A.T @ cho_solve(cf, s['mean'] - A@z - e*rng.normal(size=nd))
        effects = z*prior_sd
        mu = effects[0]
        subject = effects[1:1+ns]
        common_mean = effects[1+ns:1+2*ns]
        role_mean = effects[1+2*ns:1+3*ns]
        dyad = effects[1+3*ns:]
        common, roles = [], []
        for k, sign in enumerate(s['signs']):
            X = np.sqrt(n)*np.column_stack([np.ones(len(sign)),sign])
            cov = np.linalg.inv(np.diag([1/u**2,1/v**2]) + X.T@X/e**2)
            center = cov@X.T@s['endpoint'][k]/e**2
            draws = center + np.linalg.cholesky(cov)@rng.normal(size=(2,n-1))
            common.extend(common_mean[k]+s['q']@draws[0])
            roles.extend(role_mean[k]+s['q']@draws[1])
        common, roles = np.array(common), np.array(roles)
        eta = mu+subject[x['sa']]+subject[x['sb']]+dyad[x['dyad']]+common[x['fa']]+common[x['fb']]+roles[x['fa']]-roles[x['fb']]
        predictions.append(np.tanh(eta+e*rng.normal(size=len(d))))
        latent.append(mu)
    return np.array(predictions), np.array(latent)


def predictive_checks(observed, pred, groups):
    functions = {'mean': np.mean, 'sd': np.std,
                 'q05': lambda v: np.quantile(v,.05), 'q95': lambda v: np.quantile(v,.95),
                 'skewness': lambda v: float(np.mean(((v-np.mean(v))/np.std(v))**3)),
                 'between_group_fraction': lambda v: decomposition(v,groups)['between_group_fraction']}
    checks = []
    for label, fun in functions.items():
        obs = float(fun(observed))
        sim = np.array([fun(v) for v in pred])
        lo, hi = np.quantile(sim,[.025,.975])
        checks.append(dict(check=label, observed=obs, predictive_lo=lo, predictive_hi=hi,
                           predictive_tail_probability=float(np.mean(sim>=obs)), inside_95=bool(lo<=obs<=hi)))
    return checks


def sample_across_collapsed(d, metric, args, path, target_accept=.95):
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    s = across_sufficient_statistics(d, metric)
    with pm.Model(coords={'component':COMPONENTS}) as model:
        sd = pm.HalfNormal('sd',1.,dims='component')
        a,c,u,v,e = [sd[i]**2 for i in range(5)]
        n, nd = s['n'],len(s['mean'])
        cov = n*n*((a+u/n)*(s['b']@s['b'].T) + v/n*(s['role']@s['role'].T) + 9*np.ones((nd,nd))) + (n*n*c+e)*np.eye(nd)
        pm.MvNormal('means',mu=np.zeros(nd),cov=cov,observed=s['mean'])
        for k,(sign,r) in enumerate(zip(s['signs'],s['endpoint'])):
            cov = n*(u*np.ones((len(sign),len(sign)))+v*np.outer(sign,sign))+e*np.eye(len(sign))
            pm.MvNormal(f'endpoint_{k}',mu=np.zeros(len(sign)),cov=cov,observed=r.T)
        pm.Potential('interaction',-.5*(s['interaction_df']*pt.log(2*np.pi*e)+s['interaction_ss']/e))
        trace = pm.sample(draws=args.draws,tune=args.tune,chains=args.chains,cores=args.cores,
                          random_seed=args.seed,target_accept=target_accept,progressbar=False,blas_cores=1)
    if int(trace.sample_stats.diverging.sum()) and target_accept<.99:
        return sample_across_collapsed(d,metric,args,path,.99)
    diagnostics = az.summary(trace,var_names=['sd'],kind='diagnostics')
    diag = dict(divergences=int(trace.sample_stats.diverging.sum()),max_rhat=float(diagnostics.r_hat.max()),
                min_ess_bulk=float(diagnostics.ess_bulk.min()),min_ess_tail=float(diagnostics.ess_tail.min()),
                max_tree_depth_hits=int(trace.sample_stats.reached_max_treedepth.sum()),
                clipped_scores=int((d[metric].abs()>=1-1e-7).sum()),target_accept=target_accept,
                sampler='exact Gaussian marginalisation; balanced orthogonal blocks')
    diag['sampling_ok'] = bool(diag['divergences']==0 and diag['max_rhat']<=1.01 and diag['min_ess_bulk']>=400 and diag['max_tree_depth_hits']==0)
    sd = np.asarray(trace.posterior.sd).reshape(-1,5)
    rng = np.random.default_rng(args.seed+1)
    take = rng.choice(len(sd),min(400,len(sd)),replace=False)
    pred,mu = conditional_across_predictions(s,d,sd[take],rng)
    checks = predictive_checks(d[metric].to_numpy(),pred,(d.subject_A+':'+d.subject_B).to_numpy())
    np.savez_compressed(path,sd=sd,active=np.array(COMPONENTS),diagnostics=json.dumps(diag),checks=json.dumps(checks),
                        posterior_sd=np.asarray(trace.posterior.sd),sample_diverging=np.asarray(trace.sample_stats.diverging),
                        predictive_draw_indices=take,conditional_mu=mu)
    return sd,diag,checks


def sample_model(d, metric, within, args, path, target_accept=.95):
    if not within:
        return sample_across_collapsed(d,metric,args,path,target_accept)
    import pymc as pm
    import arviz as az
    x = design(d)
    observed = d[metric].to_numpy()
    clipped = np.clip(observed, -1 + 1e-7, 1 - 1e-7)
    y = np.arctanh(clipped)
    active = [c for c in COMPONENTS if c != 'subject_pair' or not within]
    coords = {'component': active, 'subject': x['subjects'], 'fit': np.arange(len(x['fits'])),
              'dyad': np.arange(len(x['dyads']))}
    with pm.Model(coords=coords) as model:
        sd = pm.HalfNormal('sd', sigma=1., dims='component')
        scales = {c: sd[i] for i, c in enumerate(active)}
        mu = pm.Normal('mu', mu=0, sigma=3)
        subject = pm.Normal('subject_z', 0, 1, dims='subject') * scales['subject']
        if within:
            common = pm.Normal('common_z', 0, 1, dims='fit') * scales['fitted_common']
            role = pm.Normal('role_z', 0, 1, dims='fit') * scales['fitted_role']
        else:
            # With ~100 observations per endpoint, centred effects avoid the
            # strong SD/effect correlations of a non-centred parameterisation.
            common = pm.Normal('common_effect', 0, scales['fitted_common'], dims='fit')
            role = pm.Normal('role_effect', 0, scales['fitted_role'], dims='fit')
            pm.Deterministic('common_z', common / scales['fitted_common'], dims='fit')
            pm.Deterministic('role_z', role / scales['fitted_role'], dims='fit')
        eta = mu + subject[x['sa']] + common[x['fa']] + common[x['fb']] + role[x['fa']] - role[x['fb']]
        if not within:
            dyad = pm.Normal('dyad_effect', 0, scales['subject_pair'], dims='dyad')
            pm.Deterministic('dyad_z', dyad / scales['subject_pair'], dims='dyad')
            eta = eta + subject[x['sb']] + dyad[x['dyad']]
        pm.Normal('score', mu=eta, sigma=scales['pair_residual'], observed=y)
        trace = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                          random_seed=args.seed, target_accept=target_accept, progressbar=False,
                          blas_cores=1, init='jitter+adapt_diag' if within else 'jitter+adapt_full')
    if int(trace.sample_stats.diverging.sum()) and target_accept < .99:
        print('Retrying divergent fit with target_accept=.99', flush=True)
        return sample_model(d, metric, within, args, path, target_accept=.99)
    # Keep a portable numpy posterior, not thousands of trial-level draws in git.
    post = trace.posterior
    sd_draws = np.asarray(post['sd']).reshape(-1, len(active))
    full_sd = np.zeros((len(sd_draws), len(COMPONENTS)))
    full_sd[:, [COMPONENTS.index(c) for c in active]] = sd_draws
    # Conditional predictive checks preserve subject/fit identities and orientation.
    rng = np.random.default_rng(args.seed + 1)
    n = min(400, len(sd_draws))
    take = rng.choice(len(sd_draws), n, replace=False)
    def draws(var):
        a = np.asarray(post[var])
        return a.reshape((-1,) + a.shape[2:])[take]
    scales = full_sd[take]
    subj = draws('subject_z') * scales[:, 0, None]
    common = draws('common_z') * scales[:, 2, None]
    role = draws('role_z') * scales[:, 3, None]
    pred = draws('mu')[:, None] + subj[:, x['sa']] + common[:, x['fa']] + common[:, x['fb']] + role[:, x['fa']] - role[:, x['fb']]
    if not within:
        pred += subj[:, x['sb']] + draws('dyad_z')[:, x['dyad']] * scales[:, 1, None]
    pred = np.tanh(pred + rng.normal(size=pred.shape) * scales[:, 4, None])
    groups = d.subject_A + ':' + d.subject_B
    checks = []
    functions = {'mean': lambda v: np.mean(v), 'sd': lambda v: np.std(v),
                 'q05': lambda v: np.quantile(v, .05), 'q95': lambda v: np.quantile(v, .95),
                 'skewness': lambda v: float(np.mean(((v - np.mean(v)) / np.std(v))**3)),
                 'between_group_fraction': lambda v: decomposition(v, groups.to_numpy())['between_group_fraction']}
    for label, fun in functions.items():
        obs = fun(observed)
        simulated = np.array([fun(v) for v in pred])
        lo, hi = np.quantile(simulated, [.025, .975])
        checks.append(dict(check=label, observed=obs, predictive_lo=lo, predictive_hi=hi,
                           predictive_tail_probability=float(np.mean(simulated >= obs)), inside_95=bool(lo <= obs <= hi)))
    diagnostics = az.summary(trace, var_names=['sd', 'mu', 'subject_z', 'common_z', 'role_z'] + ([] if within else ['dyad_z']), kind='diagnostics')
    divergences = int(trace.sample_stats.diverging.sum())
    diag = dict(divergences=divergences, max_rhat=float(diagnostics['r_hat'].max()),
                min_ess_bulk=float(diagnostics['ess_bulk'].min()), min_ess_tail=float(diagnostics['ess_tail'].min()),
                clipped_scores=int(np.sum(observed != clipped)), target_accept=target_accept,
                max_tree_depth_hits=int(trace.sample_stats.reached_max_treedepth.sum()),
                sampling_ok=bool(divergences == 0 and diagnostics['r_hat'].max() <= 1.01 and diagnostics['ess_bulk'].min() >= 400
                                 and int(trace.sample_stats.reached_max_treedepth.sum()) == 0))
    saved_variables = {f'posterior_{v}': np.asarray(post[v]) for v in post.data_vars}
    np.savez_compressed(path, sd=full_sd, active=np.array(active),
                        diagnostics=json.dumps(diag), checks=json.dumps(checks),
                        sample_diverging=np.asarray(trace.sample_stats.diverging), **saved_variables)
    return full_sd, diag, checks


def prior_samples(within, scale, seed, n=200000):
    rng = np.random.default_rng(seed)
    sd = np.abs(rng.normal(0, scale, (n, 5)))
    if within:
        sd[:, 1] = 0
    variance = sd**2 * contribution_multipliers(within)
    return variance / variance.sum(axis=1, keepdims=True)


def importance_weights(sd, within, prior_scale):
    active = sd[:, [0, 2, 3, 4]] if within else sd
    logw = -active.shape[1] * np.log(prior_scale) + .5 * (1 - prior_scale**-2) * (active**2).sum(axis=1)
    w = np.exp(logw - logsumexp(logw))
    return w, float(1 / (w @ w))


def weighted_quantile(x, w, q):
    idx = np.argsort(x)
    return np.interp(q, np.cumsum(w[idx]) - .5 * w[idx], x[idx])


def evidence(posterior_event, prior_event, w, ess):
    """Encompassing-prior BF: posterior odds / prior odds, not posterior odds alone.

    No infinite BF is reported when finite Monte Carlo samples miss one region.
    """
    post = float(w @ posterior_event)
    prior = float(np.mean(prior_event))
    resolved = min(post, 1 - post) * ess >= 20 and min(prior, 1 - prior) * len(prior_event) >= 20
    bf = post / (1 - post) / (prior / (1 - prior)) if resolved else np.nan
    return dict(posterior_probability=post, prior_probability=prior, bayes_factor=bf,
                bf_resolved=bool(resolved), bf_note='finite Monte Carlo odds' if resolved else 'tail probability unresolved; BF omitted')


def summarize_posteriors(posteriors, out, seed):
    summaries, events, diagnostics, checks, comparisons = [], [], [], [], []
    primary = {}
    for (name, metric, within), (sd, diag, ppc) in posteriors.items():
        tags = dict(model=name, metric=metric, within_subject=within,
                    predictive_checks_outside_95=sum(not check['inside_95'] for check in ppc))
        diagnostics.append(dict(**tags, **diag))
        checks += [dict(**tags, **v) for v in ppc]
        variance = sd**2 * contribution_multipliers(within)
        fraction = variance / variance.sum(axis=1, keepdims=True)
        primary[name, metric, within] = fraction
        for scale in [.5, 1., 2.]:
            w, ess = importance_weights(sd, within, scale)
            reliable = ess >= max(400, .2 * len(w))
            evidence_ess = min(ess,diag['min_ess_bulk'],diag['min_ess_tail'])
            prior = prior_samples(within, scale, seed)
            for i, component in enumerate(COMPONENTS):
                if within and component == 'subject_pair':
                    continue
                lo, median, hi = weighted_quantile(fraction[:, i], w, [.025, .5, .975])
                summaries.append(dict(**tags, component=component, prior_scale=scale,
                                      fraction_mean=float(w @ fraction[:, i]), fraction_median=median,
                                      fraction_lo=lo, fraction_hi=hi,
                                      variance_median=weighted_quantile(variance[:, i], w, [.5])[0],
                                      importance_ess=ess, importance_ok=bool(reliable), sampling_ok=diag['sampling_ok']))
            fit = fraction[:, 2:4].sum(axis=1)
            pfit = prior[:, 2:4].sum(axis=1)
            for component, observed, baseline in [('fitted_solution', fit, pfit), ('subject', fraction[:, 0], prior[:, 0])]:
                for threshold in [.1, .25, .5]:
                    e = evidence(observed > threshold, baseline > threshold, w, evidence_ess)
                    if not reliable or not diag['sampling_ok']:
                        e.update(bayes_factor=np.nan,bf_resolved=False,bf_note='sampling or importance diagnostics failed')
                    events.append(dict(**tags, prior_scale=scale, hypothesis=f'{component}_fraction > {threshold}',
                                       importance_ess=ess, importance_ok=bool(reliable), sampling_ok=diag['sampling_ok'], **e))
            e = evidence(fraction[:, 0] > fit, prior[:, 0] > pfit, w, evidence_ess)
            if not reliable or not diag['sampling_ok']:
                e.update(bayes_factor=np.nan,bf_resolved=False,bf_note='sampling or importance diagnostics failed')
            events.append(dict(**tags, prior_scale=scale, hypothesis='subject_fraction > fitted_solution_fraction',
                               importance_ess=ess, importance_ok=bool(reliable), sampling_ok=diag['sampling_ok'], **e))
    # Independent architecture fits: compare draws via independent random pairing.
    rng = np.random.default_rng(seed + 3)
    for metric in METRICS:
        for within in [True, False]:
            keys = [(name, metric, within) for name in MODELS]
            if not all(key in posteriors for key in keys):
                continue
            for scale in [.5, 1., 2.]:
                samples, ratios, good = [], [], True
                for key in keys:
                    sd = posteriors[key][0]
                    w, ess = importance_weights(sd, within, scale)
                    good &= ess >= max(400, .2 * len(w)) and posteriors[key][1]['sampling_ok']
                    take = rng.choice(len(sd), 20000, replace=True, p=w)
                    v = sd[take]**2 * contribution_multipliers(within)
                    samples.append(v[:, 2:4].sum(axis=1))
                    ratios.append((v[:, 2:4].sum(axis=1)) / v.sum(axis=1))
                ratio = samples[1] / samples[0]
                difference = ratios[1] - ratios[0]
                comparisons.append(dict(metric=metric, within_subject=within, prior_scale=scale,
                                        variance_ratio_mono_over_GRU=np.median(ratio),
                                        ratio_lo=np.quantile(ratio, .025), ratio_hi=np.quantile(ratio, .975),
                                        fraction_difference_mono_minus_GRU=np.median(difference),
                                        difference_lo=np.quantile(difference, .025), difference_hi=np.quantile(difference, .975),
                                        probability_mono_more_variable=float(np.mean(ratio > 1)),
                                        importance_and_sampling_ok=bool(good)))
    for name, rows in [('posterior_components', summaries), ('bayes_factors', events),
                       ('sampling_diagnostics', diagnostics), ('predictive_checks', checks),
                       ('architecture_comparisons', comparisons)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)


def run_variance(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/analyses/mechanistic_variability')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/analyses/mechanistic_variability/appendix/variance')
    parser.add_argument('--stage', choices=['all', 'descriptive', 'bayesian', 'plots', 'retry'], default='all')
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--metric', choices=list(METRICS), nargs='+', default=list(METRICS))
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = ROOT / 'results/appendix/mechanistic_variability/variance'
    figures.mkdir(parents=True, exist_ok=True)
    pairs, selected, provenance = load_inputs(args.source)
    settings = dict(draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
                    likelihood='Gaussian on atanh(similarity); not a correlation sampling-error model',
                    priors='mu Normal(0,3); component SD HalfNormal(1); independent standardized effects Normal(0,1)',
                    model='within mu+a_s+u_si+u_sj+v_si-v_sj+e; across mu+a_s+a_t+c_st+u_si+u_tj+v_si-v_tj+e',
                    sensitivity='importance reweight SD priors to HalfNormal(.5), HalfNormal(2); require ESS>=400 and >=20% draws',
                    bayes_factors='interval hypotheses via posterior odds/prior odds; no point-null variance BFs',
                    thresholds=[.1, .25, .5])
    fingerprint = hashlib.sha256(json.dumps(dict(inputs=provenance['input_sha256'], settings=settings,
                                                code_sha256=sha(__file__)), sort_keys=True).encode()).hexdigest()
    if args.stage in ['retry', 'plots']:
        previous = json.loads((out / 'manifest.json').read_text())
        assert previous['input_sha256'] == provenance['input_sha256'], 'Cannot retry changed inputs'
        fingerprint = previous['fingerprint']
        if args.stage == 'retry':
            (out / 'retry_settings.json').write_text(json.dumps(dict(draws=args.draws*2,tune=args.tune*2,seed=args.seed,chains=args.chains,
                 target_accept=.99,code_sha256=sha(__file__),policy='resample only fits failing saved sampling diagnostics'),indent=2))
    else:
        (out / 'manifest.json').write_text(json.dumps(dict(**provenance, settings=settings, fingerprint=fingerprint), indent=2))
    if args.stage in ['all', 'descriptive']:
        pairs.to_csv(out / 'input_pairs.csv', index=False)
        selected.to_csv(out / 'input_selected_models.csv', index=False)
        descriptive(pairs, selected, out)
    if args.stage in ['all', 'bayesian', 'retry']:
        cache = out / 'posteriors'
        cache.mkdir(exist_ok=True)
        results = {}
        for name, model_id in MODELS.items():
            for within in [True, False]:
                d = pairs[pairs.model_id.eq(model_id) & pairs.within_subject.eq(within)].reset_index(drop=True)
                for metric in args.metric:
                    key = name, metric, within
                    path = cache / f'{name}_{metric}_{"within" if within else "across"}_{fingerprint[:12]}.npz'
                    print(f'{key}: {"loading" if path.exists() else "sampling"}', flush=True)
                    if path.exists():
                        with np.load(path) as saved:
                            results[key] = saved['sd'], json.loads(str(saved['diagnostics'])), json.loads(str(saved['checks']))
                        if args.stage == 'retry' and not results[key][1]['sampling_ok']:
                            retry_args = argparse.Namespace(**vars(args))
                            retry_args.draws *= 2
                            retry_args.tune *= 2
                            print(f'{key}: retrying failed diagnostics with doubled warmup/draws',flush=True)
                            results[key] = sample_model(d,metric,within,retry_args,path,.99)
                    else:
                        results[key] = sample_model(d, metric, within, args, path)
        summarize_posteriors(results, out, args.seed)
    if args.stage in ['all', 'plots', 'descriptive', 'retry']:
        from NM_TinyRNN.code.figures.mechanistic_variability_variance import plot_results
        plot_results(out, figures)
    # Abort rather than silently attach results to an altered live input cache.
    for name, expected in provenance['input_sha256'].items():
        assert sha(args.source / name) == expected, 'Input changed during run; use saved snapshots for audit'
    print(f'Outputs: {out}\nFigures: {figures}', flush=True)

# Optional inclusion-versus-exclusion evidence

VARIANTS = {'full': [0,1,2,3,4], 'no_fitted_solution': [0,1,4],
            'no_additive_subject': [1,2,3,4], 'no_subject_or_dyad': [2,3,4]}


def span(matrix):
    u,s,_ = np.linalg.svd(matrix,full_matrices=False)
    return u[:,s>1e-10]


def prepare_likelihood(d, metric, within):
    """Orthogonal projections preserve likelihood constants and residual dimensions."""
    if within:
        matrices = []
        subjects = sorted(d.subject_A.unique())
        first = d[d.subject_A.eq(subjects[0])].sort_values(['model_A_outer_n','model_B_outer_n'])
        nodes = sorted(set(first.model_A_outer_n)|set(first.model_B_outer_n))
        index = {x:i for i,x in enumerate(nodes)}
        b = np.zeros((len(first),len(nodes))); role = b.copy()
        for k,row in enumerate(first.itertuples()):
            i,j=index[row.model_A_outer_n],index[row.model_B_outer_n]
            b[k,[i,j]]=1;role[k,i]=1;role[k,j]=-1
        for subject in subjects:
            g=d[d.subject_A.eq(subject)].sort_values(['model_A_outer_n','model_B_outer_n'])
            assert np.array_equal(g[['model_A_outer_n','model_B_outer_n']],first[['model_A_outer_n','model_B_outer_n']])
            matrices.append(np.arctanh(np.clip(g[metric].to_numpy(),-1+1e-7,1-1e-7)))
        z=np.array(matrices); q=span(np.column_stack([np.ones(len(first)),b,role]))
        y=z@q; one=q.T@np.ones(len(first)); B=q.T@b; D=q.T@role
        contrast=helmert(len(subjects))@y
        residual=z-y@q.T
        return dict(within=True,n_subjects=len(subjects),mean=np.sqrt(len(subjects))*y.mean(axis=0),
                    contrast=contrast.T,a=np.outer(one,one),u=B@B.T,v=D@D.T,
                    residual_ss=float(np.sum(residual**2)),residual_df=residual.size-y.size)
    s=across_sufficient_statistics(d,metric)
    q=span(np.column_stack([s['b'],s['role'],np.ones(len(s['mean']))]))
    B=q.T@s['b'];D=q.T@s['role'];one=q.T@np.ones(len(s['mean']))
    mean=q.T@s['mean']; mr=s['mean']-q@mean
    endpoints=[]; ss=s['interaction_ss']; df=s['interaction_df']
    for sign,r in zip(s['signs'],s['endpoint']):
        qe=span(np.column_stack([np.ones(len(sign)),sign]))
        y=qe.T@r; res=r-qe@y
        a=qe.T@np.ones(len(sign));b=qe.T@sign
        endpoints.append(dict(u=np.outer(a,a),v=np.outer(b,b),y=y,ss=y@y.T))
        ss+=float(np.sum(res**2));df+=res.size-y.size
    eu=np.zeros((len(endpoints),2,2));ev=eu.copy();ess=eu.copy();dimension=[]
    for i,endpoint in enumerate(endpoints):
        k=len(endpoint['y']);dimension.append(k)
        eu[i,:k,:k]=endpoint['u'];ev[i,:k,:k]=endpoint['v'];ess[i,:k,:k]=endpoint['ss']
    return dict(within=False,n=s['n'],a=B@B.T,u=B@B.T,v=D@D.T,
                intercept=np.outer(one,one),mean=mean,endpoints=endpoints,
                endpoint_u=eu,endpoint_v=ev,endpoint_ss=ess,endpoint_dimension=np.array(dimension),
                mean_residual_ss=float(mr@mr),mean_residual_df=len(mr)-len(mean),
                residual_ss=ss,residual_df=df)


def lognormal(y,cov):
    cf=cho_factor(cov,lower=True,check_finite=False)
    k=1 if y.ndim==1 else y.shape[1]
    return -.5*(k*len(y)*np.log(2*np.pi)+k*2*np.log(np.diag(cf[0])).sum()+np.sum(y*cho_solve(cf,y,check_finite=False)))


def isotropic(ss,df,v):
    return -.5*(df*np.log(2*np.pi*v)+ss/v)


def log_likelihood(s,sd):
    a,c,u,v,e=sd**2
    try:
        if s['within']:
            cov=a*s['a']+u*s['u']+v*s['v']+e*np.eye(len(s['mean']))
            result=lognormal(s['contrast'],cov)+lognormal(s['mean'],cov+9*s['n_subjects']*s['a'])
        else:
            n=s['n'];base=n*n*c+e
            cov=n*n*((a+u/n)*s['a']+v/n*s['v']+9*s['intercept'])+base*np.eye(len(s['mean']))
            result=lognormal(s['mean'],cov)+isotropic(s['mean_residual_ss'],s['mean_residual_df'],base)
            # Each endpoint block has rank <=2. Vectorise its exact normal density.
            cov=n*(u*s['endpoint_u']+v*s['endpoint_v'])
            c00=cov[:,0,0]+e;c01=cov[:,0,1]
            c11=np.where(s['endpoint_dimension']==2,cov[:,1,1]+e,1.)
            det=c00*c11-c01*c01
            if np.any(det<=0):return -np.inf
            ss=s['endpoint_ss']
            quad=(c11*ss[:,0,0]+c00*ss[:,1,1]-2*c01*ss[:,0,1])/det
            result+=-.5*(s['endpoint_dimension'].sum()*(n-1)*np.log(2*np.pi)+(n-1)*np.log(det).sum()+quad.sum())
        return float(result+isotropic(s['residual_ss'],s['residual_df'],e))
    except np.linalg.LinAlgError:
        return -np.inf


def run_case(task):
    import dynesty
    from threadpoolctl import threadpool_limits
    s,model,metric,within,variant,repeat,settings,path=task
    path=Path(path)
    if path.exists():
        return str(path)
    active=[i for i in VARIANTS[variant] if i!=1 or not within]
    def prior(unit):
        return ndtri(.5+.5*np.clip(unit,1e-12,1-1e-12))
    def likelihood(theta):
        sd=np.zeros(5);sd[active]=theta
        return log_likelihood(s,sd)
    # Independent reproducible seeds for every architecture/measure/scope/model/run.
    key=f'{model}:{metric}:{within}:{variant}:{repeat}:{settings["seed"]}'
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    start=time.monotonic()
    with threadpool_limits(limits=1):
        sampler=dynesty.NestedSampler(likelihood,prior,len(active),nlive=settings['nlive'],
                                     sample='rwalk',bound='multi',rstate=np.random.default_rng(seed))
        sampler.run_nested(dlogz=settings['dlogz'],print_progress=False)
    r=sampler.results
    sd=np.zeros((len(r.samples),5));sd[:,active]=r.samples
    metadata=dict(model=model,metric=metric,within_subject=within,variant=variant,repeat=repeat,
                  sampler_seed=seed,active=active,seconds=time.monotonic()-start,logz=float(r.logz[-1]),
                  logzerr=float(r.logzerr[-1]),ncall=int(np.sum(r.ncall)),**settings)
    np.savez_compressed(path,sd=sd,logwt=r.logwt,logz=r.logz,logzerr=r.logzerr,metadata=json.dumps(metadata))
    return str(path)


def summarize(paths,out):
    runs=[]
    for path in paths:
        with np.load(path) as saved:
            meta=json.loads(str(saved['metadata'])); sd=saved['sd'][:,meta['active']]
            base_logwt=saved['logwt']; base_weights=np.exp(base_logwt-logsumexp(base_logwt))
            base_ess=1/np.sum(base_weights**2)
            for scale in [.5,1.,2.]:
                ratio=-len(meta['active'])*np.log(scale)+.5*(1-scale**-2)*np.sum(sd**2,axis=1)
                logw=base_logwt+ratio; z=float(logsumexp(logw)); w=np.exp(logw-z)
                ess=float(1/np.sum(w*w))
                runs.append(dict(**meta,prior_scale=scale,logz_reweighted=z,posterior_weight_ess=ess,
                                 importance_ok=bool(ess>=400 and ess>=.2*base_ess)))
    runs=pd.DataFrame(runs);runs.to_csv(out/'evidence_runs.csv',index=False)
    combined=[]
    grouping=['model','metric','within_subject','variant','prior_scale']
    for keys,g in runs.groupby(grouping):
        logz=float(logsumexp(g.logz_reweighted)-np.log(len(g)))
        evidence_weights=np.exp(g.logz_reweighted-logsumexp(g.logz_reweighted))
        se=float(np.sqrt(np.sum((evidence_weights*g.logzerr)**2)))
        # Retain both internal NS errors and independent-run agreement.
        spread=float(g.logz_reweighted.max()-g.logz_reweighted.min())
        tolerance=float(3*np.sqrt(np.sum(g.logzerr**2)))
        combined.append(dict(zip(grouping,keys),logz=logz,logzerr=se,repeat_spread=spread,
                             repeat_agreement=bool(spread<=tolerance),importance_ok=bool(g.importance_ok.all())))
    combined=pd.DataFrame(combined);combined.to_csv(out/'marginal_likelihoods.csv',index=False)
    comparisons=[]
    for keys,g in combined.groupby(['model','metric','within_subject','prior_scale']):
        full=g[g.variant.eq('full')].iloc[0]
        for row in g[~g.variant.eq('full')].itertuples():
            logbf=full.logz-row.logz;err=np.hypot(full.logzerr,row.logzerr)
            comparisons.append(dict(zip(['model','metric','within_subject','prior_scale'],keys),
                                    comparison='full_vs_'+row.variant,log_bf=logbf,log10_bf=logbf/np.log(10),
                                    log10_bf_numerical_se=err/np.log(10),
                                    repeat_agreement=bool(full.repeat_agreement and row.repeat_agreement),
                                    importance_ok=bool(full.importance_ok and row.importance_ok)))
    table=pd.DataFrame(comparisons);table.to_csv(out/'inclusion_bayes_factors.csv',index=False)
    return table


def run_evidence(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/analyses/mechanistic_variability')
    parser.add_argument('--output',type=Path,default=ROOT/'data/analyses/mechanistic_variability/appendix/variance/inclusion_evidence')
    parser.add_argument('--nlive',type=int,default=400)
    parser.add_argument('--dlogz',type=float,default=.05)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--seed',type=int,default=20261116)
    args=parser.parse_args(argv)
    pairs,_,provenance=load_inputs(args.source)
    args.output.mkdir(parents=True,exist_ok=True)
    settings={key:getattr(args,key) for key in ['nlive','dlogz','seed']}
    identity=dict(inputs=provenance['input_sha256'],settings=settings,code_sha256=sha(__file__),
                  model_code_sha256=sha(__file__))
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=args.output/'runs'/fingerprint[:12];cache.mkdir(parents=True,exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(**identity,fingerprint=fingerprint,repeats=args.repeats,
                  prior='independent active SD HalfNormal(1), intercept Normal(0,3); identical retained priors in reduced models',
                  variants=VARIANTS,dependency_version=__import__('dynesty').__version__),indent=2))
    tasks=[]
    with threadpool_limits(limits=1):
        for model,model_id in MODELS.items():
            for within in [True,False]:
                d=pairs[pairs.model_id.eq(model_id)&pairs.within_subject.eq(within)]
                for metric in METRICS:
                    s=prepare_likelihood(d,metric,within)
                    for variant in VARIANTS:
                        if within and variant=='no_subject_or_dyad':continue
                        for repeat in range(args.repeats):
                            path=cache/f'{model}_{metric}_{within}_{variant}_{repeat}.npz'
                            tasks.append((s,model,metric,within,variant,repeat,settings,str(path)))
    paths=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run_case,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            path=future.result();paths.append(path)
            print(f'{i}/{len(tasks)} complete: {Path(path).name}',flush=True)
    table=summarize(paths,args.output)
    print(table[table.prior_scale.eq(1)].to_string(index=False),flush=True)
    for name,expected in provenance['input_sha256'].items():
        assert sha(args.source/name)==expected,'Inputs changed during computation'

# Exploratory factorial contrasts and observed-path attribution

ARCH=['GRU','lightGRU','monoGRU']
FACTORS=['architecture','BC','DB_removed']


def basis(k):
    if k==2:return np.array([[-1],[1]])/np.sqrt(2)
    return np.array([[-1,-1],[1,-1],[0,2]])/np.array([np.sqrt(2),np.sqrt(6)])


def matrix_for(levels,effect):
    result=np.ones((1,1))
    for name,k in levels:
        term=basis(k) if name in effect else np.ones((k,1))/np.sqrt(k)
        result=np.kron(result,term)
    return result


def anova(y,levels):
    rows=[];n=len(y)
    for size in range(1,len(levels)+1):
        for effect in itertools.combinations([name for name,_ in levels],size):
            c=y@matrix_for(levels,effect);k=c.shape[1]
            mean=c.mean(axis=0);ss_effect=n*float(mean@mean);ss_error=float(((c-mean)**2).sum())
            F=ss_effect/(ss_error/(n-1)) if ss_error>0 else np.inf
            covariance=np.atleast_2d(np.cov(c,rowvar=False,ddof=1))
            epsilon=1. if k==1 else float(np.trace(covariance)**2/(k*np.trace(covariance@covariance)))
            epsilon=float(np.clip(epsilon,1/k,1))
            if k==1:np.testing.assert_allclose(F,ttest_1samp(c[:,0],0).statistic**2,rtol=1e-9)
            rows.append(dict(effect=':'.join(effect),F=F,df1=k,df2=(n-1)*k,epsilon_GG=epsilon,df1_GG=k*epsilon,df2_GG=(n-1)*k*epsilon,
                             p_uncorrected=f_dist.sf(F,k,(n-1)*k),p_GG=f_dist.sf(F,k*epsilon,(n-1)*k*epsilon),
                             partial_eta_squared=ss_effect/(ss_effect+ss_error)))
    return rows


def scalar_weights(architecture,bc,db):return np.kron(np.kron(architecture,bc),db)


def run_factorial(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/analyses/mechanistic_variability/appendix/factorial')
    parser.add_argument('--source',type=Path,default=ROOT/'data/analyses/mechanistic_variability')
    args=parser.parse_args(argv)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'results/appendix/mechanistic_variability/factorial';figures.mkdir(parents=True,exist_ok=True)
    source=args.source;selected=pd.read_csv(source/'selected_models.htsv',sep='\t');units=pd.read_csv(source/'subject_pair_similarity.htsv',sep='\t')
    assert selected.selection_method.eq('block_reset').all()
    design=selected[['model_id','model_type','nonlinearity','decoder_bias','constraint']].drop_duplicates().copy()
    design['BC']=design.nonlinearity.eq('relu').astype(int);design['DB_removed']=(~design.decoder_bias).astype(int)
    design['reset_removed']=design.model_type.ne('GRU').astype(int);design['scalar_gate']=design.model_type.eq('monoGRU').astype(int)
    assert len(design)==12 and not design.duplicated(['model_type','BC','DB_removed']).any()
    # Missing (reset_removed=0,scalar_gate=1) makes this interaction aliased.
    gate=design[['reset_removed','scalar_gate']].to_numpy();X=np.column_stack([np.ones(len(gate)),gate,gate[:,0]*gate[:,1]])
    assert np.linalg.matrix_rank(X)==3 and np.array_equal(X[:,2],X[:,3])
    design.to_csv(out/'design.csv',index=False)
    values=units.merge(design,on='model_id',validate='many_to_one')
    anovas=[];contrasts=[]
    means=np.ones(2)/2;diff=np.array([-1,1]);a_mean=np.ones(3)/3
    gates={'remove_reset_light_minus_GRU':np.array([-1,1,0]),'scalar_mono_minus_light':np.array([0,-1,1])}
    for metric in measures.SIMILARITY_METRICS:
        is_reset=metric=='gate_reset_similarity'
        architectures=['GRU'] if is_reset else ARCH
        order=[(a,b,d) for a in architectures for b in [0,1] for d in [0,1]]
        names=[design[(design.model_type.eq(a))&design.BC.eq(b)&design.DB_removed.eq(d)].model_id.iloc[0] for a,b,d in order]
        levels=[('BC',2),('DB_removed',2)] if is_reset else [('architecture',3),('BC',2),('DB_removed',2)]
        weights={'BC':np.kron(diff,means),'DB_removed':np.kron(means,diff),'BC:DB_removed':np.kron(diff,diff)}
        if not is_reset:
            weights={k:np.kron(a_mean,v) for k,v in weights.items()}
            for name,a in gates.items():
                for label,b,d in [('',means,means),(':BC',diff,means),(':DB_removed',means,diff),(':BC:DB_removed',diff,diff)]:
                    weights[name+label]=scalar_weights(a,b,d)
        for scope in [True,False]:
            group=values[values.metric.eq(metric)&values.within_subject.eq(scope)]
            pivot=group.pivot(index=['subject_A','subject_B'],columns='model_id',values='mean_similarity').reindex(columns=names)
            assert len(pivot)==(11 if scope else 55) and pivot.notna().all().all()
            y=pivot.to_numpy();endpoints=pivot.index.to_frame(index=False);subjects=sorted(set(endpoints.subject_A)|set(endpoints.subject_B))
            if scope:
                for result in anova(y,levels):anovas.append(dict(metric=metric,n_subjects=11,architecture_scope='GRU only' if is_reset else 'all',**result))
            for effect,w in weights.items():
                d=y@w;mean=d.mean();median=np.median(d)
                loo=np.array([d[~(endpoints.subject_A.eq(s)|endpoints.subject_B.eq(s))].mean() for s in subjects])
                se=np.sqrt(10/11*((loo-loo.mean())**2).sum());stat=mean/se if se>0 else np.inf
                p=float(2*t_dist.sf(abs(stat),10)) if se>0 else (1. if mean==0 else 0.)
                sign=binomtest(int((d>0).sum()),int((d!=0).sum()),.5).pvalue if scope and (d!=0).any() else np.nan
                contrasts.append(dict(metric=metric,within_subject=scope,effect=effect,mean_contrast=mean,median_contrast=median,
                    deletion_min=loo.min(),deletion_max=loo.max(),n_subjects=11,n_units=len(d),standard_error=se,t=stat,p_raw=p,
                    p_sign_sensitivity=sign,inference='Paired mean contrast t(10)' if scope else 'Approximate subject-deletion jackknife t(10)'))
    a=pd.DataFrame(anovas);a['q_BH_omnibus31']=false_discovery_control(a.p_GG.to_numpy(),method='bh');assert len(a)==31
    c=pd.DataFrame(contrasts);assert len(c)==94;c['q_BH_contrasts94']=false_discovery_control(c.p_raw.to_numpy(),method='bh')
    mask=c.within_subject;c['q_BH_sign_sensitivity47']=np.nan
    c.loc[mask,'q_BH_sign_sensitivity47']=false_discovery_control(c.loc[mask,'p_sign_sensitivity'].fillna(1).to_numpy(),method='bh')
    a.to_csv(out/'within_repeated_measures_ANOVA.csv',index=False);c.to_csv(out/'planned_contrasts.csv',index=False)
    a.to_csv(figures/'table_within_ANOVA.csv',index=False);c.to_csv(figures/'table_planned_contrasts.csv',index=False)
    from NM_TinyRNN.code.figures.mechanistic_variability_factorial import plot_factorial, plot_interactions
    plot_interactions(values,figures)
    cells, attribution = reconstruct_attribution(values)
    cells.to_csv(out/'cell_means.csv',index=False)
    attribution.to_csv(out/'precedence_constrained_attribution.csv',index=False)
    plot_factorial(cells,c,attribution,figures)
    attribution.to_csv(figures/'table_factorial_attribution.csv',index=False)
    methods=dict(attribution='Mean marginal changes over the 12 intervention orders in which reset removal precedes scalar gating. Uses only observed configurations; conditional exploratory attribution, not unconstrained Shapley or causal effects. Contributions sum to target minus baseline. Error bars use subject-deletion SE and t(10).',source_run='nested_DA_S32_B8',selection='block_reset',design='Architecture(3) × BC(2) × decoder-bias-removal(2)',
        gate_estimability='Reset removal: lightGRU−GRU; scalar gating conditional on no reset: monoGRU−lightGRU. reset_removed×scalar_gate aliases scalar_gate and is not estimable independently.',
        BC_confound='BC is ReLU versus tanh; training energy-regularization grid also changes. Interpret as bundled configuration contrast, not isolated biological-constraint causality.',
        within='11 repeated subjects, full factorial ANOVA via orthonormal contrasts; Greenhouse-Geisser corrected dfs/p when effect df>1; partial eta squared descriptive',
        across='No ordinary ANOVA treating 55 dyads as independent. Scalar factorial contrasts use deletion of all incident dyads, approximate t(10); exploratory with 11 nodes.',
        reset_metric='Reset-gate similarity exists only in GRU: BC × DB_removed only',
        correction='BH over31 within omnibus tests; separate BH over94 planned mean contrasts spanning within/across; optional sign sensitivity corrected over47 within contrasts',
        assumptions='RM ANOVA requires distributional assumptions; GG adjusts sphericity, not arbitrary nonnormality. Across jackknife approximate. Within sign sensitivity tests direction, not the mean ANOVA estimand.',
        inputs_sha256={p.name:measures.sha256(p) for p in [source/'selected_models.htsv',source/'subject_pair_similarity.htsv']},script_sha256=measures.sha256(__file__))
    (out/'manifest.json').write_text(json.dumps(methods,indent=2)+'\n')
    text='# Exploratory factorial mechanistic variability\n\n'+''.join(f'## {k}\n\n{v}\n\n' for k,v in methods.items() if not k.endswith('sha256'))
    (figures/'README.md').write_text(text)
    print('Omnibus tests:',len(a),'BH significant:',int((a.q_BH_omnibus31<.05).sum()))
    print(a[a.q_BH_omnibus31<.05][['metric','effect','F','p_GG','q_BH_omnibus31']].round(6).to_string(index=False))
    print('Planned contrasts:',len(c),'BH significant:',int((c.q_BH_contrasts94<.05).sum()))
    print(out)


def ordered_contributions(cell_values):
    """Four marginal contributions, with reset removal preceding scalar gating.

    cell_values has shape (..., architecture=3, BC=2, DB_removed=2).
    Intervention indices are BC, reset removal, scalar gate, DB removal.
    """
    values=np.asarray(cell_values)
    if values.shape[-3:]!=(3,2,2):
        raise ValueError('Expected the complete 3 by 2 by 2 observed design')
    paths=[p for p in itertools.permutations(range(4)) if p.index(1)<p.index(2)]
    contributions=np.zeros(values.shape[:-3]+(4,))
    for path in paths:
        state=[0,0,0,0]
        previous=values[...,0,0,0]
        for intervention in path:
            state[intervention]=1
            current=values[...,state[1]+state[2],state[0],state[3]]
            contributions[...,intervention]+=current-previous
            previous=current
    contributions/=len(paths)
    np.testing.assert_allclose(contributions.sum(axis=-1),values[...,2,1,1]-values[...,0,0,0],atol=1e-12)
    return contributions


def reconstruct_attribution(values):
    cells=values.groupby(['metric','within_subject','model_type','BC','DB_removed'],as_index=False).mean_similarity.mean()
    rows=[]
    for (metric,within),group in values.groupby(['metric','within_subject']):
        if metric=='gate_reset_similarity':
            continue
        order=[(a,b,d) for a in ARCH for b in [0,1] for d in [0,1]]
        pivot=group.pivot(index=['subject_A','subject_B'],columns=['model_type','BC','DB_removed'],values='mean_similarity').reindex(columns=order)
        if pivot.isna().any().any():
            raise ValueError('Attribution requires all 12 matched configurations')
        contributions=ordered_contributions(pivot.to_numpy().reshape(-1,3,2,2))
        endpoints=pivot.index.to_frame(index=False)
        subjects=sorted(set(endpoints.subject_A)|set(endpoints.subject_B))
        all_values=np.column_stack([contributions,contributions.sum(axis=1)])
        for j,label in enumerate(['+BC','−reset gate','scalar gate','−DB','TOTAL']):
            x=all_values[:,j]
            deleted=np.array([x[~(endpoints.subject_A.eq(s)|endpoints.subject_B.eq(s))].mean() for s in subjects])
            se=np.sqrt((len(subjects)-1)/len(subjects)*((deleted-deleted.mean())**2).sum())
            half=t_dist.ppf(.975,len(subjects)-1)*se
            rows.append(dict(metric=metric,within_subject=within,intervention=label,estimate=x.mean(),standard_error=se,
                             ci95_lo=x.mean()-half,ci95_hi=x.mean()+half,deletion_min=deleted.min(),deletion_max=deleted.max(),
                             n_subjects=len(subjects),n_units=len(x),n_valid_orders=12))
    return cells,pd.DataFrame(rows)

# Direct, subject-paired configuration comparisons

DIRECT_MODELS = [
    ('GRU', '2_unit_GRU_tanh_unipolar_forced'),
    ('GRU+BC', '2_unit_GRU_relu_unipolar_forced'),
    ('lightGRU+BC', '2_unit_lightGRU_relu_unipolar_forced'),
    ('monoGRU+BC', '2_unit_monoGRU_relu_unipolar_forced'),
    ('monoGRU+BC−DB', '2_unit_monoGRU_relu_unipolar_forced_ndb'),
]
DIRECT_METRICS = [
    ('logit_value_similarity', 'Logit'), ('weight_similarity', 'Weights'),
    ('hidden_similarity', 'Hidden state'), ('gate_update_similarity', 'Update gate'),
    ('gate_reset_similarity', 'Reset gate'),
]


def direct_subject_comparisons(pairs, selected):
    """Exact signs of paired differences in within-subject fold-pair means.

    One observation per subject/configuration, averaging its 45 outer-fit pairs.
    Reset comparisons are available only when both configurations are GRUs.
    Exact zero differences are omitted from the binomial denominator. Missing
    observations for an available measure raise rather than drop a subject.
    """
    ids=[model_id for _,model_id in DIRECT_MODELS]
    metadata=selected[selected.model_id.isin(ids)].copy()
    if set(metadata.model_id)!=set(ids) or not metadata.hidden_size.eq(2).all():
        raise ValueError('Direct comparisons require the five specified two-unit configurations')
    if not metadata.selection_method.eq('block_reset').all():
        raise ValueError('Direct comparisons require block-reset-selected outer winners')
    subjects=sorted(metadata.subject_id.unique())
    meta=metadata.drop_duplicates('model_id').set_index('model_id')
    if not metadata.groupby('model_id').subject_id.nunique().eq(len(subjects)).all():
        raise ValueError('Configurations must contain the same subjects')
    within=pairs[pairs.within_subject & pairs.model_id.isin(ids)].copy()
    if not within.subject_A.eq(within.subject_B).all():
        raise ValueError('Within-subject rows must have identical subject endpoints')
    means=within.groupby(['model_id','subject_A'])[[m for m,_ in DIRECT_METRICS]].mean()
    observations=[];tests=[]
    for comparison,((label_a,id_a),(label_b,id_b)) in enumerate(zip(DIRECT_MODELS,DIRECT_MODELS[1:])):
        for metric,label in DIRECT_METRICS:
            available=metric!='gate_reset_similarity' or (meta.loc[[id_a,id_b],'model_type'].eq('GRU').all())
            record=dict(within_subject=True,test_method='Wilcoxon signed-rank',comparison=comparison,model_a=id_a,model_b=id_b,label_a=label_a,label_b=label_b,
                        metric=metric,component=label,available=bool(available),n_subjects=0,n_positive=0,n_negative=0,n_ties=0,
                        median_difference=np.nan,mean_difference=np.nan,ci95_lo=np.nan,ci95_hi=np.nan,cohens_dz=np.nan,
                        p_raw=np.nan,p_sign_sensitivity=np.nan,q_bh=np.nan)
            if available:
                a=means.xs(id_a).reindex(subjects)[metric].to_numpy()
                b=means.xs(id_b).reindex(subjects)[metric].to_numpy()
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f'Missing subject values in {label_a} versus {label_b}: {metric}')
                delta=b-a;positive=int((delta>0).sum());negative=int((delta<0).sum());n=positive+negative
                record.update(n_subjects=len(subjects),n_positive=positive,n_negative=negative,n_ties=len(subjects)-n,
                              median_difference=float(np.median(delta)),mean_difference=float(delta.mean()),
                              p_raw=float(wilcoxon(delta).pvalue) if n else 1.,
                              p_sign_sensitivity=float(binomtest(positive,n,.5).pvalue) if n else 1.)
                sd=float(delta.std(ddof=1));half=float(t_dist.ppf(.975,len(delta)-1)*sd/np.sqrt(len(delta)))
                record.update(ci95_lo=float(delta.mean())-half,ci95_hi=float(delta.mean())+half,cohens_dz=float(delta.mean())/sd if sd>0 else np.nan)
                for subject,x,y,difference in zip(subjects,a,b,delta):
                    observations.append(dict(within_subject=True,comparison=comparison,metric=metric,subject_id=subject,label_a=label_a,label_b=label_b,
                                             model_a=id_a,model_b=id_b,similarity_a=x,similarity_b=y,difference=difference))
            tests.append(record)
    tests=pd.DataFrame(tests);valid=tests.available
    tests.loc[valid,'q_bh']=false_discovery_control(tests.loc[valid,'p_raw'].to_numpy(),method='bh')
    tests['significance']=''
    for cutoff,mark in [(.05,'*'),(.01,'**'),(.001,'***')]:
        tests.loc[tests.q_bh.lt(cutoff),'significance']=mark
    return pd.DataFrame(observations),tests


def direct_across_comparisons(pairs,selected,specification):
    """Matched dyad-mean differences with subject-deletion jackknife t inference.

    Plot points are incident-dyad means (11 overlapping subject summaries).
    Tests use all 55 matched dyad differences and delete all dyads incident to
    each subject; the 11 plotted summaries are not treated as independent.
    """
    ids=[model_id for _,model_id in DIRECT_MODELS]
    subjects=sorted(selected[selected.model_id.isin(ids)].subject_id.unique())
    n=len(subjects);expected=n*(n-1)//2
    across=pairs[~pairs.within_subject & pairs.model_id.isin(ids)]
    means=across.groupby(['model_id','subject_A','subject_B'])[[m for m,_ in DIRECT_METRICS]].mean()
    observations=[];dyad_values=[];deletions=[];tests=[]
    for template in specification.to_dict('records'):
        record={**template,'within_subject':False,'test_method':f'Subject-deletion jackknife t({n-1})',
                'n_subjects':0,'n_subject_pairs':0,'n_positive':np.nan,'n_negative':np.nan,'n_ties':np.nan,
                'median_difference':np.nan,'mean_difference':np.nan,'jackknife_se':np.nan,
                'ci95_lo':np.nan,'ci95_hi':np.nan,'p_raw':np.nan,'q_bh':np.nan,'significance':''}
        if template['available']:
            metric=template['metric'];a=template['model_a'];b=template['model_b']
            d=means.xs(a)[[metric]].rename(columns={metric:'reference'}).join(
                means.xs(b)[[metric]].rename(columns={metric:'model'}),how='outer').reset_index()
            endpoints=set(d.subject_A)|set(d.subject_B)
            unordered=[tuple(sorted((x,y))) for x,y in zip(d.subject_A,d.subject_B)]
            if len(d)!=expected or len(set(unordered))!=expected or endpoints!=set(subjects) or d.subject_A.eq(d.subject_B).any():
                raise ValueError('Across comparisons require every subject dyad exactly once')
            if not np.isfinite(d[['reference','model']].to_numpy()).all():
                raise ValueError(f'Missing across-subject dyad values for {metric}')
            d['difference']=d.model-d.reference
            result=measures.contrast(d)
            record.update(n_subjects=n,n_subject_pairs=len(d),median_difference=float(d.difference.median()),
                          mean_difference=result['delta_similarity'],jackknife_se=result['jackknife_se'],
                          ci95_lo=result['delta_ci95_low'],ci95_hi=result['delta_ci95_high'],p_raw=result['p_raw'],cohens_dz=result['cohens_dz'])
            common={key:template[key] for key in ['comparison','metric','label_a','label_b','model_a','model_b']}
            for subject in subjects:
                incident=d.subject_A.eq(subject)|d.subject_B.eq(subject)
                values=d[incident]
                observations.append(dict(**common,within_subject=False,subject_id=subject,
                                         similarity_a=values.reference.mean(),similarity_b=values.model.mean(),difference=values.difference.mean()))
                deletions.append(dict(**common,deleted_subject=subject,mean_difference=d.loc[~incident,'difference'].mean(),n_remaining_dyads=int((~incident).sum())))
            for row in d.itertuples():
                dyad_values.append(dict(**common,subject_A=row.subject_A,subject_B=row.subject_B,
                                        similarity_a=row.reference,similarity_b=row.model,difference=row.difference))
        tests.append(record)
    tests=pd.DataFrame(tests);valid=tests.available
    tests.loc[valid,'q_bh']=false_discovery_control(tests.loc[valid,'p_raw'].to_numpy(),method='bh')
    for cutoff,mark in [(.05,'*'),(.01,'**'),(.001,'***')]:
        tests.loc[tests.q_bh.lt(cutoff),'significance']=mark
    return pd.DataFrame(observations),tests,pd.DataFrame(dyad_values),pd.DataFrame(deletions)


def report_direct_comparisons(pairs,selected,tables,output,manifest):
    within_values,within_tests=direct_subject_comparisons(pairs,selected)
    across_values,across_tests,dyads,deletions=direct_across_comparisons(pairs,selected,within_tests)
    observations=pd.concat([within_values,across_values],ignore_index=True)
    tests=pd.concat([within_tests,across_tests],ignore_index=True)
    target=tables/'appendix/direct_comparisons';target.mkdir(parents=True,exist_ok=True)
    observations.to_csv(target/'paired_subject_values.csv',index=False)
    (target/'sign_tests.csv').unlink(missing_ok=True)
    tests.to_csv(target/'paired_tests.csv',index=False)
    dyads.to_csv(target/'across_dyad_values.csv',index=False)
    deletions.to_csv(target/'across_subject_deletions.csv',index=False)
    tests.to_csv(output/'table_direct_model_comparisons.csv',index=False)
    methods=dict(folder=manifest['folder'],selection='Block-reset-selected outer winners',scope='Within and across subjects',
                 observation='Within: mean of 45 fold pairs per subject. Across plots: each subject mean over ten incident dyad means; these plotted summaries overlap. Each dyad mean averages 100 fold pairs on its saved reference trials',
                 effect_size='d_z = mean difference/(jackknife SE*sqrt(n)): mean/SD of subject differences within; jackknife-consistent subject-level effect across',
                 test='Within: two-sided Wilcoxon signed-rank test on 11 paired subject differences (exact sign test kept as p_sign_sensitivity). Across: mean of 55 matched dyad differences, subject-deletion jackknife SE, approximate t(10); delete each subject and all incident dyads',
                 correction='Separate BH families: 17 available tests per row; within-subject q values are unchanged',
                 absent_gates='No reset-gate test when either configuration has no reset gate; no imputation or subject deletion',
                 contrast='Second configuration minus first; no across-subject dyads are treated as independent subjects',
                 interpretation='BC is a bundled activation/regularization configuration contrast; outer folds are not independent replicates',
                 source_sha256={name:measures.sha256(tables/name) for name in ['pairwise_similarities.htsv','selected_models.htsv']})
    study.write_json(target/'methods.json',methods)
    from NM_TinyRNN.code.figures.mechanistic_variability import direct_comparison_figure
    direct_comparison_figure(observations,tests,output)
    print(tests.groupby('within_subject').agg(n_tests=('p_raw','count'),n_BH_significant=('q_bh',lambda q: int(q.lt(.05).sum()))).to_string())


# Publication orchestration and cache provenance

def adopt_metadata_only(source,output):
    """Reuse posterior filenames only when scientific inputs/protocol are identical."""
    path=output/'manifest.json'
    if not path.exists():
        raise FileNotFoundError('No variance posterior manifest; use --variance-stage all to fit explicitly')
    previous=json.loads(path.read_text())
    _,_,current=load_inputs(source)
    if previous['input_sha256']==current['input_sha256']:
        return
    for name in ['pairwise_similarities.htsv','selected_models.htsv']:
        if previous['input_sha256'][name]!=current['input_sha256'][name]:
            raise ValueError('Variance cache scientific inputs changed; refit explicitly')
    old=previous['source_manifest'];new=current['source_manifest']
    # The posterior likelihood consumes scores/selection, not source-code hashes.
    # Accept code-only fingerprint changes only with complete, equal protocol
    # metadata and the byte-identical scientific inputs verified above.
    provenance_keys={'fingerprint','source_code_sha256'}
    old_protocol={k:v for k,v in old['computation'].items() if k not in provenance_keys}
    new_protocol={k:v for k,v in new['computation'].items() if k not in provenance_keys}
    fingerprint_changed=old['computation']['fingerprint']!=new['computation']['fingerprint']
    complete_protocol=all(k in old_protocol for k in ['alignment_protocol','selection'])
    if old['folder']!=new['folder'] or old_protocol!=new_protocol or (fingerprint_changed and not complete_protocol):
        raise ValueError('Variance cache computation protocol changed; refit explicitly')
    audit=output/'metadata_migrations.json'
    history=json.loads(audit.read_text()) if audit.exists() else []
    history.append(dict(previous_source=previous['source'],previous_input_sha256=previous['input_sha256'],
                        current_input_sha256=current['input_sha256'],reason='Metadata/code provenance migration only; byte-identical pairwise scores and selected models, identical scientific protocol',
                        previous_computation_fingerprint=old['computation']['fingerprint'],current_computation_fingerprint=new['computation']['fingerprint'],
                        preserved_posterior_fingerprint=previous['fingerprint']))
    study.write_json(audit,history)
    previous.update(current)
    study.write_json(path,previous)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',default=study.DEFAULT_RUN)
    parser.add_argument('--sections',nargs='+',choices=['variance','similarities','cohens_d','factorial','evidence','direct_comparisons'],default=['variance','similarities','cohens_d','factorial','direct_comparisons'])
    parser.add_argument('--variance-stage',choices=['plots','all','descriptive','bayesian','retry'],default='plots',help='Default reuses saved posterior draws; all/bayesian explicitly permit fitting')
    parser.add_argument('--evidence-jobs',type=int,default=6,help='Workers for explicitly requested inclusion-evidence sampling')
    args=parser.parse_args(argv)
    tables,_,_=study.paths_for(args.folder)
    manifest=json.loads((tables/'manifest.json').read_text())
    if manifest['folder']!=args.folder:
        raise ValueError('Cached analysis belongs to another training run')
    selected,pairs=study.load_computation(tables,manifest)
    if not selected.selection_method.eq('block_reset').all():
        raise ValueError('Appendix requires block-reset-selected outer winners')
    output=REPO/'results/appendix/mechanistic_variability';output.mkdir(parents=True,exist_ok=True)
    from NM_TinyRNN.code.figures import mechanistic_variability as plotting
    import pandas as pd
    if any(section in args.sections for section in ['similarities','cohens_d','factorial']):
        study.statistics(selected,pairs,tables,manifest)
        study.write_json(tables/'manifest.json',manifest)
    if 'similarities' in args.sections:
        plotting.appendix_all_models(pairs,selected,output)
    if 'cohens_d' in args.sections:
        contrasts=pd.read_csv(tables/'similarity_vs_standard_gru.htsv',sep='\t')
        plotting.appendix_cohens_d(contrasts,selected,output)
    if 'factorial' in args.sections:
        run_factorial(['--source',str(tables),'--output',str(tables/'appendix/factorial')])
    if 'variance' in args.sections:
        variance_output=tables/'appendix/variance'
        if args.variance_stage in ['plots','retry']:
            adopt_metadata_only(tables,variance_output)
        run_variance(['--source',str(tables),'--output',str(variance_output),'--stage',args.variance_stage])
    if 'direct_comparisons' in args.sections:
        report_direct_comparisons(pairs,selected,tables,output,manifest)
    if 'evidence' in args.sections:
        run_evidence(['--source',str(tables),'--output',str(tables/'appendix/variance/inclusion_evidence'),'--jobs',str(args.evidence_jobs)])
    report=dict(folder=args.folder,sections=args.sections,variance_stage=args.variance_stage,
                source_sha256={name:measures.sha256(tables/name) for name in ['pairwise_similarities.htsv','selected_models.htsv']},
                outputs_sha256={str(p.relative_to(output)):measures.sha256(p) for p in output.rglob('*') if p.is_file()},
                methods='Canonical block-reset winners and A-reference trial sequences; unchanged cached similarities. Full-model Cohen panels use BH94; factorial omnibus BH31 and contrasts BH94. Hierarchical variance is exploratory on atanh(similarity); it describes similarity-score variability, not direct parameter reproducibility.')
    study.write_json(tables/'appendix/manifest.json',report)
    print(f'Appendix outputs: {output}')

if __name__=='__main__':
    main()
