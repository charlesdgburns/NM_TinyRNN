'''Here are a few general functions for plotting some stats.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pingouin as pg
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multitest import multipletests
from itertools import combinations




## Two-variable t-tests plotted easily ##
def plot_paired_comparison(df,
                           y,
                           x,
                           within_variable,
                           paired_across,
                           mean_across=None,
                           filter_by=None,
                           dodge_offset=0.15,
                           colors=None,
                           correction_method='fdr_bh',
                           ax=None):
    """
    Plots a paired comparison of y across levels of x, grouped by within_variable.
    Performs a paired t-test across levels of x, paired by paired_across.
    Corrects for multiple comparisons across within_variable groups.

    Parameters
    ----------
    df                : pd.DataFrame
    y                 : str, outcome variable (e.g. 'eval_CE')
    x                 : str, variable with (exactly 2) levels to compare (e.g. 'hidden_size')
    within_variable   : str, grouping variable shown on x-axis (e.g. 'model_type')
    paired_across     : str, variable used to pair observations (e.g. 'subject_ID')
    mean_across       : str or list, optional column(s) to average over before plotting
    filter_by         : dict, optional filters e.g. {'nonlinearity': 'relu'}
    dodge_offset      : float, horizontal spacing between paired points
    colors            : list of 2 colours for the two x levels, optional
    correction_method : str, multiple comparison correction method (default: 'fdr_bh')
                        any method accepted by statsmodels.stats.multitest.multipletests
    ax                : matplotlib axis, optional. If None, creates a new figure and axis.
    """

    df = df.copy()

    # --- Filter ---
    if filter_by:
        for col, val in filter_by.items():
            df = df[df[col] == val]

    # --- Average over nuisance variables ---
    if mean_across:
        mean_across = [mean_across] if isinstance(mean_across, str) else mean_across
        group_cols = [paired_across, within_variable, x]
        df = df.groupby(group_cols)[y].mean().reset_index()

    # --- Validate x has exactly 2 levels ---
    if df[x].nunique() != 2:
        x_vals = sorted(df[x].unique())
        raise ValueError(f"'{x}' must have exactly 2 levels for a paired t-test, found: {x_vals}")

    within_levels = sorted(df[within_variable].unique())
    n_groups = len(within_levels)

    if colors is None:
        colors = ['salmon', 'navy']

    # --- First pass: collect paired data and raw p-values ---
    all_paired = []
    raw_pvals = []
    x_levels = None
    for group in within_levels:
        group_df = df[df[within_variable] == group]

        # Check for duplicates
        dupe_check = group_df.groupby([paired_across, x]).size()
        if (dupe_check > 1).any():
            print(f"WARNING: duplicate rows for {within_variable}={group}:")
            print(dupe_check[dupe_check > 1])

        paired = group_df.pivot(index=paired_across, columns=x, values=y).dropna()

        # Extract x_levels from actual pivot columns to ensure type consistency
        if x_levels is None:
            x_levels = sorted(paired.columns.tolist())
            if len(x_levels) != 2:
                raise ValueError(f"'{x}' must have exactly 2 levels in pivot, found: {x_levels}")

        all_paired.append(paired)

        if paired.empty or len(paired) < 2:
            print(f"Warning: insufficient pairs for {within_variable}={group}, skipping.")
            raw_pvals.append(np.nan)
        else:
            _, p_val = ttest_rel(paired[x_levels[0]].values, paired[x_levels[1]].values)
            raw_pvals.append(p_val)

    # --- Correct for multiple comparisons (excluding NaNs) ---
    valid_mask = ~np.isnan(raw_pvals)
    corrected_pvals = np.full(len(raw_pvals), np.nan)
    if valid_mask.sum() > 1:
        _, corrected_pvals[valid_mask], _, _ = multipletests(
            np.array(raw_pvals)[valid_mask], method=correction_method)
    else:
        corrected_pvals[valid_mask] = np.array(raw_pvals)[valid_mask]  # no correction if only 1 valid test

    # --- Plot ---
    y_range = df[y].max() - df[y].min()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3 * n_groups, 6))
        created_fig = True
    else:
        created_fig = False

    for i, (group, paired, p_val_corr) in enumerate(zip(within_levels, all_paired, corrected_pvals)):
        if paired.empty:
            continue

        vals0 = paired[x_levels[0]].values
        vals1 = paired[x_levels[1]].values
        x0, x1 = i - dodge_offset, i + dodge_offset

        # Connecting lines
        for v0, v1 in zip(vals0, vals1):
            ax.plot([x0, x1], [v0, v1], color='grey', alpha=0.5, linewidth=0.8, zorder=1)

        # Dots
        ax.scatter([x0] * len(vals0), vals0, color=colors[0], zorder=2, s=50, alpha=0.9,
                   label=f'{x}={x_levels[0]}' if i == 0 else '')
        ax.scatter([x1] * len(vals1), vals1, color=colors[1], zorder=2, s=50, alpha=0.9,
                   label=f'{x}={x_levels[1]}' if i == 0 else '')

        # Significance annotation with corrected p-value
        sig = 'n.s.' if p_val_corr > 0.05 else ('*' if p_val_corr > 0.01 else ('**' if p_val_corr > 0.001 else '***'))
        y_max = max(vals0.max(), vals1.max())
        y_ann = y_max + 0.01 * y_range
        ax.plot([x0, x1], [y_ann, y_ann], color='black', linewidth=1)
        ax.text((x0 + x1) / 2, y_ann + 0.005 * y_range,
                f'{sig}\np={p_val_corr:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(within_levels)
    ax.set_xlabel(within_variable)
    ax.set_ylabel(y)
    ax.set_title(f'Paired comparison: {y} by {x}\n'
                 f'grouped by {within_variable}, paired across {paired_across}\n'
                 f'(multiple comparison correction: {correction_method})')
    ax.legend(title=x)
    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


## anova ##

def plot_repeated_measures_anova(df,
                                  y,
                                  x,
                                  paired_across,
                                  mean_across=None,
                                  filter_by=None,
                                  colors=None,
                                  correction_method='fdr_bh',
                                  ax=None):
    """
    Plots a repeated measures ANOVA comparison of y across levels of x,
    with post-hoc paired t-tests, FDR correction, and bracket annotations.
    Conditions ordered from worst (highest mean y) to best (lowest mean y).
    Effect size reported as mean difference in y units.

    Parameters
    ----------
    df                : pd.DataFrame
    y                 : str, outcome variable (e.g. 'eval_CE')
    x                 : str, grouping variable on x-axis (e.g. 'model_type')
    paired_across     : str, variable used to pair observations (e.g. 'subject_ID')
    mean_across       : str or list, optional column(s) to average over before plotting
    filter_by         : dict, optional filters e.g. {'nonlinearity': 'relu'}
    colors            : list of colours, one per level of x, optional
    correction_method : str, multiple comparison correction method (default: 'fdr_bh')
    ax                : matplotlib axis, optional. If None, creates a new figure and axis.
    """
    
    df = df.copy()

    # --- Filter ---
    if filter_by:
        for col, val in filter_by.items():
            df = df[df[col] == val]

    # --- Average over nuisance variables ---
    if mean_across:
        mean_across = [mean_across] if isinstance(mean_across, str) else mean_across
        group_cols = [paired_across, x]
        df = df.groupby(group_cols)[y].mean().reset_index()

    # --- Check for duplicates ---
    dupe_check = df.groupby([paired_across, x]).size()
    if (dupe_check > 1).any():
        print("WARNING: duplicate rows found after averaging — check mean_across argument:")
        print(dupe_check[dupe_check > 1])

    # --- Pivot to wide format ---
    wide = df.pivot(index=paired_across, columns=x, values=y).dropna()
    n_subjects = len(wide)
    print(f"N subjects with complete data: {n_subjects}")

    # --- Sort x levels from worst (highest mean y) to best (lowest mean y) ---
    x_levels = wide.mean().sort_values(ascending=False).index.tolist()
    n_levels  = len(x_levels)

    # --- Repeated measures ANOVA via pingouin ---
    long = wide.reset_index().melt(id_vars=paired_across, var_name=x, value_name=y)
    anova_result = pg.rm_anova(data=long, dv=y, within=x, subject=paired_across)
    print("\nRepeated measures ANOVA:")
    print(anova_result.to_string(index=False))

    # --- Robustly extract ANOVA stats ---
    f_col  = [c for c in anova_result.columns if c.lower() == 'f'][0]
    es_col = [c for c in anova_result.columns if 'ng2' in c.lower() or 'eta' in c.lower()][0]
    f_stat = anova_result[f_col].values[0]
    ng2    = anova_result[es_col].values[0]

    sphericity_violated = not anova_result['sphericity'].values[0]
    if sphericity_violated:
        p_col_plot = [c for c in anova_result.columns if 'gg' in c.lower()][0]
        p_anova = anova_result[p_col_plot].values[0]
        sphericity_str = (f'Sphericity violated (W={anova_result["W_spher"].values[0]:.3f}, '
                          f'p={anova_result["p_spher"].values[0]:.4f}), GG corrected')
    else:
        p_col = [c for c in anova_result.columns if 'p_unc' in c.lower()][0]
        p_anova = anova_result[p_col].values[0]
        sphericity_str = 'Sphericity assumption met'

    # --- Post-hoc paired t-tests with FDR correction ---
    pairs = list(combinations(x_levels, 2))
    raw_pvals = []
    t_stats = []
    for g1, g2 in pairs:
        t, p = ttest_rel(wide[g1].values, wide[g2].values)
        raw_pvals.append(p)
        t_stats.append(t)

    _, corrected_pvals, _, _ = multipletests(raw_pvals, method=correction_method)

    pair_results = {}
    print(f"\nPost-hoc paired t-tests ({correction_method} corrected):")
    for (g1, g2), t, p_raw, p_corr in zip(pairs, t_stats, raw_pvals, corrected_pvals):
        # Mean difference: g1 - g2 (higher - lower on x-axis, i.e. worse - better)
        mean_diff = wide[g1].mean() - wide[g2].mean()
        pair_results[(g1, g2)] = {'p_corr': p_corr, 'p_raw': p_raw, 't': t, 'mean_diff': mean_diff}
        sig = 'n.s.' if p_corr > 0.05 else ('*' if p_corr > 0.01 else ('**' if p_corr > 0.001 else '***'))
        print(f"  {g1} vs {g2}: t={t:.2f}, mean_diff={mean_diff:.4f}, "
              f"p_raw={p_raw:.4f}, p_corr={p_corr:.4f} {sig}")

    # --- Plot ---
    if colors is None:
        colors = sns.color_palette('Set2', n_levels)
    color_map   = {level: colors[i] for i, level in enumerate(x_levels)}
    x_positions = {level: i for i, level in enumerate(x_levels)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5 * n_levels, 7))
        created_fig = True
    else:
        created_fig = False

    # Connecting lines across subjects
    for subject in wide.index:
        ys = [wide.loc[subject, level] for level in x_levels]
        xs = [x_positions[level] for level in x_levels]
        ax.plot(xs, ys, color='grey', alpha=0.4, linewidth=0.8, zorder=1)

    # Dots and group means
    for level in x_levels:
        i    = x_positions[level]
        vals = wide[level].values
        ax.scatter([i] * len(vals), vals, color=color_map[level],
                   zorder=2, s=60, alpha=0.9, label=level)
        ax.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()],
                color=color_map[level], linewidth=2.5, zorder=3)

    # --- Bracket annotations for significant pairs only ---
    y_range      = wide.values.max() - wide.values.min()
    y_max        = wide.values.max()
    bracket_step = 0.06 * y_range

    sig_pairs = [(pair, res) for pair, res in pair_results.items() if res['p_corr'] <= 0.05]
    sig_pairs_sorted = sorted(sig_pairs,
                              key=lambda item: abs(x_positions[item[0][1]] - x_positions[item[0][0]]))

    for rank, ((g1, g2), res) in enumerate(sig_pairs_sorted):
        sig   = '*' if res['p_corr'] > 0.01 else ('**' if res['p_corr'] > 0.001 else '***')
        x1    = x_positions[g1]
        x2    = x_positions[g2]
        y_ann = y_max + bracket_step * (rank + 1)

        ax.plot([x1, x1, x2, x2],
                [y_ann - 0.2 * bracket_step, y_ann, y_ann, y_ann - 0.2 * bracket_step],
                color='black', linewidth=1)
        ax.text((x1 + x2) / 2, y_ann + 0.02 * y_range,
                f"{sig}\nΔ={res['mean_diff']:.3f}",
                ha='center', va='bottom', fontsize=9)

    # Adjust y-axis to include brackets
    if sig_pairs_sorted:
        y_max_with_brackets = y_max + bracket_step * (len(sig_pairs_sorted) + 1)
        ax.set_ylim(top=y_max_with_brackets)

    ax.set_title(f'Repeated measures ANOVA: {y} by {x}\n'
                 f'F={f_stat:.2f}, p={p_anova:.4f}, ηg²={ng2:.3f}\n'
                 f'{sphericity_str} | Post-hoc: {correction_method} corrected',
                 fontsize=10)
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(x_levels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if created_fig:
        plt.tight_layout()
        plt.show()

    return anova_result, pair_results, ax
    