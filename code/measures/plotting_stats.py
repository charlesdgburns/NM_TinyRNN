'''Here are a few general functions for plotting some stats.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pingouin as pg
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multitest import multipletests
from itertools import combinations

from scipy.stats import ttest_rel, wilcoxon, shapiro

def compute_paired_stats(df, y, x, within_variable, paired_across, 
                         mean_across=None, filter_by=None, hue_order=None,
                         correction_method='fdr_bh', alpha=0.05):
    """
    Helper function: Performs data cleaning, groups by within_variable, extracts differences,
    checks for normality (Shapiro-Wilk) and outliers (IQR), and selects the 
    appropriate parametric (t-test) or non-parametric (Wilcoxon) statistical test.
    """
    df = df.copy()

    # --- Filter and Aggregation ---
    if filter_by:
        for col, val in filter_by.items():
            df = df[df[col] == val]

    if mean_across:
        mean_across = [mean_across] if isinstance(mean_across, str) else mean_across
        group_cols = [paired_across, within_variable, x]
        df = df.groupby(group_cols)[y].mean().reset_index()

    # --- Validation ---
    if df[x].nunique() != 2:
        x_vals = sorted(df[x].unique())
        raise ValueError(f"'{x}' must have exactly 2 levels for paired testing, found: {x_vals}")

    within_levels = hue_order if hue_order is not None else sorted(df[within_variable].unique())
    
    # --- Statistical Testing Loop ---
    all_paired = []
    raw_pvals = []
    test_meta = []
    x_levels = None
    
    for group in within_levels:
        group_df = df[df[within_variable] == group]
        paired = group_df.pivot(index=paired_across, columns=x, values=y).dropna()

        if x_levels is None and not paired.empty:
            x_levels = sorted(paired.columns.tolist())
        
        all_paired.append(paired)

        if paired.empty or len(paired) < 3:
            raw_pvals.append(np.nan)
            test_meta.append({"test": "None", "reason": "Insufficient Data", "norm_p": np.nan, "has_outliers": False})
        else:
            diffs = paired[x_levels[1]].values - paired[x_levels[0]].values
            
            # Shapiro-Wilk Normality test
            _, norm_p = shapiro(diffs)
            is_normal = norm_p > alpha
            
            # Outlier Check via IQR Rule on differences
            q25, q75 = np.percentile(diffs, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            has_outliers = np.any((diffs < lower_bound) | (diffs > upper_bound))
            
            # Decision Tree Selection
            if not is_normal or has_outliers:
                stat_res = wilcoxon(paired[x_levels[0]].values, paired[x_levels[1]].values)
                chosen_test = "Wilcoxon Signed-Rank"
                reason = "Outliers detected" if has_outliers else "Non-normal differences"
            else:
                stat_res = ttest_rel(paired[x_levels[0]].values, paired[x_levels[1]].values)
                chosen_test = "Paired t-test"
                reason = "Normal distribution, no outliers"
                
            raw_pvals.append(stat_res.pvalue)
            test_meta.append({
                "test": chosen_test, 
                "reason": reason, 
                "norm_p": norm_p, 
                "has_outliers": has_outliers
            })

    # --- Multiple Comparisons Correction ---
    valid_mask = ~np.isnan(raw_pvals)
    corrected_pvals = np.full(len(raw_pvals), np.nan)
    if valid_mask.sum() > 1:
        _, corrected_pvals[valid_mask], _, _ = multipletests(
            np.array(raw_pvals)[valid_mask], method=correction_method)
    else:
        corrected_pvals[valid_mask] = np.array(raw_pvals)[valid_mask]

    # Package diagnostic output dataframe
    stats_summary = []
    for lvl, p_raw, p_corr, meta in zip(within_levels, raw_pvals, corrected_pvals, test_meta):
        stats_summary.append({
            "group": lvl, "test_used": meta["test"], "reason": meta["reason"],
            "shapiro_p": meta["norm_p"], "outliers_present": meta["has_outliers"],
            "p_raw": p_raw, "p_corrected": p_corr
        })
        
    return all_paired, x_levels, within_levels, corrected_pvals, pd.DataFrame(stats_summary)


def plot_paired_comparison(df,
                           y,
                           x,
                           within_variable,
                           paired_across,
                           mean_across=None,
                           filter_by=None,
                           dodge_offset=0.2,
                           colors=None,
                           hue_order=None,
                           correction_method='fdr_bh',
                           ax=None,
                           title = None):
    """
    Plots a highly-stylized publication-ready paired comparison plot.
    Automatically runs normality and outlier diagnostics on the fly, 
    selecting between Paired t-test and Wilcoxon Signed-Rank tests cleanly.

    Returns
    -------
    ax    : matplotlib axis object
    report_df : Pandas DataFrame containing the background diagnostic and test metrics
    """
    # 1. Delegate statistical selection and execution to backend engine
    all_paired, x_levels, within_levels, corrected_pvals, report_df = compute_paired_stats(
        df=df, y=y, x=x, within_variable=within_variable, paired_across=paired_across,
        mean_across=mean_across, filter_by=filter_by, hue_order=hue_order, 
        correction_method=correction_method
    )
    
    # 2. Print clear background console logging table for immediate verification
    print("\n" + "="*50 + "\nPAIRED COMPARISON DIAGNOSTIC RUN REPORT\n" + "="*50)
    print(report_df[["group", "test_used", "reason", "outliers_present", "p_corrected"]].to_string(index=False))
    print("="*50 + "\n")

    # 3. Setup Plot Configuration parameters safely
    n_groups = len(within_levels)
    all_y_vals = np.concatenate([p.values.flatten() for p in all_paired if not p.empty])
    y_range = all_y_vals.max() - all_y_vals.min()

    if colors is None:
        colors = ['#E66101', '#5E3C99', '#FDB863'] if n_groups <= 3 else plt.cm.tab10.colors

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5 * n_groups, 5.5))
        created_fig = True
    else:
        created_fig = False

    # Structural baseline styling layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    xticks_positions = []
    xticks_labels = []

    # 4. Rendering Visualization Loop
    for i, (group, paired, p_val_corr) in enumerate(zip(within_levels, all_paired, corrected_pvals)):
        if paired.empty:
            continue

        vals0 = paired[x_levels[0]].values
        vals1 = paired[x_levels[1]].values
        x0, x1 = i - dodge_offset, i + dodge_offset
        
        xticks_positions.extend([x0, x1])
        xticks_labels.extend([str(x_levels[0]), str(x_levels[1])])

        group_color = colors[i % len(colors)]

        # Light background pairing lines
        for v0, v1 in zip(vals0, vals1):
            ax.plot([x0, x1], [v0, v1], color='#B0B0B0', alpha=0.4, linewidth=1.0, zorder=1)

        # Main Data Scatter points
        ax.scatter([x0] * len(vals0), vals0, color=group_color, edgecolor='none', zorder=2, s=60, alpha=0.85)
        ax.scatter([x1] * len(vals1), vals1, color=group_color, edgecolor='none', zorder=2, s=60, alpha=0.85)

        # Text/Bracket Annotation System
        if not np.isnan(p_val_corr):
            sig = 'n.s.' if p_val_corr > 0.05 else ('*' if p_val_corr > 0.01 else ('**' if p_val_corr > 0.001 else '***'))
            y_max = max(vals0.max(), vals1.max())
            y_ann = y_max + 0.02 * y_range
            
            ax.plot([x0, x1], [y_ann, y_ann], color='#333333', linewidth=1.2, zorder=3)
            ax.text((x0 + x1) / 2, y_ann + 0.008 * y_range,
                    f'{sig}\np = {p_val_corr:.3f}', ha='center', va='bottom', fontsize=9, color='#333333')

        # Multi-Indexed Bracket Layout Architecture placement
        ax_xmin = (x0 - (-0.5)) / (n_groups)
        ax_xmax = (x1 - (-0.5)) / (n_groups)
        ax_mid  = (i - (-0.5)) / (n_groups)
        
        bracket_y = -0.12
        bracket_tick = 0.02
        
        ax.plot([ax_xmin, ax_xmin, ax_xmax, ax_xmax], 
                [bracket_y + bracket_tick, bracket_y, bracket_y, bracket_y + bracket_tick], 
                color='#555555', linewidth=1.2, transform=ax.transAxes, clip_on=False)
        
        ax.text(ax_mid, bracket_y - 0.02, group, 
                ha='center', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes, clip_on=False)

    # 5. Final Axis Frame Formatting Details
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels, fontsize=10)
    
    ax.set_xlabel(x, fontsize=11, labelpad=40)  
    ax.set_ylabel(y, fontsize=11, fontweight='bold')
    if title is None:
        title = f'Paired Analysis of {y} by {x}'
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold', loc='left')

    if created_fig:
        plt.tight_layout()
        plt.show()
        
    return ax, report_df
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

    sphericity_cols = [c for c in anova_result.columns if 'spher' in c.lower()]
    if sphericity_cols:
        sphericity_col = sphericity_cols[0]
        sphericity_violated = not anova_result[sphericity_col].values[0]
    else:
        sphericity_col = None
        sphericity_violated = False

    if sphericity_violated:
        gg_cols = [c for c in anova_result.columns if 'gg' in c.lower()]
        p_anova = anova_result[gg_cols[0]].values[0] if gg_cols else anova_result[[c for c in anova_result.columns if 'p_unc' in c.lower() or 'p-unc' in c.lower() or c.lower() == 'p']][0]
        sphericity_str = (f'Sphericity violated ({sphericity_col}={anova_result[sphericity_col].values[0]:.3f}), '
                          f'p={p_anova:.4f}, GG corrected')
    else:
        p_unc_cols = [c for c in anova_result.columns
                      if 'p_unc' in c.lower() or 'p-unc' in c.lower() or c.lower() == 'p']
        p_col = p_unc_cols[0]
        p_anova = anova_result[p_col].values[0]
        sphericity_str = ('Sphericity assumption met'
                          if sphericity_col else 'Sphericity not tested / not available')

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
    bracket_step = 0.2 * y_range

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
    