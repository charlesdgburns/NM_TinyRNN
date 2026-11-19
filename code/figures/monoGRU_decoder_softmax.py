"""Decoder geometry of one two-unit monoGRU fit: hidden states -> softmax scores -> scores plus decoder bias.

Panels: (a) hidden states h; (b) scores z = W_decoder h; (c) z + b_decoder. Score axes are the softmax inputs
z_A (choice A = choice 0) and z_B. The background is the logit z_A - z_B, so the softmax output
P(A) = sigmoid(z_A - z_B) is read from the colour bar; the dashed diagonal is logit = 0 (P(A) = 0.5).
The same trials, evenly spaced in the unbiased logit, are drawn in every panel. The bias adds the same vector
to every score, drawn once as an arrow.

Used by code/analyses/monoGRU_decoder_bias.py, which reads the fit and calls `decoder_softmax_figure`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

GREY = '#4d4d4d'                 # 'with decoder bias' colour in figure_decoder_bias
STYLE = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.linewidth': .8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans'}
TITLES = [r'Hidden states $\mathbf{h}$', r'Softmax$(W_\mathrm{decoder}\,\mathbf{h})$',
          r'Softmax$(W_\mathrm{decoder}\,\mathbf{h}+\mathbf{b}_\mathrm{decoder})$']


def shown_trials(logit, n_show):
    """Indices of trials evenly spaced in the given logit (duplicates removed)."""
    return np.unique([int(np.argmin(np.abs(logit - v))) for v in np.linspace(logit.min(), logit.max(), n_show)])


def _cross_axes(ax, lim, label_x, label_y):
    for side in ('left', 'bottom'):
        ax.spines[side].set_position('zero'); ax.spines[side].set_color('black'); ax.spines[side].set_zorder(3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.text(1.02, .5, label_x, transform=ax.transAxes, ha='left', va='center', fontsize=8, clip_on=False)
    ax.text(.5, 1.03, label_y, transform=ax.transAxes, ha='center', va='bottom', fontsize=8, clip_on=False)
    ticks = [v for v in ax.get_xticks() if 0 < abs(v) < lim]; ax.set_xticks(ticks); ax.set_yticks(ticks)


def decoder_softmax_figure(hidden, weight, bias, label, n_show=12, score_lim=1.9, vmax=3.0, bias_label_xy=(.46, .08)):
    """Three-panel decoder-geometry figure.

    hidden: (trials, 2) hidden states; weight: (2 outputs, 2 units) decoder weight; bias: (2,) decoder bias;
    label: header text (e.g. 'With decoder bias: WS22, outer fold 10, inner fold 0'); bias_label_xy: score-space
    position of the '+b_decoder' label (default placed for WS22, outer fold 10, inner fold 0). Returns the figure.
    """
    hidden, weight, bias = (np.asarray(a, float) for a in (hidden, weight, bias))
    z0 = hidden @ weight.T; z1 = z0 + bias; l0 = z0[:, 0] - z0[:, 1]; l1 = z1[:, 0] - z1[:, 1]
    idx = shown_trials(l0, n_show)
    cmap, norm = plt.get_cmap('coolwarm'), Normalize(-vmax, vmax)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw=dict(wspace=.62, left=.06, right=.86, top=.76, bottom=.1))
        ax = axes[0]; _cross_axes(ax, 1.08 * np.abs(hidden).max(), 'Unit 1', 'Unit 2')
        ax.scatter(hidden[idx, 0], hidden[idx, 1], s=34, facecolor='#a6a6a6', edgecolor='white', lw=.7, zorder=4)
        grid = np.linspace(-score_lim, score_lim, 300); gx, gy = np.meshgrid(grid, grid)
        for ax, z, logit, with_bias in [(axes[1], z0, l0, False), (axes[2], z1, l1, True)]:
            ax.imshow(gx - gy, extent=[-score_lim, score_lim, -score_lim, score_lim], origin='lower', cmap=cmap, norm=norm, alpha=.4, zorder=0, interpolation='bilinear')
            _cross_axes(ax, score_lim, r'$z_A$', r'$z_B$'); ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1])
            ax.plot([-score_lim, score_lim], [-score_lim, score_lim], color='black', lw=.8, ls=(0, (4, 2.5)), zorder=1)
            ax.text(.64 * score_lim, .73 * score_lim, 'logit = 0', rotation=45, ha='center', va='center', fontsize=6.5, zorder=6)
            if with_bias:   # one arrow for the shared offset, displaced perpendicular to b so it does not cover the origin
                perp = np.array([bias[1], -bias[0]]) / np.linalg.norm(bias); perp = perp if perp.sum() > 0 else -perp
                start = .32 * perp
                ax.add_patch(FancyArrowPatch(tuple(start), tuple(start + bias), arrowstyle='-|>', mutation_scale=9, color='black', lw=1.1, zorder=6, shrinkA=0, shrinkB=0))
                ax.text(*bias_label_xy, r'$+\,\mathbf{b}_\mathrm{decoder}$', fontsize=7.5, ha='left', va='bottom', zorder=6)
            order = np.argsort(np.abs(logit[idx]))
            ax.scatter(z[idx[order], 0], z[idx[order], 1], s=34, c=cmap(norm(logit[idx[order]])), edgecolor='white', lw=.7, zorder=5)
        fig.canvas.draw(); renderer = fig.canvas.get_renderer(); inverse = fig.transFigure.inverted()
        for i in range(2):
            p = inverse.transform(axes[i].get_tightbbox(renderer)); q = inverse.transform(axes[i + 1].get_tightbbox(renderer))
            y = axes[i].get_position().y0 + axes[i].get_position().height / 2
            fig.add_artist(FancyArrowPatch((p[1, 0] + .008, y), (q[0, 0] - .008, y), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=10, color='black', lw=1))
        top = max(ax.get_position().y1 for ax in axes) + .1
        for ax, letter, title in zip(axes, 'abc', TITLES):
            p = ax.get_position()
            fig.text(p.x0 + p.width / 2, top, title, ha='center', va='bottom', fontsize=8.5)
            fig.text(p.x0 - .045, top, letter, ha='left', va='bottom', fontsize=10, fontweight='bold')
        fig.text(.02, .955, label, fontsize=8, fontweight='bold', color=GREY, ha='left', va='bottom')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.935, .16, .011, .52]), extend='both')
        colorbar.set_label(r'Logit $z_A - z_B$', fontsize=7); colorbar.ax.tick_params(labelsize=6)
        to_p = lambda x: 1 / (1 + np.exp(-x)); to_logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        prob_axis = colorbar.ax.secondary_yaxis('left', functions=(to_p, to_logit))
        prob_axis.set_ticks([.1, .5, .9]); prob_axis.tick_params(labelsize=6); prob_axis.set_ylabel('P(A)', fontsize=7, labelpad=1)
    return fig
