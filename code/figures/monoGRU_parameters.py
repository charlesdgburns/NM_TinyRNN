"""Equation-style parameter figures extracted from monoGRU_mechanisms.ipynb."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm

def plot_model_equations(model, equations, cmap_name="coolwarm", cell=0.26,
                          row_h=0.42, fontsize=17, vec_fontsize=16, x0=0.3,
                          row_gap=0.55, colorbar=True, use_latex_font=True,
                          icon_pad=0.09, vector_pad=0.09, vector_text_pad=0.16,
                          dpi=300, colorbar_width=0.45, colorbar_height=0.14,
                          colorbar_fontsize=12):
    """
    Render a set of gate/update equations for `model`, with each weight
    matrix drawn inline as a small colour-coded icon on a shared,
    zero-centred colour scale (matching your existing parameter plot).

    Parameters
    ----------
    model : any object with .named_parameters() yielding (name, tensor)
        (a torch.nn.Module works directly).
    equations : list of dicts, one per equation row, each:
        {
          "lhs": "n_t",                 # left-hand side, mathtext (no $)
          "activation": r"\\phi",       # wrapped in ( ... ); plain words
                                         # like "Softmax" are auto-\\mathrm'd
          "terms": [
              {"param": "rnn.W_ih", "transpose": True,
               "vector_labels": ["c_t", "a_t", "o_t"]},
              {"param": "rnn.W_hh", "transpose": True,
               "vector_labels": ["h_{1,t-1}", "h_{2,t-1}"]},
              {"param": "rnn.bias_h"},   # lone icon, no vector -> bias term
          ],
        }
        "transpose" is needed whenever the stored weight is (in, out)
        rather than the (out, in) convention used for W @ x in the
        equation (torch.nn.Linear-style weights are already (out, in)
        and don't need it). Scalars and 1-D params (biases) are reshaped
        automatically.
    cmap_name, cell, row_h, fontsize, vec_fontsize, x0, row_gap : layout knobs.
    colorbar : whether to draw a shared colour scale at the top.
    use_latex_font : if True (default), render all text with matplotlib's
        Computer Modern mathtext font ('cm'), which closely matches real
        LaTeX output, instead of the default sans-serif mathtext.
    icon_pad : gap (inches) between a matrix icon's cells and its brackets.
    vector_pad : gap (inches) between a vector's text and its brackets.
    vector_text_pad : left-inset (inches) of vector labels from the
        left bracket (keep >= vector_pad so labels clear the bracket tick).
    dpi : resolution of the returned figure (affects both plt.show() and
        any savefig() that doesn't override dpi itself).
    colorbar_width, colorbar_height : size of the colour scale, as a
        fraction of the figure width / an absolute size in inches.
    colorbar_fontsize : font size for the colorbar's tick labels and title
        (the title is drawn 1pt larger).

    Returns
    -------
    matplotlib.figure.Figure
    """
    params = dict(model.named_parameters())

    if use_latex_font:
        # Persists for the session (mathtext is rendered lazily, at
        # show()/savefig() time, so scoping this to only the function
        # body wouldn't survive until then). Call
        # matplotlib.rcdefaults() to undo, or pass use_latex_font=False.
        plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

    # ---- resolve every referenced parameter into a plain 2D numpy array ----
    def resolve(term):
        p = params[term["param"]].detach().cpu().numpy()
        if p.ndim == 0:
            p = p.reshape(1, 1)
        elif p.ndim == 1:
            p = p.reshape(-1, 1)
        if term.get("transpose") and p.ndim == 2:
            p = p.T
        return p

    for eq in equations:
        for term in eq["terms"]:
            term["_matrix"] = resolve(term)

    all_vals = np.concatenate(
        [t["_matrix"].ravel() for eq in equations for t in eq["terms"]]
    )
    vmax = float(np.abs(all_vals).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # ---- low-level drawing helpers (inch-based: 1 data unit == 1 inch) ----
    def width(ax, artist):
        fig = ax.figure
        fig.canvas.draw()
        b = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0_, _), (x1_, _) = ax.transData.inverted().transform([[b.x0, b.y0], [b.x1, b.y1]])
        return x1_ - x0_

    def bracket(ax, x, y0, y1, side):
        t = 0.05 if side == "left" else -0.05
        p = Path([(x + t, y1), (x, y1), (x, y0), (x + t, y0)],
                  [Path.MOVETO] + [Path.LINETO] * 3)
        ax.add_patch(PathPatch(p, fc="none", ec="0.15", lw=1.6,
                                capstyle="round", zorder=5))

    def icon(ax, x, yc, M):
        nr, nc = M.shape
        w, h = nc * cell, nr * cell
        yt = yc + h / 2
        for r in range(nr):
            for c in range(nc):
                ax.add_patch(Rectangle(
                    (x + c * cell, yt - (r + 1) * cell), cell - 0.012, cell - 0.012,
                    fc=cmap(norm(M[r, c])), ec="white", lw=0.7, zorder=4,
                ))
        bracket(ax, x - icon_pad, yt - h, yt, "left")
        bracket(ax, x + w + icon_pad, yt - h, yt, "right")
        return w + 2 * icon_pad + 0.06

    def vector(ax, x, yc, labels):
        n = len(labels)
        h = n * row_h
        yt = yc + h / 2
        texts = [
            ax.text(x + vector_text_pad, yt - (i + 0.5) * row_h, f"${l}$",
                    ha="left", va="center", fontsize=vec_fontsize, zorder=5)
            for i, l in enumerate(labels)
        ]
        w = max(width(ax, t) for t in texts) + vector_text_pad
        bracket(ax, x - vector_pad, yt - h, yt, "left")
        bracket(ax, x + w + vector_pad, yt - h, yt, "right")
        return w + 2 * vector_pad + 0.06

    def text(ax, x, yc, s, fs):
        t = ax.text(x, yc, s, ha="left", va="center", fontsize=fs, zorder=5)
        return width(ax, t) + 0.05

    def equation_row(ax, yc, lhs, terms, activation):
        act = activation if activation.lstrip().startswith("\\") or "{" in activation \
            else rf"\mathrm{{{activation}}}"
        x = x0
        x += text(ax, x, yc, f"${lhs} = {act}$", fontsize)
        x += text(ax, x, yc, "$($", fontsize * 1.9)
        for i, term in enumerate(terms):
            if i:
                x += text(ax, x, yc, "$+$", fontsize) + icon_pad
            x += icon(ax, x, yc, term["_matrix"])
            if term.get("vector_labels"):
                x += vector(ax, x, yc, term["vector_labels"])
        x += text(ax, x, yc, "$)$", fontsize * 1.9)
        return x

    # ---- vertical layout ----
    def term_height(term):
        h_icon = term["_matrix"].shape[0] * cell
        h_vec = len(term.get("vector_labels", [])) * row_h if term.get("vector_labels") else 0
        return max(h_icon, h_vec)

    row_heights = [max(term_height(t) for t in eq["terms"]) for eq in equations]
    cbar_top_gap, cbar_bottom_gap = 0.15, 0.55  # inches: above / below the bar itself
    top_margin = (cbar_top_gap + colorbar_height + cbar_bottom_gap) if colorbar else 0.2
    bottom_margin = 0.2
    total_h = sum(row_heights) + row_gap * (len(equations) - 1) + top_margin + bottom_margin

    # draw on a generously wide scratch canvas, then crop to actual content
    scratch_w = 22.0
    fig = plt.figure(figsize=(scratch_w, total_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, scratch_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    y = total_h - top_margin
    max_x = 0.0
    for eq, rh in zip(equations, row_heights):
        yc = y - rh / 2
        x_end = equation_row(ax, yc, eq["lhs"], eq["terms"], eq["activation"])
        max_x = max(max_x, x_end)
        y -= rh + row_gap

    final_w = max_x + 0.3
    fig.set_size_inches(final_w, total_h)
    ax.set_xlim(0, final_w)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.06, 1 - (cbar_top_gap + colorbar_height) / total_h,
                             colorbar_width, colorbar_height / total_h])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Parameter value", fontsize=colorbar_fontsize + 1)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    return fig
