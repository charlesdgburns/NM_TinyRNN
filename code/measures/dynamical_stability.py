"""
Fixed-point / linear-stability analysis of a network's candidate-state
update

    n_t = ReLU( W_ih^T x + W_hh^T h_{t-1} + b_h )

for a set of fixed inputs x, treating it as an autonomous discrete-time
dynamical system h_{t+1} = F(h_t; x) with h_{t-1} replaced by the network's
own previous candidate state (i.e. iterating n_t on itself).

Why this can be solved analytically
------------------------------------
ReLU is piecewise-linear: for any given hidden state, each unit is either
"active" (pre-activation > 0, so it behaves as an affine map for that unit)
or "inactive" (clamped to 0). For a hidden size of H there are 2^H such
linear regions. Within a fixed region S (the set of active units), the
fixed-point equation

    h*_S = (W_hh)_{S,S} h*_S + b_eff_S ,   h*_{not S} = 0

is linear and solvable directly. We enumerate every region, solve it, and
keep only the solutions that are *self-consistent* (i.e. the solved point
actually satisfies the sign pattern it assumed). This is exact, not an
iterative/numerical root-find, and it also gives us every fixed point,
not just the one closest to some initial guess.

Stability
---------
This is a discrete-time map, so local stability at a fixed point is
governed by the *magnitude* of the Jacobian's eigenvalues (spectral
radius), not their sign: stable if all |eigenvalue| < 1, unstable if all
> 1, and a saddle if some are above and some below 1. The Jacobian in
region S is simply diag(mask_S) @ W_hh, since d(ReLU)/dz is 1 for active
units and 0 for inactive ones.
"""

import itertools
import numpy as np


# --------------------------------------------------------------------------
# pulling the right weights out of the model
# --------------------------------------------------------------------------

def get_candidate_state_weights(model, w_ih_name="rnn.W_ih", w_hh_name="rnn.W_hh",
                                 bias_name="rnn.bias_h", transpose=True):
    """
    Extract W_ih, W_hh, b_h from `model.named_parameters()` and orient them
    as (hidden, input) / (hidden, hidden) so that
        n_t = relu(W_ih @ x + W_hh @ h + b_h).

    Set `transpose=False` if your parameters are already stored that way
    (e.g. torch.nn.Linear-style weights); leave it True if, like the model
    in this notebook, they're stored (in_features, out_features).
    """
    params = dict(model.named_parameters())
    W_ih = params[w_ih_name].detach().cpu().numpy()
    W_hh = params[w_hh_name].detach().cpu().numpy()
    b_h = params[bias_name].detach().cpu().numpy()
    if transpose:
        W_ih = W_ih.T
        W_hh = W_hh.T
    b_h = np.atleast_1d(b_h).reshape(-1)
    return W_ih.astype(float), W_hh.astype(float), b_h.astype(float)


# --------------------------------------------------------------------------
# core dynamics
# --------------------------------------------------------------------------

def relu(z):
    return np.maximum(z, 0.0)


def step(h, x, W_ih, W_hh, b_h):
    """One update of the candidate-state map: h_{t+1} = F(h_t; x)."""
    return relu(W_ih @ x + W_hh @ h + b_h)


def find_fixed_points(x, W_ih, W_hh, b_h, tol=1e-8):
    """
    Enumerate every linear region of the ReLU map for fixed input `x` and
    solve for a fixed point within it analytically, keeping only
    self-consistent solutions.

    Returns a list of dicts, each with:
        'h'               fixed-point hidden state, shape (H,)
        'active_mask'     bool array, which units are active (> 0) there
        'jacobian'        local Jacobian dF/dh at the fixed point
        'eigenvalues'     eigenvalues of the Jacobian
        'spectral_radius' max |eigenvalue|
        'stability'       'stable' / 'unstable' / 'saddle' / 'marginal'
    """
    x = np.asarray(x, dtype=float)
    H = W_hh.shape[0]
    b_eff = W_ih @ x + b_h  # fold the fixed input into an effective bias
    I = np.eye(H)
    fixed_points = []

    for pattern in itertools.product([0, 1], repeat=H):
        mask = np.array(pattern, dtype=float)
        active = mask.astype(bool)

        h = np.zeros(H)
        if active.any():
            A_sub = W_hh[np.ix_(active, active)]
            b_sub = b_eff[active]
            M = I[np.ix_(active, active)] - A_sub
            if abs(np.linalg.det(M)) < tol:
                continue  # singular in this region: no isolated fixed point
            h[active] = np.linalg.solve(M, b_sub)

        # self-consistency check: does h actually respect the sign pattern
        # ('pattern') it was solved under?
        pre_activation = W_hh @ h + b_eff
        ok_active = np.all(pre_activation[active] > -tol) if active.any() else True
        ok_inactive = np.all(pre_activation[~active] <= tol) if (~active).any() else True
        if not (ok_active and ok_inactive):
            continue

        jacobian = mask[:, None] * W_hh  # == diag(mask) @ W_hh
        eigenvalues = np.linalg.eigvals(jacobian)
        mags = np.abs(eigenvalues)
        spectral_radius = float(mags.max()) if len(mags) else 0.0

        if spectral_radius < 1 - 1e-9:
            stability = "stable"
        elif spectral_radius > 1 + 1e-9:
            stability = "unstable"
        else:
            stability = "marginal"
        if len(mags) > 1 and mags.min() < 1 - 1e-9 and mags.max() > 1 + 1e-9:
            stability = "saddle"

        fixed_points.append(dict(
            h=h, active_mask=active, jacobian=jacobian,
            eigenvalues=eigenvalues, spectral_radius=spectral_radius,
            stability=stability,
        ))

    # de-duplicate points that coincide at region boundaries
    unique = []
    for fp in fixed_points:
        if not any(np.allclose(fp["h"], u["h"], atol=1e-6) for u in unique):
            unique.append(fp)
    return unique


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _format_eigs(eigenvalues):
    parts = []
    for e in eigenvalues:
        if abs(e.imag) > 1e-9:
            parts.append(f"{e.real:+.3f}{e.imag:+.3f}j")
        else:
            parts.append(f"{e.real:+.3f}")
    return ", ".join(parts)


def analyze_candidate_state(model, inputs, w_ih_name="rnn.W_ih",
                             w_hh_name="rnn.W_hh", bias_name="rnn.bias_h",
                             transpose=True, verbose=True):
    """
    Run the full fixed-point / stability analysis of the candidate-state
    update for every input in `inputs`.

    Returns a dict: {tuple(x): [fixed_point_dict, ...], ...}
    """
    W_ih, W_hh, b_h = get_candidate_state_weights(
        model, w_ih_name, w_hh_name, bias_name, transpose
    )
    results = {}
    for x in inputs:
        x = np.asarray(x, dtype=float)
        fps = find_fixed_points(x, W_ih, W_hh, b_h)
        results[tuple(x.tolist())] = fps
        if verbose:
            print(f"\nInput x = {x.tolist()}")
            if not fps:
                print("  no fixed points in this region set")
                continue
            for i, fp in enumerate(fps):
                active_str = "".join("1" if a else "0" for a in fp["active_mask"])
                print(f"  fixed point {i + 1}: h* = {np.round(fp['h'], 4).tolist()}"
                      f"   (active units: {active_str})")
                print(f"      eigenvalues = [{_format_eigs(fp['eigenvalues'])}]"
                      f"   spectral radius = {fp['spectral_radius']:.4f}"
                      f"   -> {fp['stability']}")
    return results


# --------------------------------------------------------------------------
# optional: 2D phase portrait (only meaningful when hidden size == 2)
# --------------------------------------------------------------------------

def plot_phase_portraits(model, inputs, w_ih_name="rnn.W_ih", w_hh_name="rnn.W_hh",
                          bias_name="rnn.bias_h", transpose=True,
                          span=3.0, grid=21, figsize=(9, 8), dpi=200):
    """
    For a hidden size of 2, draw the vector field h -> F(h; x) - h and mark
    the fixed points (colour-coded by stability) for each input, one panel
    per input. Returns the matplotlib Figure.

    The view is restricted to the positive quadrant (h1, h2 >= 0): since
    every h here is itself the output of a ReLU (either the previous
    candidate state, or the initial state), the dynamics never actually
    visit negative coordinates, so that region is not reachable and isn't
    shown.
    """
    import matplotlib.pyplot as plt

    W_ih, W_hh, b_h = get_candidate_state_weights(
        model, w_ih_name, w_hh_name, bias_name, transpose
    )
    H = W_hh.shape[0]
    if H != 2:
        raise ValueError(f"plot_phase_portraits only supports hidden size 2, got {H}")

    colors = {"stable": "tab:blue", "unstable": "tab:red",
              "saddle": "tab:orange", "marginal": "tab:gray"}

    n = len(inputs)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.ravel()

    grid_vals = np.linspace(0.0, span, grid)
    Hh1, Hh2 = np.meshgrid(grid_vals, grid_vals)

    for ax, x in zip(axes, inputs):
        x = np.asarray(x, dtype=float)
        b_eff = W_ih @ x + b_h
        # vector field: F(h) - h, vectorised over the grid
        pts = np.stack([Hh1.ravel(), Hh2.ravel()], axis=1)  # (N, 2)
        nxt = relu(pts @ W_hh.T + b_eff)                     # (N, 2)
        delta = nxt - pts
        U = delta[:, 0].reshape(Hh1.shape)
        V = delta[:, 1].reshape(Hh1.shape)
        speed = np.hypot(U, V)

        ax.streamplot(Hh1, Hh2, U, V, color=speed, cmap="Greys", density=1.1,
                       linewidth=0.8, arrowsize=0.8)

        fps = find_fixed_points(x, W_ih, W_hh, b_h)
        for fp in fps:
            ax.scatter(*fp["h"], s=90, color=colors.get(fp["stability"], "k"),
                       edgecolor="k", zorder=5, linewidth=1.2)

        ax.set_title(f"x = {x.tolist()}", fontsize=11)
        ax.set_xlabel("$h_1$"); ax.set_ylabel("$h_2$")
        ax.set_xlim(0, span); ax.set_ylim(0, span)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_aspect("equal")

    for ax in axes[n:]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markeredgecolor="k", markersize=9, label=s)
               for s, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Candidate-state ($n_t$) dynamics under fixed inputs", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


# --------------------------------------------------------------------------
# self-test with fake parameters (swap FakeModel for your real model)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    class FakeParam:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=float)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._arr

    class FakeModel:
        def __init__(self, params):
            self._params = params
        def named_parameters(self):
            return self._params.items()

    rng = np.random.default_rng(3)
    model = FakeModel({
        "rnn.W_ih": FakeParam(rng.uniform(-1, 1, (3, 2))),   # (input=3, hidden=2)
        "rnn.W_hh": FakeParam(rng.uniform(-1, 1, (2, 2))),   # (hidden=2, hidden=2)
        "rnn.bias_h": FakeParam(rng.uniform(-1, 1, (2,))),
    })

    inputs = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]

    results = analyze_candidate_state(model, inputs)

    fig = plot_phase_portraits(model, inputs)
    fig.savefig("/home/claude/phase_portraits.png", bbox_inches="tight")
    print("\nsaved phase_portraits.png")