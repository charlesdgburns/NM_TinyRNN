"""Trial validation and the original binary-choice likelihood conventions.

Choice 0 is left (poke_4); choice 1 is right (poke_6). Every input, including
forced and unscored trials, updates model state. Masks affect scoring only.
"""

import sys

import numpy as np

LOG_MAX_FLOAT = np.log(sys.float_info.max / 2.1)
log_max_float = LOG_MAX_FLOAT  # Original public spelling.


def _binary_array(values, name):
    """Validate a one-dimensional Boolean or numeric 0/1 array."""
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must be a one-dimensional 0/1 array")
    return values.astype(int, copy=True)


def _choice_array(values):
    values = np.asarray(values)
    if values.ndim == 1 and np.isin(values, ("poke_4", "poke_6")).all():
        return (values == "poke_6").astype(int)
    return _binary_array(values, "choices")


def _column(trials_df, name, required=True):
    """Accept original two-level trial columns or a simple flat DataFrame."""
    if getattr(trials_df.columns, "nlevels", 1) == 1:
        key = name
    elif trials_df.columns.nlevels == 2:
        key = (name, "")
    else:
        raise ValueError("trials_df must have flat or two-level columns")
    if key not in trials_df.columns:
        if required:
            raise ValueError(f"trials_df is missing {name!r}")
        return None
    values = trials_df[key]
    if getattr(values, "ndim", None) != 1:
        raise ValueError(f"trials_df has duplicate or ambiguous {name!r} columns")
    return values.to_numpy()


def get_trial_data(session, *, apply_ll_mask=True):
    """Return choices, outcomes and the free-choice scoring mask.

    Sessions may supply ``choices``, ``outcomes`` and ``free_choice`` arrays,
    or the original ``trials_df`` with choice/outcome/forced_choice columns.
    An absent forced-choice flag is rejected; callers must explicitly declare
    all-free data. ``ll_mask`` never changes learning or trial order.
    """
    if hasattr(session, "choices") and hasattr(session, "outcomes"):
        choices = _choice_array(session.choices)
        outcomes = _binary_array(session.outcomes, "outcomes")
        if hasattr(session, "free_choice"):
            free_choice = _binary_array(session.free_choice, "free_choice").astype(bool)
        elif hasattr(session, "forced_choice"):
            free_choice = ~_binary_array(session.forced_choice, "forced_choice").astype(bool)
        else:
            raise ValueError("Session must explicitly supply free_choice or forced_choice")
        if hasattr(session, "forced_choice"):
            forced = _binary_array(session.forced_choice, "forced_choice").astype(bool)
            if not np.array_equal(free_choice, ~forced):
                raise ValueError("free_choice and forced_choice disagree")
    elif hasattr(session, "trials_df"):
        choices = _choice_array(_column(session.trials_df, "choice"))
        outcomes = _binary_array(_column(session.trials_df, "outcome"), "outcomes")
        free_values = _column(session.trials_df, "free_choice", required=False)
        forced_values = _column(session.trials_df, "forced_choice", required=False)
        if free_values is None and forced_values is None:
            raise ValueError("trials_df must explicitly supply free_choice or forced_choice")
        if free_values is not None:
            free_choice = _binary_array(free_values, "free_choice").astype(bool)
        else:
            free_choice = ~_binary_array(forced_values, "forced_choice").astype(bool)
        if forced_values is not None:
            forced = _binary_array(forced_values, "forced_choice").astype(bool)
            if not np.array_equal(free_choice, ~forced):
                raise ValueError("free_choice and forced_choice disagree")
    else:
        raise TypeError("Session must supply trial arrays or trials_df")
    n_trials = len(choices)
    if not n_trials or len(outcomes) != n_trials or len(free_choice) != n_trials:
        raise ValueError("Trial arrays must have equal, positive length")
    if hasattr(session, "n_trials") and session.n_trials != n_trials:
        raise ValueError("session.n_trials differs from the trial-array length")
    if apply_ll_mask and getattr(session, "ll_mask", None) is not None:
        ll_mask = _binary_array(session.ll_mask, "ll_mask").astype(bool)
        if len(ll_mask) != n_trials:
            raise ValueError("ll_mask must have one element per original trial")
        free_choice &= ll_mask
    return choices, outcomes, free_choice


def softmax_probs(Q, iTemp):
    """Original binary softmax: Q has shape (2, n_trials), iTemp is nonnegative.

    The right probability is computed as 1 - P(left), deliberately retaining
    the original rounding and saturation convention for likelihood parity.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != 2 or not np.isfinite(Q).all():
        raise ValueError("Q must be a finite (2, n_trials) array")
    if not np.isfinite(iTemp) or iTemp < 0:
        raise ValueError("iTemp must be finite and nonnegative")
    with np.errstate(over="ignore"):
        difference = iTemp * (Q[1] - Q[0])
    if np.isnan(difference).any():
        raise ValueError("Nonfinite decision difference")
    difference = np.minimum(difference, LOG_MAX_FLOAT)
    probabilities = np.empty_like(Q)
    probabilities[0] = 1.0 / (1.0 + np.exp(difference))
    probabilities[1] = 1.0 - probabilities[0]
    return probabilities


def log_safe(x):
    """Original likelihood floor; this is not clipping to machine epsilon."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return np.log(0.5e-200 + (1 - 1e-200) * x)
