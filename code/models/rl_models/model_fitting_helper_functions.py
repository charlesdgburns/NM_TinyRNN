"""Fit one cognitive-model parameter vector per subject, with explicit reset/score identities.

The optimizer, priors and ML/MAP objectives follow Thomas Akam's RL fitting
notebook and the local adaptation. Trial sequences reset at session boundaries;
forced inputs update state, while only the model's eligible targets are scored.
"""

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

if __package__:
    from .model_implementation_helper_functions import get_trial_data
else:
    from model_implementation_helper_functions import get_trial_data


# Prior distributions also supply random optimizer starting points.
beta_prior = stats.beta(a=2, b=2)
gamma_prior = stats.gamma(a=2, scale=2)
norm_prior = stats.norm(scale=5)
_PRIORS = {"unit": beta_prior, "pos": gamma_prior, "unc": norm_prior}
_BOUNDS = {"unit": (0.0, 1.0), "pos": (0.0, None), "unc": (None, None)}


def get_init_param_values(param_ranges, rng=None):
    """Sample independent prior starting points without changing NumPy's global RNG."""
    rng = np.random.default_rng(rng)
    _validate_ranges(param_ranges)
    return np.array([_PRIORS[param_range].rvs(random_state=rng) for param_range in param_ranges], dtype=float)


def get_bounds(model):
    """Return model bounds, rejecting overrides outside the declared mathematical domain."""
    _validate_model(model)
    custom_bounds = getattr(model, "param_bounds", None) or {}
    unknown = set(custom_bounds) - set(model.param_names)
    if unknown:
        raise ValueError(f"Bounds refer to unknown parameters: {sorted(unknown)}")
    bounds = []
    for name, param_range in zip(model.param_names, model.param_ranges):
        natural_lower, natural_upper = _BOUNDS[param_range]
        bound = custom_bounds.get(name, _BOUNDS[param_range])
        if len(bound) != 2:
            raise ValueError(f"Bounds for {name} must be (lower, upper)")
        lower, upper = (None if value is None else float(value) for value in bound)
        if any(value is not None and not np.isfinite(value) for value in (lower, upper)):
            raise ValueError(f"Bounds for {name} must be finite or None")
        if natural_lower is not None and (lower is None or lower < natural_lower):
            raise ValueError(f"Lower bound for {name} leaves its {param_range} domain")
        if natural_upper is not None and (upper is None or upper > natural_upper):
            raise ValueError(f"Upper bound for {name} leaves its {param_range} domain")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound exceeds upper bound for {name}")
        bounds.append((lower, upper))
    return bounds


def log_prior_prob(param_values, param_ranges):
    """Return original independent log priors, floored at -1000 per parameter.

    Custom bounds constrain optimization; the original prior densities are retained
    without a truncation-normalization constant, matching the source procedure.
    """
    _validate_ranges(param_ranges)
    param_values = np.asarray(param_values, dtype=float)
    if param_values.shape != (len(param_ranges),) or not np.isfinite(param_values).all():
        raise ValueError("Prior parameter vector has the wrong shape or nonfinite entries")
    terms = np.array([_PRIORS[kind].logpdf(value) for value, kind in zip(param_values, param_ranges)])
    return float(np.maximum(terms, -1000.0).sum())


def multi_session_log_likelihood(sessions, model, param_values):
    """Sum likelihoods, restarting the model at the beginning of every supplied session."""
    return float(sum(model.session_log_likelihood(session, param_values) for session in sessions))


def scored_data_identity(sessions):
    """Return reset-aware behavior/mask identity and the actual eligible-target count.

    The hash includes ordered inputs, score masks and subject/session identifiers,
    not just counts. Session order is immaterial because each sequence resets.
    Changing a forced history input therefore also changes the comparison identity.
    """
    sessions = _validate_sessions(sessions)
    digest = hashlib.sha256()
    n_scored = 0
    for session in sorted(sessions, key=session_identity):
        choices, outcomes, scored = get_trial_data(session)
        digest.update(json.dumps(session_identity(session), ensure_ascii=False).encode("utf-8"))
        digest.update(source_session_id(session).encode("utf-8"))
        trial_ids = getattr(session, "trial_ids", None)
        if trial_ids is None:
            trials_df = getattr(session, "trials_df", None)
            trial_ids = np.arange(len(choices)) if trials_df is None else trials_df.index
        if len(trial_ids) != len(choices):
            raise ValueError("Trial identities must have the same length as the input sequence")
        digest.update(json.dumps([str(value) for value in trial_ids], ensure_ascii=False).encode("utf-8"))
        digest.update(np.array([len(choices)], dtype="<i8").tobytes())
        for array in (choices, outcomes, scored):
            digest.update(np.asarray(array, dtype=np.uint8).tobytes())
        n_scored += int(scored.sum())
    return dict(scored_targets_hash=digest.hexdigest(), n_free_choice_trials=n_scored)


def session_identity(session):
    """Return the explicit (subject, reset-sequence ID) pair used for fit matching."""
    info = getattr(session, "info", None)
    subject = getattr(info, "subject", None)
    session_id = getattr(info, "session_id", None)
    if subject is None or session_id is None or pd.isna(subject) or pd.isna(session_id):
        raise ValueError("Each session needs finite info.subject and info.session_id identifiers")
    return str(subject), str(session_id)


def source_session_id(session):
    """Return the physical recording ID; all its reset sequences stay in one CV fold."""
    value = getattr(session.info, "source_session_id", session.info.session_id)
    if value is None or pd.isna(value):
        raise ValueError("Physical source-session ID cannot be missing")
    return str(value)


def score_sessions(sessions, model, param_values):
    """Score one frozen parameter vector; accuracy uses left action on an exact tie."""
    sessions = _validate_sessions(sessions)
    identity = scored_data_identity(sessions)
    log_likelihood = 0.0
    n_correct = 0
    for session in sessions:
        choices, _, scored = get_trial_data(session)
        forward = model.forward(session, param_values)
        if not np.array_equal(forward["scored"], scored):
            raise ValueError("Model forward mask disagrees with the declared trial mask")
        log_likelihood += float(forward["log_likelihood"])
        probabilities = np.asarray(forward["choice_probs"], dtype=float)
        if probabilities.shape != (2, len(choices)) or not np.isfinite(probabilities).all():
            raise ValueError("Model probabilities must be finite with shape (2, n_trials)")
        predicted = probabilities[1] > 0.5
        n_correct += int((predicted[scored] == choices[scored]).sum())
    n_scored = identity["n_free_choice_trials"]
    return dict(
        **identity,
        log_likelihood=log_likelihood,
        nll=-log_likelihood,
        nll_per_trial=-log_likelihood / n_scored if n_scored else None,
        n_correct=n_correct,
        accuracy=n_correct / n_scored if n_scored else None,
        accuracy_tie_policy="left_on_tie",
    )


def fit_session(session, model, repeats=10, MAP_fit=False, **kwargs):
    """Fit one reset sequence; see fit_sessions for optimizer and diagnostic options."""
    fit = fit_sessions([session], model, repeats=repeats, MAP_fit=MAP_fit, **kwargs)
    fit["session_id"] = session_identity(session)[1]
    return fit


def fit_sessions(
    sessions,
    model,
    repeats=10,
    MAP_fit=False,
    compute_CVLL=False,
    *,
    seed=0,
    initial_params=None,
    optimizer_options=None,
    verbose=False,
):
    """Fit shared parameters to one subject's sessions using multi-start L-BFGS-B.

    ``repeats`` counts random prior starts; supplied ``initial_params`` vectors
    add feasible starts. Random starts are clipped to custom bounds, matching
    SciPy's bound handling. Supplied starts outside bounds are rejected.
    The lowest finite objective wins, as in the original fitter; unsuccessful
    convergence is warned and recorded, never silently relabelled successful.
    Information criteria use unpenalized likelihood, scored targets and the
    number of parameters not fixed by equal lower/upper bounds, including for MAP.
    """
    sessions = _validate_sessions(sessions)
    _validate_model(model)
    if not isinstance(repeats, (int, np.integer)) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    identity = scored_data_identity(sessions)
    n_scored = identity["n_free_choice_trials"]
    if n_scored < 1:
        raise ValueError("No eligible free-choice targets to fit")
    bounds = get_bounds(model)
    starts = _get_supplied_starts(initial_params, model.n_params, bounds)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        start = get_init_param_values(model.param_ranges, rng=rng)
        starts.append(_clip_to_bounds(start, bounds))

    def objective(params):
        if not np.isfinite(params).all():
            return np.inf
        value = -multi_session_log_likelihood(sessions, model, params)
        if MAP_fit:
            value -= log_prior_prob(params, model.param_ranges)
        return float(value) if np.isfinite(value) else np.inf

    attempts = []
    results = []
    options = {} if optimizer_options is None else dict(optimizer_options)
    for repeat, start in enumerate(starts):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds, options=options)
        results.append(fitted)
        attempts.append(
            dict(
                repeat=repeat,
                initial=start.tolist(),
                param_values=np.asarray(fitted.x).tolist(),
                objective=float(fitted.fun),
                success=bool(fitted.success),
                status=int(getattr(fitted, "status", 0 if fitted.success else 1)),
                message=str(fitted.message),
                n_iterations=int(getattr(fitted, "nit", 0)),
                n_evaluations=int(getattr(fitted, "nfev", 0)),
                warnings=sorted({f"{item.category.__name__}: {item.message}" for item in caught}),
            )
        )
        if verbose:
            print(f"{model.name}: start {repeat + 1}/{len(starts)}, objective={fitted.fun:.6g}")
    finite = [i for i, result in enumerate(results) if np.isfinite(result.fun) and np.isfinite(result.x).all()]
    if not finite:
        raise RuntimeError(f"All {len(starts)} optimization attempts were nonfinite for {model.name}")
    best_index = min(finite, key=lambda i: results[i].fun)
    best = results[best_index]
    if not best.success:
        warnings.warn(f"Best {model.name} fit did not converge: {best.message}", RuntimeWarning, stacklevel=2)
    scores = score_sessions(sessions, model, best.x)
    ll = scores["log_likelihood"]
    if not np.isfinite(ll):
        raise RuntimeError(f"Best {model.name} fit has a nonfinite likelihood")
    n_free_params = sum(lower is None or upper is None or lower != upper for lower, upper in bounds)
    aic = -2 * ll + 2 * n_free_params
    fit = dict(
        model=model.name,
        method="MAP" if MAP_fit else "ML",
        param_names=list(model.param_names),
        param_ranges=list(model.param_ranges),
        n_params=model.n_params,
        n_optimized_params=n_free_params,
        n_ic_params=n_free_params,
        param_values=np.asarray(best.x).tolist(),
        bounds=[list(bound) for bound in bounds],
        subject=session_identity(sessions[0])[0],
        session_ids=[session_identity(session)[1] for session in sessions],
        n_sessions=len(sessions),
        n_source_sessions=len({source_session_id(session) for session in sessions}),
        n_trials=sum(len(get_trial_data(session)[0]) for session in sessions),
        **scores,
        AIC=aic,
        AICc=(
            aic + 2 * n_free_params * (n_free_params + 1) / (n_scored - n_free_params - 1)
            if n_scored > n_free_params + 1
            else float("inf")
        ),
        BIC=-2 * ll + np.log(n_scored) * n_free_params,
        log_posterior_prob=ll + log_prior_prob(best.x, model.param_ranges) if MAP_fit else None,
        objective=float(best.fun),
        seed=None if seed is None else int(seed),
        best_repeat=best_index,
        best_success=bool(best.success),
        successful_fits=sum(bool(result.success) for result in results),
        total_attempts=len(starts),
        random_starts=int(repeats),
        attempts=attempts,
        CVLL=None,
        CV_nll_per_trial=None,
        CV_accuracy=None,
        cv_n_free_choice_trials=None,
        cv_scored_targets_hash=None,
        cv_folds=None,
    )
    if compute_CVLL:
        if __package__:
            from .model_comparison_helper_functions import cross_validated_scores
        else:
            from model_comparison_helper_functions import cross_validated_scores
        cv = cross_validated_scores(
            sessions,
            model,
            MAP_fit=MAP_fit,
            repeats=repeats,
            seed=seed,
            optimizer_options=optimizer_options,
            verbose=verbose,
        )
        fit.update(
            CVLL=cv["CVLL"],
            CV_nll_per_trial=cv["CV_nll_per_trial"],
            CV_accuracy=cv["CV_accuracy"],
            cv_n_free_choice_trials=cv["n_free_choice_trials"],
            cv_scored_targets_hash=cv["scored_targets_hash"],
            cv_folds=cv["folds"],
        )
    return fit


def fit_subjects(sessions, model, MAP_fit=False, compute_CVLL=False, plot=False, *, repeats=10, seed=0, **kwargs):
    """Return one row per subject/model/method, retaining nested optimizer diagnostics."""
    if plot:
        raise ValueError("Plotting is a separate stage; pass the returned fit table to the plotting helpers")
    sessions = tuple(sessions)
    if not sessions:
        raise ValueError("No sessions supplied")
    subjects = sorted({session_identity(session)[0] for session in sessions})
    subject_fits = []
    for index, subject in enumerate(subjects):
        selected = [session for session in sessions if session_identity(session)[0] == subject]
        fit = fit_sessions(
            selected,
            model,
            repeats=repeats,
            MAP_fit=MAP_fit,
            compute_CVLL=compute_CVLL,
            seed=None if seed is None else int(seed) + index,
            **kwargs,
        )
        fit.update({f"params.{name}": value for name, value in zip(fit["param_names"], fit["param_values"])})
        subject_fits.append(fit)
    fits_df = pd.DataFrame(subject_fits)
    # Give optional scalar scores a stable numeric dtype before ML/MAP tables
    # are concatenated. ML posterior scores are deliberately undefined (NaN).
    for name in ("log_posterior_prob", "CVLL", "CV_nll_per_trial", "CV_accuracy"):
        fits_df[name] = pd.to_numeric(fits_df[name], errors="raise").astype(float)
    return fits_df


def _validate_ranges(param_ranges):
    if any(kind not in _PRIORS for kind in param_ranges):
        raise ValueError("Parameter ranges must be 'unit', 'pos' or 'unc'")


def _validate_model(model):
    if len(model.param_names) != model.n_params or len(model.param_ranges) != model.n_params:
        raise ValueError("Model parameter metadata have inconsistent lengths")
    if len(set(model.param_names)) != model.n_params:
        raise ValueError("Model parameter names must be unique")
    _validate_ranges(model.param_ranges)


def _validate_sessions(sessions):
    sessions = tuple(sessions)
    if not sessions:
        raise ValueError("At least one reset sequence is required")
    identities = [session_identity(session) for session in sessions]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate subject/session IDs; each reset sequence needs a unique identity")
    if len({subject for subject, _ in identities}) != 1:
        raise ValueError("Fit one subject at a time; use fit_subjects for multiple subjects")
    return sessions


def _clip_to_bounds(params, bounds):
    lower = [-np.inf if bound[0] is None else bound[0] for bound in bounds]
    upper = [np.inf if bound[1] is None else bound[1] for bound in bounds]
    return np.clip(params, lower, upper)


def _get_supplied_starts(initial_params, n_params, bounds):
    if initial_params is None:
        return []
    array = np.asarray(initial_params, dtype=float)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != n_params or not np.isfinite(array).all():
        raise ValueError("initial_params must contain finite vectors with one entry per parameter")
    if not np.array_equal(array, np.array([_clip_to_bounds(row, bounds) for row in array])):
        raise ValueError("A supplied initial parameter vector lies outside model bounds")
    return [row.copy() for row in array]
