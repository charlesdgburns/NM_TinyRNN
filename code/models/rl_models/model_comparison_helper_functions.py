"""Leave-recording-session-out scoring and paired subject-level model comparisons.

No renderer is imported here. Returned tables are the complete scientific input
for optional plotting; one subject, rather than a trial or CV fold, is the unit.
"""

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .model_fitting_helper_functions import (
        _validate_sessions,
        fit_sessions,
        score_sessions,
        scored_data_identity,
        session_identity,
        source_session_id,
    )
else:
    from model_fitting_helper_functions import (
        _validate_sessions,
        fit_sessions,
        score_sessions,
        scored_data_identity,
        session_identity,
        source_session_id,
    )


_METRICS = {
    # (higher_is_better, already_per_trial, cross_validated)
    "AIC": (False, False, False),
    "AICc": (False, False, False),
    "BIC": (False, False, False),
    "CVLL": (True, False, True),
    "log_likelihood": (True, False, False),
    "nll": (False, False, False),
    "nll_per_trial": (False, True, False),
    "accuracy": (True, True, False),
    "CV_nll_per_trial": (False, True, True),
    "CV_accuracy": (True, True, True),
}


def cross_validated_log_likelihood(sessions, model, MAP_fit=False, **kwargs):
    """Return LOSO log likelihood; use cross_validated_scores for fold diagnostics."""
    return cross_validated_scores(sessions, model, MAP_fit=MAP_fit, **kwargs)["CVLL"]


def cross_validated_scores(
    sessions,
    model,
    MAP_fit=False,
    *,
    repeats=10,
    seed=0,
    optimizer_options=None,
    verbose=False,
):
    """Fit each fold without its entire physical recording, then pool held-out errors.

    Each supplied sequence resets model state. Multiple blocks with the same
    ``info.source_session_id`` are kept together: none of a test recording's
    blocks enters its training fit. The requested ML/MAP method, restart count
    and optimizer settings apply to every fold. No full-data fitted parameters
    are reused as starts in cross-validation.
    """
    sessions = _validate_sessions(sessions)
    source_ids = sorted({source_session_id(session) for session in sessions})
    if len(source_ids) < 2:
        raise ValueError("LOSO requires at least two physical source sessions, not merely two reset blocks")
    folds = []
    for fold, held_out in enumerate(source_ids):
        train_sessions = [session for session in sessions if source_session_id(session) != held_out]
        test_sessions = [session for session in sessions if source_session_id(session) == held_out]
        fitted = fit_sessions(
            train_sessions,
            model,
            repeats=repeats,
            MAP_fit=MAP_fit,
            seed=None if seed is None else int(seed) + fold + 1,
            optimizer_options=optimizer_options,
            verbose=verbose,
        )
        scores = score_sessions(test_sessions, model, fitted["param_values"])
        folds.append(
            dict(
                fold=fold,
                held_out_session=held_out,
                training_source_sessions=[value for value in source_ids if value != held_out],
                held_out_reset_sequences=[session_identity(session)[1] for session in test_sessions],
                n_train_sessions=len(train_sessions),
                n_test_sequences=len(test_sessions),
                n_test_free=scores["n_free_choice_trials"],
                log_likelihood=scores["log_likelihood"],
                n_correct=scores["n_correct"],
                scored_targets_hash=scores["scored_targets_hash"],
                fit=fitted,
            )
        )
    identity = scored_data_identity(sessions)
    n_scored = identity["n_free_choice_trials"]
    if n_scored < 1:
        raise ValueError("No held-out targets are eligible for scoring")
    cvll = float(sum(fold["log_likelihood"] for fold in folds))
    n_correct = int(sum(fold["n_correct"] for fold in folds))
    return dict(
        model=model.name,
        method="MAP" if MAP_fit else "ML",
        **identity,
        CVLL=cvll,
        CV_nll_per_trial=-cvll / n_scored,
        CV_accuracy=n_correct / n_scored,
        accuracy_tie_policy="left_on_tie",
        n_folds=len(folds),
        folds=folds,
    )


def model_comparison(
    model_fits,
    metric="BIC",
    parametric=True,
    normalise=True,
    print_subject_values=False,
):
    """Return paired subject values and mean/SEM rankings, separately for ML/MAP.

    Parameters
    ----------
    model_fits : DataFrame or sequence of DataFrames
        One row per subject/model/method, obtained from ``fit_subjects``. Required
        hashes/counts verify identical inputs, reset sequences and scored trials.
    metric : str
        AIC, AICc, BIC, CVLL, log_likelihood, nll, nll_per_trial,
        accuracy, CV_nll_per_trial or CV_accuracy. Lower is better except
        likelihoods and accuracy. Already-normalized metrics are not divided twice.
    parametric : bool
        Paired one-sided t-test if True; paired Wilcoxon test otherwise.
    normalise : bool
        Divide extensive metrics by their scored-target counts before averaging
        subjects. Rates such as accuracy always retain their existing units.

    Returns
    -------
    comparison : dict
        ``summary_df`` has one row per method/model. ``subject_values_df`` has one
        row per subject/model/method. SEM uses subjects. P values compare each
        alternative with the best model selected in this same table, so they are
        descriptive, post-selection and uncorrected for multiple comparisons.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {tuple(_METRICS)}")
    if isinstance(model_fits, pd.DataFrame):
        fits_df = model_fits.copy()
    else:
        tables = list(model_fits)
        if not tables or any(not isinstance(table, pd.DataFrame) for table in tables):
            raise ValueError("model_fits must contain at least one fit DataFrame")
        fits_df = pd.concat(tables, ignore_index=True)
    if fits_df.empty:
        raise ValueError("No subject fits supplied")
    higher_is_better, is_rate, is_cv = _METRICS[metric]
    count_column = "cv_n_free_choice_trials" if is_cv else "n_free_choice_trials"
    hash_column = "cv_scored_targets_hash" if is_cv else "scored_targets_hash"
    required = {"subject", "model", "method", metric, count_column, hash_column}
    missing = required - set(fits_df.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    if fits_df[list(required)].isna().any().any():
        raise ValueError(f"Missing values in {metric} comparison; compute requested scores for every model")
    if fits_df.duplicated(["subject", "model", "method"]).any():
        raise ValueError("Expected one row per subject/model/method; pool folds before model comparison")
    summary_rows = []
    subject_rows = []
    for method, method_df in fits_df.groupby("method", sort=True):
        if method not in ("ML", "MAP"):
            raise ValueError(f"Unknown fit method {method!r}; expected ML or MAP")
        _check_matched_subjects(method_df, count_column, hash_column)
        numeric = method_df[[metric, count_column]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric[:, 1] <= 0).any():
            raise ValueError(f"Nonfinite {metric} or nonpositive target count; AICc needs n > k + 1")
        if not np.equal(numeric[:, 1], np.floor(numeric[:, 1])).all():
            raise ValueError("Scored-target counts must be positive integers")
        selected_df = method_df[["subject", "model", "method", metric, count_column]].copy()
        selected_df["metric"] = metric
        selected_df["raw_value"] = selected_df[metric].astype(float)
        selected_df["n_scored"] = selected_df[count_column].astype(int)
        selected_df["value"] = selected_df["raw_value"]
        if normalise and not is_rate:
            selected_df["value"] /= selected_df["n_scored"]
        selected_df["per_trial"] = bool(is_rate or normalise)
        subject_rows.append(
            selected_df[["subject", "model", "method", "metric", "raw_value", "value", "n_scored", "per_trial"]]
        )
        values_df = selected_df.pivot(index="subject", columns="model", values="value")
        means = values_df.mean().sort_values(ascending=not higher_is_better, kind="stable")
        best_model = means.index[0]
        for rank, model_name in enumerate(means.index, start=1):
            model_values = values_df[model_name].to_numpy()
            p_value = np.nan
            if model_name != best_model:
                p_value = _paired_p_value(model_values, values_df[best_model].to_numpy(), higher_is_better, parametric)
            summary_rows.append(
                dict(
                    method=method,
                    metric=metric,
                    model=model_name,
                    rank=rank,
                    mean=float(np.mean(model_values)),
                    sem=float(stats.sem(model_values)) if len(model_values) > 1 else np.nan,
                    n_subjects=len(model_values),
                    per_trial=bool(is_rate or normalise),
                    higher_is_better=higher_is_better,
                    reference_model=best_model,
                    p_value=p_value,
                    test="one-sided paired t" if parametric else "one-sided paired Wilcoxon",
                    inference="descriptive; reference selected on these data; multiple comparisons uncorrected",
                )
            )
    subject_values_df = pd.concat(subject_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    if print_subject_values:
        print(subject_values_df.to_string(index=False))
    return dict(summary_df=summary_df, subject_values_df=subject_values_df)


def _check_matched_subjects(method_df, count_column, hash_column):
    subject_sets = [set(group["subject"]) for _, group in method_df.groupby("model")]
    if any(subjects != subject_sets[0] for subjects in subject_sets[1:]):
        raise ValueError("Models do not cover identical subjects; no subjects are silently dropped")
    for subject, group in method_df.groupby("subject", sort=False):
        if group[count_column].nunique() != 1 or group[hash_column].nunique() != 1:
            raise ValueError(
                f"Models for subject {subject} did not use identical scored targets/history/reset identities"
            )


def _paired_p_value(alternative_values, reference_values, higher_is_better, parametric):
    """Return descriptive one-sided probability, with an explicit identical-pairs case."""
    if len(alternative_values) < 2:
        return np.nan
    differences = alternative_values - reference_values
    if np.all(differences == 0):
        return 1.0
    side = "less" if higher_is_better else "greater"
    if parametric:
        return float(stats.ttest_rel(alternative_values, reference_values, alternative=side).pvalue)
    return float(stats.wilcoxon(differences, alternative=side).pvalue)
