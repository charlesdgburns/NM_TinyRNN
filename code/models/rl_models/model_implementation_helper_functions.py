"""Trial validation and the original binary-choice likelihood conventions.

Choice 0 is left (poke_4); choice 1 is right (poke_6). Every input, including
forced and unscored trials, updates model state. Masks affect scoring only.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


# Explicit reset sequences and portable CSV loading.

class Session:
    """One reset sequence with explicit subject, recording and trial identities.

    Arrays are one-dimensional, equally long and nonempty. Forced inputs still
    update model state; only free trials satisfying ll_mask enter the likelihood.
    source_session_id identifies the recording held out in session-level CV.
    session_id identifies a reset sequence and must be unique within a subject.
    """

    def __init__(
        self,
        choices,
        outcomes,
        forced_choice=None,
        *,
        subject="synthetic",
        session_id="session_1",
        source_session_id=None,
        sequence_id=None,
        trial_ids=None,
        ll_mask=None,
        metadata=None,
    ):
        self.choices = np.asarray(choices).copy()
        self.outcomes = np.asarray(outcomes).copy()
        forced = np.zeros(self.choices.shape, dtype=bool) if forced_choice is None else np.asarray(forced_choice)
        if forced.shape != self.choices.shape or not np.isin(forced, [0, 1]).all():
            raise ValueError("forced_choice must be a binary array matching choices")
        self.free_choice = ~forced.astype(bool)
        self.ll_mask = np.ones(self.choices.shape, dtype=bool) if ll_mask is None else np.asarray(ll_mask).copy()
        self.choices, self.outcomes, scored = get_trial_data(self)
        self.n_trials = len(self.choices)
        self.n_free_choice_trials = int(scored.sum())
        self.n_physical_free_choice_trials = int(self.free_choice.sum())
        self.trial_ids = np.arange(self.n_trials) if trial_ids is None else np.asarray(trial_ids).copy()
        if self.trial_ids.shape != (self.n_trials,) or pd.isna(self.trial_ids).any():
            raise ValueError("trial_ids must provide one nonmissing identity per trial")
        if pd.Index(self.trial_ids).has_duplicates:
            raise ValueError("trial_ids must be unique within a reset sequence")
        subject = _identity(subject, "subject")
        session_id = _identity(session_id, "session_id")
        source_session_id = (
            session_id if source_session_id is None else _identity(source_session_id, "source_session_id")
        )
        if sequence_id is not None:
            sequence_id = _identity(sequence_id, "sequence_id")
        metadata = {} if metadata is None else dict(metadata)
        reserved = {"subject", "session_id", "source_session_id", "sequence_id"}
        if reserved.intersection(metadata):
            raise ValueError("metadata cannot override subject/session/sequence identities")
        self.info = SimpleNamespace(
            subject=subject,
            session_id=session_id,
            source_session_id=source_session_id,
            sequence_id=sequence_id,
            **metadata,
        )
        self.ll_mask = self.ll_mask.astype(bool)
        self.trials_df = pd.DataFrame(
            {"choice": self.choices, "outcome": self.outcomes, "forced_choice": ~self.free_choice},
            index=pd.Index(self.trial_ids, name="trial"),
        )
        for array in (self.choices, self.outcomes, self.free_choice, self.ll_mask, self.trial_ids):
            array.setflags(write=False)


def _identity(value, name):
    """Validate an identifier before string conversion could disguise missing data."""
    if not pd.api.types.is_scalar(value) or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{name} must be a nonmissing, nonempty scalar identifier")
    return str(value)


def _boolean_values(values, name):
    """Accept CSV 0/1 or true/false while rejecting missing or ambiguous values."""
    normalised = values.map(lambda value: value.lower() if isinstance(value, str) else value)
    converted = normalised.map({True: True, False: False, 1: True, 0: False, "true": True, "false": False})
    if converted.isna().any():
        raise ValueError(f"{name} must contain only 0/1 or true/false, without missing values")
    return converted.to_numpy(dtype=bool)


def sessions_from_dataframe(trials_df, *, score_column=None):
    """Build reset sequences without reordering or dropping source trials.

    Required flat columns: subject, session_id, trial, choice, outcome,
    forced_choice. Optional sequence_id splits a physical recording into
    independently reset sequences; LOSO still holds out the whole recording.
    Optional score_column selects likelihood targets, never input histories.
    Reward-probability metadata must be constant within a physical session.
    """
    required = {"subject", "session_id", "trial", "choice", "outcome", "forced_choice"}
    missing = required.difference(trials_df.columns)
    if missing:
        raise ValueError(f"Missing trial columns: {sorted(missing)}")
    if trials_df.empty or trials_df[list(required)].isna().any().any():
        raise ValueError("Input trials must be nonempty and required fields nonmissing")
    if score_column is not None and score_column not in trials_df:
        raise ValueError(f"Missing likelihood-mask column {score_column!r}")
    trials_df = trials_df.copy()
    trials_df["subject"] = trials_df["subject"].astype(str)
    trials_df["session_id"] = trials_df["session_id"].astype(str)
    if trials_df.duplicated(["subject", "session_id", "trial"]).any():
        raise ValueError("Duplicate physical subject/session/trial identity")
    if "sequence_id" in trials_df and trials_df["sequence_id"].isna().any():
        raise ValueError("sequence_id cannot be missing")
    sessions = []
    for (subject, source_id), recording_df in trials_df.groupby(["subject", "session_id"], sort=False):
        trial_numbers = pd.to_numeric(recording_df["trial"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(trial_numbers).all() or not np.all(np.diff(trial_numbers) > 0):
            raise ValueError(f"Trials must be strictly chronological within {subject}/{source_id}")
        metadata = {}
        for name in ("reward_prob_high", "reward_prob_low"):
            if name in recording_df:
                if recording_df[name].isna().any() or recording_df[name].nunique() != 1:
                    raise ValueError(f"{name} must be constant/nonmissing within a recording")
                metadata[name] = float(recording_df[name].iloc[0])
        if len(metadata) == 1:
            raise ValueError("Provide both reward_prob_high and reward_prob_low, or neither")
        if "sequence_id" in recording_df:
            sequence_labels = recording_df["sequence_id"].astype(str)
            runs = sequence_labels.loc[sequence_labels.ne(sequence_labels.shift())]
            if runs.duplicated().any():
                raise ValueError("A reset sequence must occupy one contiguous run within a recording")
            groups = recording_df.groupby("sequence_id", sort=False)
        else:
            groups = [(None, recording_df)]
        for sequence_id, sequence_df in groups:
            reset_id = source_id if sequence_id is None else json.dumps([source_id, str(sequence_id)])
            choices = sequence_df["choice"]
            if choices.isin(["poke_4", "poke_6"]).all():
                choices = choices.map({"poke_4": 0, "poke_6": 1})
            sessions.append(
                Session(
                    choices.to_numpy(),
                    sequence_df["outcome"].to_numpy(),
                    _boolean_values(sequence_df["forced_choice"], "forced_choice"),
                    subject=subject,
                    session_id=reset_id,
                    source_session_id=source_id,
                    sequence_id=sequence_id,
                    trial_ids=sequence_df["trial"].to_numpy(),
                    ll_mask=None if score_column is None else _boolean_values(sequence_df[score_column], score_column),
                    metadata=metadata,
                )
            )
    return sessions


def load_sessions(csv_path, *, score_column=None):
    """Read the documented flat CSV schema; no private project loader is needed."""
    csv_path = Path(csv_path)
    trials_df = pd.read_csv(csv_path, dtype={"subject": str, "session_id": str, "sequence_id": str})
    return sessions_from_dataframe(trials_df, score_column=score_column)
