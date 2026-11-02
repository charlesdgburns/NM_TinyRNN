"""Explicit behavioral sequences and portable CSV input for cognitive-model fitting.

One CSV row is one physical trial. A Session is one independently reset sequence;
several sequences may belong to the same physical recording session. Choice is
0=left and 1=right. Current input updates states used to predict the NEXT choice.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

if __package__:
    from .model_implementation_helper_functions import get_trial_data
else:
    from model_implementation_helper_functions import get_trial_data


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
