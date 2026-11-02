"""Portable CSV/reset identities and an end-to-end two-model CLI comparison."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from rl_models.model_implementation_helper_functions import get_trial_data
from rl_models.model_implementation_model_classes import Q_model
from rl_models.run_comparison import _split_groups, main
from rl_models.sessions import Session, load_sessions, sessions_from_dataframe


def trial_table(*, n_trials=24, blocks=False):
    """One row per physical trial in two mice and two recordings per mouse."""
    rng = np.random.default_rng(291026)
    rows = []
    for subject in ["M01", "M02"]:
        for session_id in ["day01", "day02"]:
            for trial in range(n_trials):
                rows.append(dict(
                    subject=subject,
                    session_id=session_id,
                    trial=trial + 1,
                    choice=int(rng.integers(0, 2)),
                    outcome=int(rng.integers(0, 2)),
                    forced_choice=trial % 7 == 0,
                    eligible=trial != 1,
                ))
                if blocks:
                    rows[-1]["sequence_id"] = "first" if trial < n_trials // 2 else "second"
    return pd.DataFrame(rows)


class SessionContractTests(unittest.TestCase):
    def test_session_copies_inputs_and_declares_all_free_default(self):
        choices = np.array([0, 1, 1])
        outcomes = np.array([1, 0, 1])
        session = Session(choices, outcomes, subject="M", session_id="day")
        choices[0], outcomes[0] = 1, 0
        np.testing.assert_array_equal(session.choices, [0, 1, 1])
        np.testing.assert_array_equal(session.outcomes, [1, 0, 1])
        self.assertTrue(session.free_choice.all())
        self.assertEqual(session.info.source_session_id, "day")
        for array in [session.choices, session.outcomes, session.free_choice, session.ll_mask, session.trial_ids]:
            self.assertFalse(array.flags.writeable)

    def test_session_rejects_missing_or_ambiguous_identity(self):
        for field in ["subject", "session_id"]:
            for missing in [None, np.nan, ""]:
                with self.subTest(field=field, missing=missing), self.assertRaises(ValueError):
                    Session([0, 1], [1, 0], **{field: missing})
        for missing in [np.nan, ""]:
            with self.subTest(source_id=missing), self.assertRaises(ValueError):
                Session([0, 1], [1, 0], source_session_id=missing)
        with self.assertRaises(ValueError):
            Session([0, 1], [1, 0], trial_ids=[3, 3])
        with self.assertRaises(ValueError):
            Session([0, 1], [1, 0], metadata={"subject": "other"})

    def test_csv_roundtrip_preserves_trials_masks_and_recording_block_identities(self):
        trials_df = trial_table(blocks=True)
        original_df = trials_df.copy(deep=True)
        trials_df["choice"] = trials_df.choice.map({0: "poke_4", 1: "poke_6"})
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "trials.csv"
            trials_df.to_csv(csv_path, index=False)
            sessions = load_sessions(csv_path, score_column="eligible")
        self.assertEqual(len(sessions), 8)
        self.assertEqual(sum(s.n_trials for s in sessions), len(trials_df))
        expected_count = int((~trials_df.forced_choice & trials_df.eligible).sum())
        self.assertEqual(sum(s.n_free_choice_trials for s in sessions), expected_count)
        for session in sessions:
            selected_df = original_df.loc[
                original_df.subject.eq(session.info.subject)
                & original_df.session_id.eq(session.info.source_session_id)
                & original_df.sequence_id.eq(session.info.sequence_id)
            ]
            np.testing.assert_array_equal(session.trial_ids, selected_df.trial)
            np.testing.assert_array_equal(session.choices, selected_df.choice)
            np.testing.assert_array_equal(session.outcomes, selected_df.outcome)
            np.testing.assert_array_equal(session.free_choice, ~selected_df.forced_choice)
            np.testing.assert_array_equal(session.ll_mask, selected_df.eligible)

    def test_csv_rejects_missing_columns_duplicate_targets_and_reordered_trials(self):
        original = trial_table()
        for name in ["subject", "session_id", "trial", "choice", "outcome", "forced_choice"]:
            with self.subTest(missing_column=name), self.assertRaises(ValueError):
                sessions_from_dataframe(original.drop(columns=name))
        duplicated = pd.concat([original, original.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            sessions_from_dataframe(duplicated)
        reordered = original.copy()
        reordered.iloc[[0, 1]] = reordered.iloc[[1, 0]].to_numpy()
        with self.assertRaises(ValueError):
            sessions_from_dataframe(reordered)
        for name in ["subject", "session_id", "trial"]:
            missing = original.copy()
            missing.loc[0, name] = None
            with self.subTest(missing_value=name), self.assertRaises(ValueError):
                sessions_from_dataframe(missing)

    def test_csv_rejects_noncontiguous_reset_sequence_and_inconsistent_environment(self):
        trials_df = trial_table(blocks=True)
        trials_df.loc[2, "sequence_id"] = "second"
        with self.assertRaises(ValueError):
            sessions_from_dataframe(trials_df)
        trials_df = trial_table()
        trials_df["reward_prob_high"], trials_df["reward_prob_low"] = .75, .25
        trials_df.loc[0, "reward_prob_high"] = .9
        with self.assertRaises(ValueError):
            sessions_from_dataframe(trials_df)
        with self.assertRaises(ValueError):
            sessions_from_dataframe(trial_table(), score_column="missing")

    def test_invalid_model_bounds_and_contradictory_masks_are_rejected(self):
        for bound in [None, np.array(.2), (0., np.nan)]:
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                Q_model(param_bounds={"alpha": bound})
        conflicting = SimpleNamespace(
            choices=np.array([0, 1]), outcomes=np.array([1, 0]),
            free_choice=np.array([True, True]), forced_choice=np.array([False, True]),
        )
        with self.assertRaises(ValueError):
            get_trial_data(conflicting)


class SplitAndCliTests(unittest.TestCase):
    def setUp(self):
        self.sessions = sessions_from_dataframe(trial_table(blocks=True), score_column="eligible")
        self.plan = {"splits": [
            {"name": "held_out_day", "subject": subject, "train": ["day01"], "eval": ["day02"]}
            for subject in ["M01", "M02"]
        ]}

    def test_split_plan_holds_out_whole_recordings_and_preserves_reset_units(self):
        groups = _split_groups(self.sessions, self.plan)
        self.assertEqual(len(groups), 1)
        name, training, held_out = groups[0]
        self.assertEqual(name, "held_out_day")
        self.assertEqual(len(training), 4)
        self.assertEqual(len(held_out["eval"]), 4)
        self.assertEqual({s.info.source_session_id for s in training}, {"day01"})
        self.assertEqual({s.info.source_session_id for s in held_out["eval"]}, {"day02"})
        self.assertEqual({s.info.subject for s in training}, {"M01", "M02"})

    def test_split_plan_rejects_overlap_unknown_ids_missing_subjects_and_required_keys(self):
        plans = []
        overlap = copy.deepcopy(self.plan)
        overlap["splits"][0]["eval"] = ["day01"]
        plans.append(overlap)
        unknown = copy.deepcopy(self.plan)
        unknown["splits"][0]["eval"] = ["missing"]
        plans.append(unknown)
        plans.append({"splits": self.plan["splits"][:1]})
        missing = copy.deepcopy(self.plan)
        del missing["splits"][0]["subject"]
        plans.append(missing)
        for plan in plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                _split_groups(self.sessions, plan)

    def test_cli_writes_strict_json_csv_metrics_and_refuses_output_overwrite(self):
        trials_df = trial_table(n_trials=20)
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "trials.csv"
            output_dir = directory / "comparison"
            trials_df.to_csv(csv_path, index=False)
            input_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            arguments = [
                "--input", str(csv_path), "--output", str(output_dir),
                "--models", "Q_basic_bias", "Reward_EMA_prsvEMA",
                "--methods", "ML", "MAP", "--repeats", "1", "--seed", "291026",
                "--score-column", "eligible", "--cv",
            ]
            with redirect_stdout(io.StringIO()):
                main(arguments)
            expected_files = {"RUN.json", "fits.json", "subject_fits.csv", "comparison_summary.csv"}
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))
            manifest = json.loads((output_dir / "RUN.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertFalse(any("DataFrame concatenation" in value for value in manifest["warnings"]))
            self.assertEqual(manifest["n_fit_rows"], 8)
            self.assertEqual(manifest["input_sha256"], input_hash)
            fits = json.loads((output_dir / "fits.json").read_text(), parse_constant=self._reject_json_constant)
            self.assertEqual(len(fits), 8)
            self.assertEqual({fit["method"] for fit in fits}, {"ML", "MAP"})
            self.assertTrue(all(len(fit["cv_folds"]) == 2 for fit in fits))
            summary_df = pd.read_csv(output_dir / "comparison_summary.csv")
            metrics = {"AIC", "BIC", "log_likelihood", "nll", "nll_per_trial", "accuracy"}
            metrics |= {"CVLL", "CV_nll_per_trial", "CV_accuracy"}
            self.assertTrue(metrics.issubset(set(summary_df.metric)))
            self.assertEqual(set(summary_df.n_subjects), {2})
            self.assertFalse(
                summary_df.duplicated(["protocol", "split", "method", "model", "metric", "normalise"]).any()
            )
            self.assertEqual(len(pd.read_csv(output_dir / "subject_fits.csv")), 8)
            before = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                main(arguments)
            self.assertEqual(before, {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()})
            self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(), input_hash)

    @staticmethod
    def _reject_json_constant(value):
        raise AssertionError(f"Nonstandard JSON numeric constant: {value}")


if __name__ == "__main__":
    unittest.main()
