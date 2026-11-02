"""Public registry and frozen held-out scoring through the saved-output workflow."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rl_models import make_model
from rl_models.example import get_example_trials_df
from rl_models.model_fitting_helper_functions import score_sessions
from rl_models.run_comparison import run_comparison
from rl_models.sessions import load_sessions


class PublicWorkflowTests(unittest.TestCase):
    def test_registry_validates_custom_bounds_at_construction(self):
        with self.assertRaises(ValueError):
            make_model("Q_basic_bias", param_bounds={"unknown_parameter": (0, 1)})
        with self.assertRaises(ValueError):
            make_model("Q_basic_bias", param_bounds={"alpha": (-1, 1)})
        model = make_model("Q_basic_bias", param_bounds={"alpha": (.2, .8)})
        self.assertEqual(model.param_bounds["alpha"], (.2, .8))

    def test_explicit_held_out_run_uses_frozen_parameters_and_reports_unused_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "trials.csv"
            trials_df = get_example_trials_df(n_subjects=2, n_sessions=3, n_trials=20)
            trials_df.to_csv(input_path, index=False)
            plan = {"splits": [dict(
                name="held_out", subject=subject, train=["session_1"], eval=["session_2"]
            ) for subject in trials_df.subject.unique()]}
            result = run_comparison(
                input_path, root / "fits", models=["Q_basic_bias"], methods=["ML", "MAP"],
                repeats=1, split_plan=plan, test="both",
            )
            manifest = result["manifest"]
            self.assertTrue(manifest["complete"])
            protocol = manifest["data_use"]["protocols"][0]
            self.assertEqual(protocol["unused_input_trials"], 40)
            self.assertEqual(len(protocol["unused_source_sessions"]), 2)
            sessions = load_sessions(input_path)
            fits = json.loads((root / "fits" / "fits.json").read_text())
            saved = pd.read_csv(root / "fits" / "held_out_scores.csv")
            for fitted in fits:
                test_sessions = [session for session in sessions
                                 if session.info.subject == fitted["subject"]
                                 and session.info.source_session_id == "session_2"]
                expected = score_sessions(test_sessions, make_model(fitted["model"]), fitted["param_values"])
                row = saved.loc[saved.subject.eq(fitted["subject"]) & saved.method.eq(fitted["method"])].iloc[0]
                self.assertAlmostEqual(row.log_likelihood, expected["log_likelihood"])
                self.assertEqual(row.scored_targets_hash, expected["scored_targets_hash"])
                self.assertEqual(fitted["n_source_sessions"], 1)
            comparisons = result["summary_df"]
            self.assertEqual(set(comparisons.split), {"train", "eval"})
            self.assertEqual(set(comparisons.test), {"one-sided paired t", "one-sided paired Wilcoxon"})

    def test_small_sample_aicc_is_explicitly_undefined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials_df = get_example_trials_df(n_subjects=1, n_sessions=1, n_trials=3)
            input_path = root / "trials.csv"
            trials_df.to_csv(input_path, index=False)
            result = run_comparison(
                input_path, root / "fits", models=(name for name in ["Q_basic_bias"]),
                methods=["ML"], repeats=1,
            )
            self.assertEqual(result["manifest"]["models"], ["Q_basic_bias"])
            self.assertTrue(np.isinf(result["fits_df"].iloc[0].AICc))
            omitted = result["manifest"]["omitted_comparison_metrics"]
            self.assertEqual(omitted[0]["metric"], "AICc")
            stored = json.loads((root / "fits" / "fits.json").read_text())
            self.assertIsNone(stored[0]["AICc"])


if __name__ == "__main__":
    unittest.main()
