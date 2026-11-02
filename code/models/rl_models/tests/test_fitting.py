"""Small reproducible fitting, grouped cross-validation and comparison checks."""

import copy
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from rl_models import model_comparison_helper_functions as comparison
from rl_models import model_fitting_helper_functions as fitting
from rl_models.model_implementation_model_classes import Q_model, Q_model_bias
from rl_models.sessions import Session


def sessions_for_subject(subject="mouse"):
    """Three short recordings; one reset sequence per recording."""
    rng = np.random.default_rng(4321)
    sessions = []
    for index in range(3):
        choices = rng.integers(0, 2, 24)
        rewards = rng.integers(0, 2, 24)
        forced = np.zeros(24, dtype=bool)
        forced[::7] = True
        mask = np.ones(24, dtype=bool)
        mask[0] = False
        sessions.append(Session(
            choices, rewards, forced, subject=subject, session_id=f"day{index}", ll_mask=mask
        ))
    return sessions


class FittingTests(unittest.TestCase):
    def test_priors_bounds_and_custom_bound_validation(self):
        params = [.3, 2.1, -.2]
        expected = stats.beta(2, 2).logpdf(.3) + stats.gamma(2, scale=2).logpdf(2.1)
        expected += stats.norm(scale=5).logpdf(-.2)
        self.assertAlmostEqual(fitting.log_prior_prob(params, ["unit", "pos", "unc"]), expected)
        model = Q_model_bias(param_bounds={"alpha": (.1, .8), "iTemp": (.2, 7.)})
        self.assertEqual(fitting.get_bounds(model), [(.1, .8), (.2, 7.), (None, None)])
        for bounds in [{"alpha": (-.1, .8)}, {"alpha": (.9, .2)}, {"unknown": (0., 1.)}]:
            with self.assertRaises(ValueError):
                fitting.get_bounds(Q_model_bias(param_bounds=bounds))
        with self.assertRaises(ValueError):
            fitting.get_init_param_values(["invalid"], rng=0)

    def test_prior_sampling_and_fits_do_not_change_global_rng(self):
        state = np.random.get_state()
        one = fitting.get_init_param_values(["unit", "pos", "unc"], rng=123)
        two = fitting.get_init_param_values(["unit", "pos", "unc"], rng=123)
        np.testing.assert_array_equal(one, two)
        sessions = sessions_for_subject()
        kwargs = dict(repeats=2, seed=123, optimizer_options={"maxiter": 200})
        first = fitting.fit_sessions(sessions, Q_model(), **kwargs)
        second = fitting.fit_sessions(sessions, Q_model(), **kwargs)
        np.testing.assert_array_equal(first["param_values"], second["param_values"])
        self.assertEqual(first["attempts"], second["attempts"])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_ml_map_objectives_and_information_criteria_use_scored_targets(self):
        sessions = sessions_for_subject()
        model = Q_model_bias()
        n_scored = sum(s.n_free_choice_trials for s in sessions)
        for map_fit in [False, True]:
            result = fitting.fit_sessions(
                sessions, model, repeats=2, seed=11, MAP_fit=map_fit,
                optimizer_options={"maxiter": 200}
            )
            ll = sum(model.session_log_likelihood(s, result["param_values"]) for s in sessions)
            self.assertAlmostEqual(result["log_likelihood"], ll)
            self.assertEqual(result["n_free_choice_trials"], n_scored)
            self.assertAlmostEqual(result["AIC"], -2 * ll + 2 * model.n_params)
            self.assertAlmostEqual(result["BIC"], -2 * ll + np.log(n_scored) * model.n_params)
            self.assertAlmostEqual(result["nll_per_trial"], -ll / n_scored)
            if map_fit:
                expected = ll + fitting.log_prior_prob(result["param_values"], model.param_ranges)
                self.assertAlmostEqual(result["log_posterior_prob"], expected)
                self.assertAlmostEqual(result["objective"], -expected)
            else:
                self.assertIsNone(result["log_posterior_prob"])
                self.assertAlmostEqual(result["objective"], -ll)
            self.assertEqual(result["total_attempts"], 2)
            self.assertIn("best_success", result)
            self.assertEqual(result["n_sessions"], 3)

    def test_independent_sessions_reset_values_and_history(self):
        first, second, _ = sessions_for_subject()
        model = Q_model()
        params = [.25, 3.5]
        expected = model.session_log_likelihood(first, params) + model.session_log_likelihood(second, params)
        self.assertEqual(fitting.multi_session_log_likelihood([first, second], model, params), expected)
        self.assertAlmostEqual(model.forward(second, params)["choice_probs"][0, 0], .5)

    def test_all_fixed_bounds_handle_scipy_special_optimizer_result(self):
        model = Q_model(param_bounds={"alpha": (.3, .3), "iTemp": (2., 2.)})
        result = fitting.fit_sessions(sessions_for_subject(), model, repeats=1, MAP_fit=True, seed=0)
        np.testing.assert_array_equal(result["param_values"], [.3, 2.])
        self.assertTrue(result["best_success"])
        self.assertEqual(result["total_attempts"], 1)
        self.assertTrue(np.isfinite(result["objective"]))

    def test_loso_keeps_all_blocks_from_one_recording_out_of_training(self):
        source = sessions_for_subject()
        blocks = []
        for recording in source:
            for index, positions in enumerate([slice(0, 12), slice(12, 24)]):
                block = Session(
                    recording.choices[positions], recording.outcomes[positions], ~recording.free_choice[positions],
                    subject="mouse", session_id=f"{recording.info.session_id}_block{index}",
                    ll_mask=recording.ll_mask[positions],
                )
                block.info.source_session_id = recording.info.session_id
                blocks.append(block)
        result = comparison.cross_validated_scores(
            blocks, Q_model(), MAP_fit=True, repeats=1, seed=77,
            optimizer_options={"maxiter": 150}
        )
        self.assertEqual(result["n_folds"], 3)
        self.assertEqual(result["n_free_choice_trials"], sum(s.n_free_choice_trials for s in blocks))
        self.assertAlmostEqual(result["CVLL"], sum(f["log_likelihood"] for f in result["folds"]))
        self.assertAlmostEqual(result["CV_nll_per_trial"], -result["CVLL"] / result["n_free_choice_trials"])
        for fold in result["folds"]:
            self.assertNotIn(fold["held_out_session"], fold["training_source_sessions"])
            self.assertEqual(len(fold["held_out_reset_sequences"]), 2)
            self.assertEqual(len(fold["fit"]["session_ids"]), 4)
            self.assertEqual(fold["fit"]["method"], "MAP")

    def test_loso_rejects_one_physical_session_even_with_multiple_reset_blocks(self):
        first, second, _ = sessions_for_subject()
        first.info.source_session_id = second.info.source_session_id = "one_recording"
        with self.assertRaises(ValueError):
            comparison.cross_validated_scores([first, second], Q_model(), repeats=1)

    def test_fitting_rejects_mixed_subjects_duplicate_ids_and_empty_scoring(self):
        one = sessions_for_subject("one")[0]
        two = sessions_for_subject("two")[0]
        with self.assertRaises(ValueError):
            fitting.fit_sessions([one, two], Q_model(), repeats=1)
        with self.assertRaises(ValueError):
            fitting.fit_sessions([one, one], Q_model(), repeats=1)
        empty = Session([0, 1], [1, 0], [True, True], subject="one", session_id="forced")
        with self.assertRaises(ValueError):
            fitting.fit_sessions([empty], Q_model(), repeats=1)

    def test_comparison_identity_includes_ordered_inputs_masks_and_trial_ids(self):
        original = sessions_for_subject()[0]
        baseline = fitting.scored_data_identity([original])["scored_targets_hash"]
        changed = copy.deepcopy(original)
        changed.trial_ids = np.asarray(changed.trial_ids) + 1000
        self.assertNotEqual(fitting.scored_data_identity([changed])["scored_targets_hash"], baseline)
        changed = copy.deepcopy(original)
        changed.info.source_session_id = "different_recording"
        self.assertNotEqual(fitting.scored_data_identity([changed])["scored_targets_hash"], baseline)
        changed = copy.deepcopy(original)
        changed.outcomes = changed.outcomes.copy()
        changed.outcomes[0] = 1 - changed.outcomes[0]
        self.assertNotEqual(fitting.scored_data_identity([changed])["scored_targets_hash"], baseline)


def comparison_rows():
    """One fit per subject/model/method on equal within-subject scored identities."""
    rows = []
    for subject, n_trials, shift in [("one", 20, 0.), ("two", 40, 1.), ("three", 30, 2.)]:
        for name, disadvantage in [("best", 0.), ("alternative", 3. * (1 + .1 * shift))]:
            ll = -10 - shift - disadvantage
            rows.append(dict(
                subject=subject, model=name, method="ML", n_params=2,
                log_likelihood=ll, nll=-ll, nll_per_trial=-ll / n_trials,
                n_free_choice_trials=n_trials, scored_targets_hash=subject,
                AIC=-2 * ll + 4, AICc=-2 * ll + 4 + 12 / (n_trials - 3),
                BIC=-2 * ll + 2 * np.log(n_trials), CVLL=ll - 1,
                cv_n_free_choice_trials=n_trials, cv_scored_targets_hash=subject,
                CV_nll_per_trial=-(ll - 1) / n_trials, accuracy=.7 - disadvantage / 100,
                CV_accuracy=.65 - disadvantage / 100,
            ))
    return pd.DataFrame(rows)


class ComparisonTests(unittest.TestCase):
    def test_comparison_uses_paired_subjects_and_correct_metric_direction(self):
        fits_df = comparison_rows()
        for metric in ["AIC", "BIC", "CVLL", "log_likelihood", "nll", "accuracy"]:
            result = comparison.model_comparison(fits_df, metric=metric, normalise=True)
            summary_df = result["summary_df"]
            self.assertEqual(summary_df.iloc[0]["model"], "best")
            self.assertEqual(summary_df.iloc[0]["n_subjects"], 3)
            values_df = result["subject_values_df"]
            self.assertEqual(len(values_df), 6)
            if metric == "AIC":
                expected = (fits_df.loc[fits_df.model.eq("best"), "AIC"] / np.array([20, 40, 30])).mean()
                self.assertAlmostEqual(summary_df.iloc[0]["mean"], expected)

    def test_comparison_rejects_incomplete_unmatched_and_duplicate_data(self):
        rows = comparison_rows()
        with self.assertRaises(ValueError):
            comparison.model_comparison(rows.iloc[:-1], metric="BIC")
        altered = rows.copy()
        altered.loc[0, "scored_targets_hash"] = "different targets"
        with self.assertRaises(ValueError):
            comparison.model_comparison(altered, metric="BIC")
        altered = rows.copy()
        altered.loc[0, "cv_scored_targets_hash"] = "different CV targets"
        with self.assertRaises(ValueError):
            comparison.model_comparison(altered, metric="CVLL")
        with self.assertRaises(ValueError):
            comparison.model_comparison(pd.concat([rows, rows.iloc[:1]], ignore_index=True), metric="BIC")

    def test_comparison_keeps_ml_and_map_separate(self):
        ml_df = comparison_rows()
        map_df = ml_df.copy()
        map_df["method"] = "MAP"
        result = comparison.model_comparison(pd.concat([ml_df, map_df], ignore_index=True), metric="AIC")
        self.assertEqual(set(result["summary_df"].method), {"ML", "MAP"})
        self.assertEqual(len(result["summary_df"]), 4)


if __name__ == "__main__":
    unittest.main()
