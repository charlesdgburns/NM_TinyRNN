"""Regression checks for release review: constrained penalties and public models."""

import ast
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl_models import MEAN_MODELS, Q_mean_model, Q_model, Session, make_mean_models, make_model
from rl_models.example import get_example_trials_df
from rl_models.model_fitting_helper_functions import fit_sessions
from rl_models.run_comparison import run_comparison


class InformationCriterionTests(unittest.TestCase):
    def test_fixed_parameters_do_not_receive_information_criterion_penalties(self):
        session = Session([0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0])
        model = Q_model(param_bounds={"alpha": (.3, .3), "iTemp": (2., 2.)})
        fitted = fit_sessions([session], model, repeats=1, seed=13)
        deviance = -2 * fitted["log_likelihood"]
        self.assertAlmostEqual(fitted["AIC"], deviance)
        self.assertAlmostEqual(fitted["AICc"], deviance)
        self.assertAlmostEqual(fitted["BIC"], deviance)
        self.assertEqual(fitted["n_params"], 2)  # Parameter-vector length remains unchanged.
        self.assertEqual(fitted["n_ic_params"], 0)

    def test_partially_fixed_parameters_use_only_the_remaining_free_dimension(self):
        session = Session([0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0])
        model = Q_model(param_bounds={"alpha": (.3, .3)})
        fitted = fit_sessions([session], model, repeats=1, seed=13)
        n = fitted["n_free_choice_trials"]
        deviance = -2 * fitted["log_likelihood"]
        self.assertAlmostEqual(fitted["AIC"], deviance + 2)
        self.assertAlmostEqual(fitted["AICc"], deviance + 2 + 4 / (n - 2))
        self.assertAlmostEqual(fitted["BIC"], deviance + np.log(n))
        self.assertEqual(fitted["n_ic_params"], 1)


class MeanForgettingTests(unittest.TestCase):
    def test_public_registry_exposes_basic_and_all_mean_variants(self):
        self.assertEqual(make_model("Q_basic").n_params, 2)
        self.assertEqual(tuple(make_mean_models()), MEAN_MODELS)
        for name in MEAN_MODELS:
            model = make_model(name)
            self.assertIsInstance(model, Q_mean_model)
            self.assertEqual(model.name, name)
            self.assertEqual(model.n_params, 3)
        with self.assertRaises(ValueError):
            make_model("Q_release_mean", param_bounds={"forget_rate": (-.1, 1.)})

    def test_paper_and_release_have_different_chosen_value_updates(self):
        session = Session([0], [1])
        paper = make_model("Q_paper_mean_init_half").forward(session, [.4, .7, 2.])
        release = make_model("Q_release_mean").forward(session, [.4, .3, 2.])
        np.testing.assert_allclose(paper["values_post"][:, 0], [.7, .5])
        np.testing.assert_allclose(release["values_post"][:, 0], [.64, .5])

    def test_paper_unchosen_forgetting_matches_closed_form_run(self):
        session = Session([0, 0, 0], [1, 1, 1])
        result = make_model("Q_paper_mean_init_zero").forward(session, [.3, .6, 2.])
        t = np.arange(1, 4)
        np.testing.assert_allclose(result["values_post"][0], 1 - .7 ** t)
        np.testing.assert_allclose(result["values_post"][1], .5 * (1 - .6 ** t))

    def test_forgetting_endpoints_and_original_q_nesting(self):
        session = Session([0, 1, 1, 0], [1, 0, 1, 0])
        original = Q_model().forward(session, [.3, 2.])
        nested = make_model("Q_paper_mean_init_zero").forward(session, [.3, 1., 2.])
        np.testing.assert_array_equal(nested["values_post"], original["values_post"])
        fully_forgotten = make_model("Q_release_mean").forward(session, [.3, 1., 2.])
        np.testing.assert_allclose(fully_forgotten["values_post"], .5)

    def test_forced_inputs_masks_reset_and_exports_use_the_same_policy(self):
        full = Session([0, 1, 1, 0], [1, 0, 1, 0])
        masked = Session(full.choices, full.outcomes, [False, True, False, False],
                         ll_mask=[False, True, True, True])
        for name in MEAN_MODELS:
            model = make_model(name)
            reference = model.forward(full, [.3, .6, 2.])
            result = model.forward(masked, [.3, .6, 2.])
            np.testing.assert_array_equal(result["values_post"], reference["values_post"])
            np.testing.assert_allclose(result["values"][:, 0], model.initial_value)
            self.assertEqual(result["log_likelihood"], result["trial_log_likelihood"][[2, 3]].sum())
            exported = model.get_trial_variables(masked, [.3, .6, 2.])
            np.testing.assert_array_equal(exported[("model_vars", "logit_right", "")], result["logit_right"])

    def test_mean_models_execute_saved_ml_map_loso_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials_path = root / "trials.csv"
            get_example_trials_df(n_subjects=2, n_sessions=2, n_trials=24).to_csv(trials_path, index=False)
            result = run_comparison(trials_path, root / "fits", models=MEAN_MODELS,
                                    methods=["ML", "MAP"], repeats=1, compute_cv=True)
            self.assertTrue(result["manifest"]["complete"])
            self.assertEqual(len(result["fits_df"]), 12)
            self.assertEqual(set(result["fits_df"].model), set(MEAN_MODELS))
            self.assertEqual(set(result["fits_df"].n_ic_params), {3})
            for specification in result["manifest"]["model_specifications"]:
                model = make_model(specification["name"])
                self.assertEqual(specification["options"]["initial_value"], model.initial_value)
                self.assertEqual(specification["options"]["rule"], model.rule)


@unittest.skipUnless(os.environ.get("TINY_RNN_SOURCE"), "Optional released mean-kernel source audit")
class ReleasedMeanSourceTests(unittest.TestCase):
    def test_released_step_body_matches_without_importing_numba(self):
        path = Path(os.environ["TINY_RNN_SOURCE"]) / "agents/RTS_RL_agents_v1/Model_based_forgetful.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_MB_step_core_variables")
        function.decorator_list = []
        isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"np": np, "INIT_Q": .5}
        exec(compile(isolated, str(path), "exec"), namespace)
        step = namespace["_MB_step_core_variables"]
        session = Session([0, 1, 1, 0, 0], [1, 0, 1, 0, 1])
        for alpha, forgetting in [(.3, .2), (0., 1.), (1., 0.)]:
            values = np.full(2, .5)
            expected = []
            for choice, reward in zip(session.choices, session.outcomes):
                values, readout = step(alpha, forgetting, 1., choice, choice, reward, values, values)
                np.testing.assert_array_equal(values, readout)
                expected.append(values.copy())
            result = make_model("Q_release_mean").forward(session, [alpha, forgetting, 2.])
            np.testing.assert_array_equal(result["values_post"], np.asarray(expected).T)


if __name__ == "__main__":
    unittest.main()
