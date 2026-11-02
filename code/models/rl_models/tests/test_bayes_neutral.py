"""Neutral reward probabilities are valid, uninformative Bayesian observations."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from rl_models import model_implementation_model_classes as models
from rl_models.model_implementation_helper_functions import log_safe
from rl_models.sessions import Session


def neutral_session():
    """A mixed input sequence with explicitly equal reward probabilities."""
    return Session(
        [0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0], [False, True, False, False, False, False],
        subject="neutral_mouse", session_id="neutral_session",
        ll_mask=[True, True, True, False, True, True],
        metadata={"reward_prob_high": .5, "reward_prob_low": .5},
    )


class NeutralBayesianTests(unittest.TestCase):
    def test_equal_reward_probabilities_leave_only_hazard_propagation(self):
        belief, hazard = .8, .13
        expected = (1 - hazard) * belief + hazard * (1 - belief)
        for asymmetric in (False, True):
            for choice in (0, 1):
                for reward in (0, 1):
                    with self.subTest(asymmetric=asymmetric, choice=choice, reward=reward):
                        actual = models._bayes_update(
                            belief, choice, reward, hazard, .5, .5, asymmetric=asymmetric
                        )
                        self.assertAlmostEqual(actual, expected)

    def test_asymmetric_session_neutral_setting_has_constant_belief_and_history_only_policy(self):
        session = neutral_session()
        beta, hazard, bias, kappa, eta = 3., .13, .2, .4, .3
        model = models.BayesAsym_2AB(prsv=False, prsv_ema=True)
        result = model.forward(session, [beta, hazard, bias, kappa, eta])
        np.testing.assert_allclose(result["belief_b"], .5, atol=0, rtol=0)
        np.testing.assert_allclose(result["belief_post"], .5, atol=0, rtol=0)
        np.testing.assert_allclose(result["values"], .5, atol=0, rtol=0)
        expected_logit = beta * (kappa * result["history"] - bias)
        np.testing.assert_allclose(result["logit_right"], expected_logit)
        p_left = expit(-expected_logit)
        probabilities = np.vstack([p_left, 1 - p_left])
        chosen = probabilities[session.choices, np.arange(session.n_trials)]
        expected_ll = log_safe(chosen)[result["scored"]].sum()
        self.assertAlmostEqual(result["log_likelihood"], expected_ll)

    def test_symmetric_class_still_ignores_session_environment_metadata(self):
        session = neutral_session()
        # The symmetric class retains its original fixed .75/.25 environment;
        # neutral probabilities are supported by its step helper, not a new flag.
        result = models.BayesSym_2AB(prsv=False).forward(session, [3., .13, .2])
        self.assertNotEqual(result["belief_post"][0], .5)
        with self.assertRaises(ValueError):
            bad = Session([0, 1], [1, 0], metadata={"reward_prob_high": .2, "reward_prob_low": .8})
            models.BayesAsym_2AB(prsv=False).forward(bad, [3., .13, .2])


@unittest.skipUnless(os.environ.get("ORIGINAL_RL_SOURCE"), "Optional original-source parity audit")
class NeutralSourceParityTests(unittest.TestCase):
    def test_neutral_asymmetric_likelihood_matches_unedited_source(self):
        source_dir = Path(os.environ["ORIGINAL_RL_SOURCE"])
        helper_name = "model_implementation_helper_functions"
        previous_helper = sys.modules.get(helper_name)
        try:
            helper_spec = importlib.util.spec_from_file_location(
                helper_name, source_dir / "model_implementation_helper_functions.py"
            )
            helper = importlib.util.module_from_spec(helper_spec)
            sys.modules[helper_name] = helper
            helper_spec.loader.exec_module(helper)
            spec = importlib.util.spec_from_file_location(
                "original_neutral_bayes_audit", source_dir / "model_implementation_model_classes.py"
            )
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
        finally:
            if previous_helper is None:
                sys.modules.pop(helper_name, None)
            else:
                sys.modules[helper_name] = previous_helper
        session = neutral_session()
        old_session = SimpleNamespace(
            trials_df=pd.DataFrame({
                ("choice", ""): np.where(session.choices, "poke_6", "poke_4"),
                ("outcome", ""): session.outcomes,
                ("forced_choice", ""): ~session.free_choice,
            }), n_trials=session.n_trials, ll_mask=session.ll_mask, info=session.info,
        )
        for history in ("none", "single", "ema"):
            options = dict(prsv=history == "single", prsv_ema=history == "ema")
            params = [3., .13, .2] + ([.4] if history != "none" else [])
            if history == "ema":
                params += [.3]
            with self.subTest(history=history):
                expected = original.BayesAsym_2AB(**options).session_log_likelihood(old_session, params)
                actual = models.BayesAsym_2AB(**options).session_log_likelihood(session, params)
                self.assertAlmostEqual(actual, expected, places=12)
                for choice in (0, 1):
                    for reward in (0, 1):
                        expected_step = original._bayes_update(.8, choice, reward, .13, .5, .5)
                        actual_step = models._bayes_update(.8, choice, reward, .13, .5, .5)
                        self.assertAlmostEqual(actual_step, expected_step, places=14)


if __name__ == "__main__":
    unittest.main()
