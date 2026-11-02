"""Independent equation checks for likelihoods, state timing and model nesting."""

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
from rl_models.model_implementation_helper_functions import get_trial_data
from rl_models.sessions import Session


def example_session(*, session_id="s1", subject="mouse"):
    """A complete input sequence with separate forced and scoring exclusions."""
    return Session(
        [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        forced_choice=[False, True, False, False, False, True, False, False, False, False],
        subject=subject,
        session_id=session_id,
        ll_mask=[False, True, True, True, False, True, True, True, True, True],
    )


def reference_q(session, *, alpha, beta, bias=0.0, kappa=0.0, eta=1.0, forgetting=0.0, target=0.0):
    """Direct scalar equations, independent of the package implementation."""
    choices, rewards, scored = get_trial_data(session)
    values = np.empty((2, len(choices)))
    histories = np.empty(len(choices))
    right_logits = np.empty(len(choices))
    q_left = q_right = history = 0.0
    for t, (choice, reward) in enumerate(zip(choices, rewards)):
        values[:, t] = q_left, q_right
        histories[t] = history
        right_logits[t] = beta * (q_right - q_left + kappa * history - bias)
        if choice == 0:
            q_left = (1 - alpha) * q_left + alpha * reward
            q_right = (1 - forgetting) * q_right + forgetting * target
        else:
            q_right = (1 - alpha) * q_right + alpha * reward
            q_left = (1 - forgetting) * q_left + forgetting * target
        history = (1 - eta) * history + eta * (2 * choice - 1)
    p_left = expit(-right_logits)
    probabilities = np.vstack([p_left, 1 - p_left])
    trial_ll = np.log(5e-201 + probabilities[choices, np.arange(len(choices))])
    return values, histories, right_logits, float(trial_ll[scored].sum())


def reference_bayes(session, params, *, asymmetric, history="ema", high=0.75, low=0.25):
    """Hazard-first Bayesian update and the fitted belief-contrast policy."""
    choices, rewards, scored = get_trial_data(session)
    beta, hazard, bias = params[:3]
    kappa = params[3] if history != "none" else 0.0
    eta = params[4] if history == "ema" else 1.0
    belief, trace = 0.5, 0.0
    beliefs, posterior, right_logits = [], [], []
    for choice, reward in zip(choices, rewards):
        beliefs.append(belief)
        right_logits.append(beta * (2 * belief - 1 + kappa * trace - bias))
        predicted = hazard + (1 - 2 * hazard) * belief
        if asymmetric and not reward:
            belief = predicted
        else:
            likelihood_right = high if choice == 1 else low
            likelihood_left = low if choice == 1 else high
            if not reward:
                likelihood_right, likelihood_left = 1 - likelihood_right, 1 - likelihood_left
            numerator = likelihood_right * predicted
            denominator = numerator + likelihood_left * (1 - predicted)
            belief = numerator / (max(denominator, 1e-12) if asymmetric else denominator)
        posterior.append(belief)
        trace = (1 - eta) * trace + eta * (2 * choice - 1)
    right_logits = np.asarray(right_logits)
    probabilities = np.vstack([expit(-right_logits), 1 - expit(-right_logits)])
    trial_ll = np.log(5e-201 + probabilities[choices, np.arange(len(choices))])
    return np.asarray(beliefs), np.asarray(posterior), right_logits, float(trial_ll[scored].sum())


class ModelEquationTests(unittest.TestCase):
    def setUp(self):
        self.session = example_session()

    def test_requested_legacy_q_likelihoods_match_explicit_equations(self):
        cases = [
            (models.Q_model(), [.31, 3.7], {}),
            (models.Q_model_bias(), [.31, 3.7, -.12], {"bias": -.12}),
            (models.Q_model_forget(), [.31, 3.7, .18], {"forgetting": .18, "target": .5}),
            (
                models.Q_model_config(bias=True, prsv=True),
                [.31, 3.7, -.12, .28],
                {"bias": -.12, "kappa": .28},
            ),
            (
                models.Q_model_config(bias=True, prsv_ema=True, forget=True),
                [.31, 3.7, -.12, .28, .42, .18],
                {"bias": -.12, "kappa": .14, "eta": .42, "forgetting": .18, "target": .5},
            ),
        ]
        for model, params, extra in cases:
            with self.subTest(model=model.name):
                expected = reference_q(self.session, alpha=.31, beta=3.7, **extra)
                actual = model.forward(self.session, params)
                np.testing.assert_allclose(actual["values"], expected[0], atol=1e-14)
                np.testing.assert_allclose(actual["logit_right"], expected[2], atol=1e-14)
                self.assertAlmostEqual(actual["log_likelihood"], expected[3], places=12)
                self.assertEqual(model.session_log_likelihood(self.session, params), actual["log_likelihood"])

    def test_bayesian_policy_and_hazard_order_match_reference(self):
        for asymmetric, cls in [(False, models.BayesSym_2AB), (True, models.BayesAsym_2AB)]:
            for history in ["none", "single", "ema"]:
                model = cls(prsv=history == "single", prsv_ema=history == "ema")
                params = [4.1, .13, -.2] + ([.35] if history != "none" else [])
                params += [.27] if history == "ema" else []
                expected = reference_bayes(self.session, params, asymmetric=asymmetric, history=history)
                actual = model.forward(self.session, params)
                with self.subTest(asymmetric=asymmetric, history=history):
                    np.testing.assert_allclose(actual["belief_b"], expected[0], atol=1e-14)
                    np.testing.assert_allclose(actual["belief_post"], expected[1], atol=1e-14)
                    np.testing.assert_allclose(actual["logit_right"], expected[2], atol=1e-14)
                    self.assertAlmostEqual(actual["log_likelihood"], expected[3], places=12)

    def test_asymmetric_omission_is_hazard_only_and_environment_override_is_explicit(self):
        session = Session([1, 0, 1], [1, 0, 1], subject="mouse", session_id="rhos")
        session.info.reward_prob_high, session.info.reward_prob_low = .9, .1
        params = [3.0, .1, 0.0]
        actual = models.BayesAsym_2AB(prsv=False).forward(session, params)
        expected = reference_bayes(session, params, asymmetric=True, history="none", high=.9, low=.1)
        np.testing.assert_allclose(actual["belief_b"], expected[0], atol=1e-14)
        self.assertAlmostEqual(actual["belief_post"][1], .1 + .8 * actual["belief_b"][1])
        ignored = models.BayesAsym_2AB(prsv=False, read_env_from_session=False).forward(session, params)
        self.assertGreater(abs(actual["belief_post"][0] - ignored["belief_post"][0]), .1)
        symmetric = models.BayesSym_2AB(prsv=False).forward(session, params)
        symmetric_reference = reference_bayes(session, params, asymmetric=False, history="none")
        np.testing.assert_allclose(symmetric["belief_b"], symmetric_reference[0], atol=1e-14)
        self.assertIs(models.BayesAsym, models.BayesAsym_2AB)

    def test_reward_ema_and_independent_forgetting_nest_exactly(self):
        alpha, beta, bias, kappa, eta = .23, 4.2, -.07, .31, .37
        params = [alpha, beta, bias, kappa, eta]
        ema = models.Q_history_model(learning="trial", history="ema")
        zero = models.Q_forgetting_model(target=0.0, history="ema")
        free = models.Q_forgetting_model(target="free", history="ema")
        reference = ema.forward(self.session, params)
        for model, values in [(zero, params + [alpha]), (free, params + [alpha, 0.0])]:
            actual = model.forward(self.session, values)
            for key in ["values", "values_post", "history", "logit_right"]:
                np.testing.assert_allclose(actual[key], reference[key], atol=1e-14)
            self.assertAlmostEqual(actual["log_likelihood"], reference["log_likelihood"], places=12)
        rewards = get_trial_data(self.session)[1]
        previous_sum = 0.0
        for t, reward in enumerate(rewards):
            expected_sum = (1 - alpha) * previous_sum + alpha * reward
            self.assertAlmostEqual(reference["values_post"][:, t].sum(), expected_sum)
            previous_sum = expected_sum

    def test_eta_one_and_zero_history_weight_recover_parent_models(self):
        for learning in ["visit", "trial"]:
            single = models.Q_history_model(learning=learning, history="single")
            ema = models.Q_history_model(learning=learning, history="ema")
            params = [.29, 2.6, .1, -.3]
            np.testing.assert_allclose(
                single.forward(self.session, params)["logit_right"],
                ema.forward(self.session, params + [1.0])["logit_right"],
                atol=1e-14,
            )
        basic = models.Q_model_bias().forward(self.session, [.29, 2.6, .1])
        history = models.Q_history_model(learning="visit", history="ema").forward(
            self.session, [.29, 2.6, .1, 0.0, .23]
        )
        np.testing.assert_allclose(basic["logit_right"], history["logit_right"], atol=1e-14)

    def test_legacy_half_history_requires_coefficient_conversion(self):
        legacy = models.Q_model_config(bias=True, prsv_ema=True, forget=True)
        consistent = models.Q_forgetting_model(target=.5, history="ema")
        lp = [.32, 3.4, -.11, .8, .23, .41]
        cp = [.32, 3.4, -.11, .4, .23, .41]
        np.testing.assert_allclose(
            legacy.forward(self.session, lp)["logit_right"],
            consistent.forward(self.session, cp)["logit_right"],
            atol=1e-14,
        )

    def test_forced_trials_update_state_but_masks_only_change_scoring(self):
        all_scored = Session(
            self.session.choices, self.session.outcomes, ~self.session.free_choice,
            subject="mouse", session_id="full"
        )
        model = models.Q_history_model(learning="trial", history="ema")
        params = [.4, 4., -.2, .3, .6]
        masked = model.forward(self.session, params)
        full = model.forward(all_scored, params)
        np.testing.assert_array_equal(masked["values"], full["values"])
        np.testing.assert_array_equal(masked["history"], full["history"])
        np.testing.assert_array_equal(masked["logit_right"], full["logit_right"])
        self.assertAlmostEqual(masked["log_likelihood"], masked["trial_log_likelihood"][masked["scored"]].sum())
        self.assertLess(masked["scored"].sum(), full["scored"].sum())
        # Trial5 is forced but changes the state used for scoring trial6.
        self.assertNotEqual(masked["values"][1, 6], masked["values"][1, 5])

    def test_exports_use_likelihood_policy_without_mutating_caller(self):
        choices, outcomes = self.session.choices.copy(), self.session.outcomes.copy()
        for model, params in [
            (models.Q_model_bias(), [.3, 4., .1]),
            (models.BayesSym_2AB(prsv=False, prsv_ema=True), [4., .08, -.1, .3, .2]),
            (models.BayesAsym_2AB(prsv=False, prsv_ema=True), [4., .08, -.1, .3, .2]),
        ]:
            result = model.forward(self.session, params)
            variables_df = model.get_trial_variables(self.session, params)
            self.assertEqual(variables_df.columns.nlevels, 3)
            np.testing.assert_allclose(
                variables_df.xs("logit_right", level=1, axis=1).iloc[:, 0], result["logit_right"]
            )
            chosen = result["choice_probs"][choices, np.arange(len(choices))]
            exported = variables_df.xs("choice_prob", level=1, axis=1).iloc[:, 0].to_numpy()
            np.testing.assert_allclose(exported[result["scored"]], chosen[result["scored"]])
            self.assertTrue(np.isnan(exported[~result["scored"]]).all())
        np.testing.assert_array_equal(self.session.choices, choices)
        np.testing.assert_array_equal(self.session.outcomes, outcomes)

    def test_dataframe_adapter_and_invalid_inputs_are_not_silently_recoded(self):
        trials_df = pd.DataFrame({
            ("choice", ""): ["poke_4", "poke_6", "poke_4"],
            ("outcome", ""): [1, 0, 1],
            ("forced_choice", ""): [False, True, False],
        })
        session = SimpleNamespace(trials_df=trials_df, n_trials=3, ll_mask=np.array([True, True, False]))
        choices, rewards, scored = get_trial_data(session)
        np.testing.assert_array_equal(choices, [0, 1, 0])
        np.testing.assert_array_equal(rewards, [1, 0, 1])
        np.testing.assert_array_equal(scored, [True, False, False])
        for bad in [np.array([0, 2]), np.array([0., np.nan])]:
            with self.assertRaises(ValueError):
                Session(bad, [0, 1])
        with self.assertRaises(ValueError):
            models.Q_model().forward(self.session, [.5, float("inf")])
        with self.assertRaises(ValueError):
            models.Q_model_config(prsv=True, prsv_ema=True)
        with self.assertRaises(ValueError):
            models.BayesSym_2AB(prsv=True, prsv_ema=True)

    def test_extreme_finite_logit_retains_original_finite_likelihood_floor(self):
        session = Session([0, 1, 0, 1], [1, 1, 1, 1], subject="mouse", session_id="extreme")
        actual = models.Q_model().forward(session, [1.0, 1e6])
        self.assertTrue(np.isfinite(actual["trial_log_likelihood"]).all())
        self.assertAlmostEqual(actual["trial_log_likelihood"][1], np.log(5e-201))
        np.testing.assert_allclose(actual["choice_probs"].sum(axis=0), 1.0)


@unittest.skipUnless(os.environ.get("ORIGINAL_RL_SOURCE"), "Optional original-source parity audit")
class OriginalSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_dir = Path(os.environ["ORIGINAL_RL_SOURCE"])
        helper_name = "model_implementation_helper_functions"
        saved_helper = sys.modules.get(helper_name)
        helper_spec = importlib.util.spec_from_file_location(helper_name, source_dir / f"{helper_name}.py")
        helper = importlib.util.module_from_spec(helper_spec)
        sys.modules[helper_name] = helper
        helper_spec.loader.exec_module(helper)
        spec = importlib.util.spec_from_file_location(
            "original_rl_parity_audit", source_dir / "model_implementation_model_classes.py"
        )
        cls.original = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.original)
        finally:
            if saved_helper is None:
                del sys.modules[helper_name]
            else:
                sys.modules[helper_name] = saved_helper

    def test_all_preserved_likelihoods_match_the_unedited_local_source(self):
        session = example_session()
        choices, rewards, _ = get_trial_data(session)
        old_session = SimpleNamespace(
            trials_df=pd.DataFrame({
                ("choice", ""): np.where(choices, "poke_6", "poke_4"),
                ("outcome", ""): rewards,
                ("forced_choice", ""): ~session.free_choice,
            }),
            n_trials=len(choices), ll_mask=session.ll_mask, info=session.info,
        )
        cases = [("Q_model", {}), ("Q_model_bias", {}), ("Q_model_forget", {})]
        for bias in (False, True):
            for history in ("none", "single", "ema"):
                for forget in (False, True):
                    cases.append(("Q_model_config", dict(
                        bias=bias, prsv=history == "single", prsv_ema=history == "ema", forget=forget
                    )))
        for name in ("BayesSym_2AB", "BayesAsym_2AB"):
            for history in ("none", "single", "ema"):
                cases.append((name, dict(prsv=history == "single", prsv_ema=history == "ema")))
        rng = np.random.default_rng(291026)
        for name, kwargs in cases:
            with self.subTest(model=name, kwargs=kwargs):
                original = getattr(self.original, name)(**kwargs)
                cleaned = getattr(models, name)(**kwargs)
                self.assertEqual(original.param_names, cleaned.param_names)
                self.assertEqual(original.param_ranges, cleaned.param_ranges)
                for _ in range(3):
                    params = [rng.beta(2, 2) if kind == "unit" else
                              rng.gamma(2, 2) if kind == "pos" else rng.normal(0, .5)
                              for kind in original.param_ranges]
                    self.assertAlmostEqual(
                        original.session_log_likelihood(old_session, params),
                        cleaned.session_log_likelihood(session, params),
                        places=12,
                    )


if __name__ == "__main__":
    unittest.main()
