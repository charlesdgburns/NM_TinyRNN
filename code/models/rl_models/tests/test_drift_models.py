"""Independent drift equations, controlled omissions and existing-model equivalences."""

import ast
import os
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit

from rl_models import DRIFT_MODELS, PRIMARY_MODELS, Q_drift_model, Session, make_model
from rl_models.model_implementation_model_classes import Q_forgetting_model, Q_history_model, Q_model


def example_session(mask=None):
    return Session(
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        forced_choice=[False, True, False, False, True, False, False, False, False, False],
        ll_mask=mask,
    )


def reference_other(session, *, retention, omission_rate, decrement, target="other"):
    """Scalar left/right equations, independent of the implementation's indexing."""
    left = right = 0.0
    before, after = [], []
    for choice, reward in zip(session.choices, session.outcomes):
        before.append((left, right))
        if reward:
            if choice == 0:
                left, right = retention * left + 1.0, right - decrement
            else:
                left, right = left - decrement, retention * right + 1.0
        elif choice == 0:
            left = (1 - omission_rate) * left + omission_rate * (right if target == "other" else 0.0)
        else:
            right = (1 - omission_rate) * right + omission_rate * (left if target == "other" else 0.0)
        after.append((left, right))
    return np.asarray(before).T, np.asarray(after).T


class DriftModelTests(unittest.TestCase):
    def test_published_other_equations_and_local_likelihood_match_independent_reference(self):
        session = example_session()
        rng = np.random.default_rng(412)
        for _ in range(6):
            alpha1, alpha2, decrement = rng.uniform(.05, .95, 3)
            beta = rng.uniform(.2, 4)
            model = Q_drift_model("other")
            result = model.forward(session, [alpha1, alpha2, decrement, beta])
            before, after = reference_other(
                session, retention=alpha1, omission_rate=alpha2, decrement=decrement
            )
            np.testing.assert_allclose(result["values"], before, atol=1e-14)
            np.testing.assert_allclose(result["values_post"], after, atol=1e-14)
            logits = beta * (before[1] - before[0])
            p_left = expit(-logits)
            probabilities = np.array([p_left, 1 - p_left])
            chosen = probabilities[session.choices, np.arange(session.n_trials)]
            expected_ll = np.log(5e-201 + chosen)[result["scored"]].sum()
            np.testing.assert_allclose(result["logit_right"], logits, atol=1e-14)
            self.assertAlmostEqual(result["log_likelihood"], expected_ll, places=12)

    def test_rewarded_values_are_unclipped_and_decrement_is_not_side_bias(self):
        session = Session([0, 0, 0, 0], [1, 1, 1, 1])
        model = Q_drift_model("other")
        result = model.forward(session, [1.0, .4, .3, 2.0])
        np.testing.assert_allclose(result["values_post"][0], [1, 2, 3, 4])
        np.testing.assert_allclose(result["values_post"][1], [-.3, -.6, -.9, -1.2])
        np.testing.assert_allclose(result["values"][:, 0], [0, 0])
        self.assertEqual(result["logit_right"][0], 0.0)

    def test_matched_zero_control_changes_only_the_omission_target(self):
        params = [.8, .3, .2, 2.0]
        reward_only = Session([0, 1, 0, 0, 1], [1, 1, 1, 1, 1])
        other = Q_drift_model("other")
        matched = Q_drift_model("zero_matched")
        np.testing.assert_array_equal(
            other.forward(reward_only, params)["values_post"],
            matched.forward(reward_only, params)["values_post"],
        )
        common = Session([0, 1, 0], [1, 1, 0])
        actual_other, actual_zero = other.forward(common, params), matched.forward(common, params)
        np.testing.assert_array_equal(actual_other["values"], actual_zero["values"])
        unchosen_value = actual_other["values"][1, 2]
        self.assertAlmostEqual(
            actual_other["values_post"][0, 2] - actual_zero["values_post"][0, 2],
            params[1] * unchosen_value,
        )
        self.assertEqual(actual_other["values_post"][1, 2], unchosen_value)
        before, after = reference_other(common, retention=.8, omission_rate=.3, decrement=.2, target="zero")
        np.testing.assert_allclose(actual_zero["values"], before, atol=1e-14)
        np.testing.assert_allclose(actual_zero["values_post"], after, atol=1e-14)

    def test_sum_difference_equations_distinguish_equalization_from_zero_decay(self):
        session = example_session()
        retention, omission_rate, decrement, beta = .8, .3, .2, 2.0
        for rule in ("other", "zero_matched"):
            result = Q_drift_model(rule).forward(session, [retention, omission_rate, decrement, beta])
            total = result["values"].sum(axis=0)
            difference = result["values"][1] - result["values"][0]
            expected_difference = []
            for t, (choice, reward) in enumerate(zip(session.choices, session.outcomes)):
                side = 2 * choice - 1
                if reward:
                    next_difference = (1 + retention) / 2 * difference[t]
                    next_difference += side * ((retention - 1) / 2 * total[t] + 1 + decrement)
                elif rule == "other":
                    next_difference = (1 - omission_rate) * difference[t]
                    self.assertAlmostEqual(
                        result["values_post"][:, t].sum(), total[t] - side * omission_rate * difference[t]
                    )
                else:
                    next_difference = (1 - omission_rate / 2) * difference[t] - side * omission_rate / 2 * total[t]
                expected_difference.append(next_difference)
            np.testing.assert_allclose(
                result["values_post"][1] - result["values_post"][0], expected_difference, atol=1e-14
            )

    def test_zero_drift_is_existing_zero_forgetting_under_rate_conversion(self):
        session = example_session()
        alpha, retention, beta, bias, prsv, eta = .3, .73, 3.4, -.2, .4, .25
        drift = Q_drift_model("zero", bias=True, history="ema")
        existing = Q_forgetting_model(target=0.0, history="ema")
        actual = drift.forward(session, [alpha, retention, beta, bias, prsv, eta])
        expected = existing.forward(session, [alpha, beta, bias, prsv, eta, 1 - retention])
        for name in ["values", "values_post", "history", "history_post", "choice_probs", "logit_right"]:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-14)
        self.assertAlmostEqual(actual["log_likelihood"], expected["log_likelihood"], places=12)

    def test_zero_drift_contains_basic_q_and_reward_ema(self):
        session = example_session()
        drift = Q_drift_model("zero")
        actual = drift.forward(session, [.3, 1.0, 2.0])
        basic = Q_model().forward(session, [.3, 2.0])
        np.testing.assert_allclose(actual["logit_right"], basic["logit_right"], atol=1e-14)
        ema = Q_history_model("trial", "none", bias=False).forward(session, [.3, 2.0])
        actual = drift.forward(session, [.3, .7, 2.0])
        np.testing.assert_allclose(actual["values_post"], ema["values_post"], atol=1e-14)

    def test_b_zero_ablation_removes_only_the_decrement_parameter(self):
        session = example_session()
        for rule in ("other", "zero_matched"):
            free = Q_drift_model(rule).forward(session, [.8, .3, 0.0, 2.0])
            fixed = Q_drift_model(rule, decaying_bias=False).forward(session, [.8, .3, 2.0])
            np.testing.assert_array_equal(free["values_post"], fixed["values_post"])
            self.assertEqual(free["log_likelihood"], fixed["log_likelihood"])

    def test_history_extensions_have_consistent_single_ema_and_zero_weight_limits(self):
        session = example_session()
        for rule, core in [("zero", [.3, .7, 2.0]), ("other", [.8, .3, .2, 2.0])]:
            plain = Q_drift_model(rule).forward(session, core)
            single = Q_drift_model(rule, bias=True, history="single").forward(session, core + [-.1, .3])
            ema = Q_drift_model(rule, bias=True, history="ema").forward(session, core + [-.1, .3, 1.0])
            zero = Q_drift_model(rule, bias=True, history="ema").forward(session, core + [0.0, 0.0, .4])
            np.testing.assert_allclose(single["logit_right"], ema["logit_right"], atol=1e-14)
            np.testing.assert_array_equal(plain["logit_right"], zero["logit_right"])
            self.assertEqual(single["history_post"][1], 1.0)  # Forced right input updates history.

    def test_forced_inputs_and_masks_preserve_states_and_export_the_fitted_policy(self):
        session = example_session(mask=[False, True, True, True, False, True, True, False, True, True])
        full = example_session()
        model = Q_drift_model("other", bias=True, history="ema")
        params = [.8, .3, .2, 2.0, -.1, .3, .2]
        actual, reference = model.forward(session, params), model.forward(full, params)
        np.testing.assert_array_equal(actual["values_post"], reference["values_post"])
        np.testing.assert_array_equal(actual["history"], reference["history"])
        self.assertEqual(actual["log_likelihood"], actual["trial_log_likelihood"][actual["scored"]].sum())
        exported = model.get_trial_variables(session, params)
        np.testing.assert_array_equal(exported[("model_vars", "logit_right", "")], actual["logit_right"])
        self.assertEqual(exported.columns.nlevels, 3)
        np.testing.assert_array_equal(model.session_log_likelihood(session, params, True), actual["values"])

    def test_additional_registry_is_opt_in_and_bounds_are_validated(self):
        self.assertFalse(set(DRIFT_MODELS) & set(PRIMARY_MODELS))
        counts = [3, 4, 4, 3, 6, 7, 7, 6, 3, 6]
        for name, count in zip(DRIFT_MODELS, counts):
            model = make_model(name)
            self.assertEqual(model.name, name)
            self.assertEqual(model.n_params, count)
        for name in PRIMARY_MODELS:
            self.assertEqual(make_model(name).name, name)
        for kwargs in [dict(rule="bad"), dict(history="bad"), dict(param_bounds={"alpha1": (-.1, 1.0)})]:
            with self.assertRaises(ValueError):
                Q_drift_model(**kwargs)
        model = make_model("Q_drift_other", param_bounds={"alpha1": (.2, .8)})
        with self.assertRaises(ValueError):
            model.forward(example_session(), [.9, .3, .2, 2.0])
        with self.assertRaises(ValueError):
            model.forward(example_session(), [.7, .3, -.2, 2.0])


@unittest.skipUnless(os.environ.get("TINY_RNN_SOURCE"), "Set TINY_RNN_SOURCE to audit unedited source kernels")
class OriginalDriftSourceTests(unittest.TestCase):
    def test_unedited_upstream_step_kernels_match_for_two_action_adaptation(self):
        """Run original function bodies without requiring their numba decoration.

        Only the requested top-level function is extracted. Its statements are
        unchanged; imports, classes, module execution and JIT are not exercised.
        The two-action adaptation fixes p_transit=1 and second_step=choice.
        Source softmax saturation is not claimed bit-identical to local policy.
        """
        source_dir = Path(os.environ["TINY_RNN_SOURCE"]) / "agents/RTS_RL_agents_v1"
        session = example_session()
        rng = np.random.default_rng(903)
        for rule, filename in [("zero", "Model_based_decay.py"), ("other", "Model_based_mix_decay.py")]:
            path = source_dir / filename
            tree = ast.parse(path.read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "_MB_step_core_variables")
            function.decorator_list = []
            isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            namespace = {"np": np}
            exec(compile(isolated, str(path), "exec"), namespace)
            source_step = namespace["_MB_step_core_variables"]
            for _ in range(4):
                rates = rng.uniform(0, 1, 3 if rule == "other" else 2).tolist()
                beta = 2.1
                model = Q_drift_model(rule)
                actual = model.forward(session, rates + [beta])
                Q = np.zeros(2)
                source_before, source_after = [], []
                for choice, reward in zip(session.choices, session.outcomes):
                    source_before.append(Q.copy())
                    Q, Q_readout = source_step(*rates, 1.0, choice, choice, reward, Q, Q)
                    source_after.append(Q.copy())
                    np.testing.assert_array_equal(Q_readout, Q)
                np.testing.assert_array_equal(actual["values"], np.asarray(source_before).T)
                np.testing.assert_array_equal(actual["values_post"], np.asarray(source_after).T)


if __name__ == "__main__":
    unittest.main()
