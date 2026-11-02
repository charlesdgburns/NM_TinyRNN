"""Explicit paper-equation and released-code mean-forgetting baselines.

Ji-An et al. (2025), Methods Eq. 10 and released MBf kernel at commit
a8fc159a2b2d451be6756ae244b29b7d991aac1f. The paper updates only the
unchosen value towards .5; MBf first learns and then forgets BOTH values.
Paper initialization is left unspecified by Eq10: .5 is an explicit matched
choice here, with a separate zero-init sensitivity. No bias/history extensions.
"""

import numpy as np

if __package__:
    from .model_implementation_helper_functions import get_trial_data
    from .model_implementation_model_classes import _Model, _parameters, _set_parameters, _finish_forward
else:
    from model_implementation_helper_functions import get_trial_data
    from model_implementation_model_classes import _Model, _parameters, _set_parameters, _finish_forward


class Q_mean_model(_Model):
    """Takam-style scalar reference, with explicit rule and initialization.

    alpha: chosen prediction-error learning rate [0,1]. retention: paper
    unchosen RETENTION [0,1]; forget_rate: released BOTH-value update rate
    [0,1]. iTemp: nonnegative inverse temperature. Initial values are fixed,
    not additional fitted parameters. Every input updates, including forced;
    likelihood is computed only for the session's scored free choices.
    """

    def __init__(self, rule="paper", *, initial_value=.5, param_bounds=None):
        if rule not in ("paper", "released"):
            raise ValueError("Choose paper or released")
        if initial_value not in (0., .5) or (rule == "released" and initial_value != .5):
            raise ValueError("Released code starts at .5; paper comparison allows explicit 0/.5")
        self.rule, self.initial_value = rule, float(initial_value)
        if rule == "released":
            self.name = "Q_release_mean"
        else:
            self.name = "Q_paper_mean_init_half" if initial_value == .5 else "Q_paper_mean_init_zero"
        names = ["alpha", "forget_rate" if rule == "released" else "retention", "iTemp"]
        _set_parameters(self, names, ["unit", "unit", "pos"], param_bounds)

    def forward(self, session, param_values):
        """Return pre/post-input states; the rule also applies on forced inputs."""
        pv = _parameters(self, param_values)
        choices, rewards, scored = get_trial_data(session)
        values = np.zeros((2, len(choices)))
        values_post = np.zeros_like(values)
        Q = np.full(2, self.initial_value)
        for t, (choice, reward) in enumerate(zip(choices, rewards)):
            values[:, t] = Q
            Q[choice] = (1. - pv.alpha) * Q[choice] + pv.alpha * reward
            if self.rule == "released":
                # The released kernel forgets both values AFTER chosen learning.
                Q = (1. - pv.forget_rate) * Q + .5 * pv.forget_rate
            else:
                Q[1 - choice] = pv.retention * Q[1 - choice] + .5 * (1. - pv.retention)
            values_post[:, t] = Q
        history = np.zeros(len(choices))
        result = _finish_forward(
            session, choices, scored, values, values_post, history, history.copy(), values[1] - values[0], pv.iTemp
        )
        result["history_bonus"] = history.copy()
        return result


def make_mean_models():
    """Return all three explicit mean-forgetting variants, without fitting."""
    models = (
        Q_mean_model("paper", initial_value=.5),
        Q_mean_model("paper", initial_value=0.),
        Q_mean_model("released", initial_value=.5),
    )
    return {model.name: model for model in models}
