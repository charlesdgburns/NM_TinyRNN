"""Two-action drift rules adapted from Ji-An, Benna and Mattar (2025).

The published zero-drift model maps to MB1; published other-drift is MB0md with
p_transit=1. Their rewarded rules differ as well as their omission rules.
``zero_matched`` is therefore a new control: it keeps MB0md's rewarded rule
and changes only the omission target from the alternative value to zero.

Source: jil095/tinyRNN, commit a8fc159a2b2d451be6756ae244b29b7d991aac1f,
agents/RTS_RL_agents_v1/{Model_based_decay,Model_based_mix_decay}.py.
The source's 'beta' retention is called retention here; its 'b' decrement is
called decay_bias to distinguish both from temperature and side bias.
"""

import numpy as np

if __package__:
    from .model_implementation_helper_functions import get_trial_data
    from .model_implementation_model_classes import (
        _Model,
        _choice_history,
        _finish_forward,
        _parameters,
        _set_parameters,
    )
else:
    from model_implementation_helper_functions import get_trial_data
    from model_implementation_model_classes import (
        _Model,
        _choice_history,
        _finish_forward,
        _parameters,
        _set_parameters,
    )


class Q_drift_model(_Model):
    """Published drift cores, with explicitly optional local policy extensions.

    ``zero``: chosen PE learning alpha; unchosen retention in [0,1].
    ``other``: rewarded Qc'=alpha1*Qc+1, Qu'=Qu-decay_bias; omission
    Qc'=(1-alpha2)*Qc+alpha2*Qu and unchanged Qu.
    ``zero_matched``: identical reward updates, but omission Qc'=(1-alpha2)*Qc.

    alpha1 is RETENTION, while alpha2 is an interpolation/update rate. Values
    start at zero and are never clipped. Thus other/matched values need not
    be positive, below one, or expected rewards. decaying_bias=False fixes
    the reward-contingent unchosen decrement to zero and removes its parameter.

    bias/history are our policy extensions, not original-paper parameters.
    The full signed history is right-positive; positive side bias favors left.
    Forced and masked inputs update every state; masks only affect likelihood.
    Pre-choice arrays predict the current choice, as in all local clean models.
    """

    def __init__(self, rule="other", *, bias=False, history="none", decaying_bias=True, param_bounds=None):
        if rule not in ("zero", "other", "zero_matched"):
            raise ValueError("rule must be zero, other or zero_matched")
        if history not in ("none", "single", "ema"):
            raise ValueError("history must be none, single or ema")
        self.rule, self.bias, self.history = rule, bool(bias), history
        self.decaying_bias = bool(decaying_bias) if rule != "zero" else False
        self.perseveration = history
        self.name = f"Q_drift_{rule}"
        if rule == "zero":
            names, ranges = ["alpha", "retention", "iTemp"], ["unit", "unit", "pos"]
        else:
            names, ranges = ["alpha1", "alpha2"], ["unit", "unit"]
            if self.decaying_bias:
                names += ["decay_bias"]
                ranges += ["unit"]
            else:
                self.name += "_no_decay"
            names += ["iTemp"]
            ranges += ["pos"]
        if self.bias:
            self.name += "_bias"
            names += ["bias"]
            ranges += ["unc"]
        if history != "none":
            self.name += "_prsvEMA" if history == "ema" else "_prsv"
            names += ["prsv"]
            ranges += ["unc"]
        if history == "ema":
            names += ["alp_p"]
            ranges += ["unit"]
        _set_parameters(self, names, ranges, param_bounds)

    def forward(self, session, param_values):
        """Return actual pre/post-input states and the local likelihood policy."""
        pv = _parameters(self, param_values)
        choices, rewards, scored = get_trial_data(session)
        history, history_post = _choice_history(choices, self.history, getattr(pv, "alp_p", 1.0))
        values = np.zeros((2, len(choices)))
        values_post = np.zeros_like(values)
        Q = np.zeros(2)
        for t, (choice, reward) in enumerate(zip(choices, rewards)):
            values[:, t] = Q
            unchosen = 1 - choice
            if self.rule == "zero":
                Q[choice] = (1.0 - pv.alpha) * Q[choice] + pv.alpha * reward
                Q[unchosen] = pv.retention * Q[unchosen]
            elif reward:
                Q[choice] = pv.alpha1 * Q[choice] + 1.0
                Q[unchosen] -= getattr(pv, "decay_bias", 0.0)
            else:
                target = Q[unchosen] if self.rule == "other" else 0.0
                Q[choice] += pv.alpha2 * (target - Q[choice])
            values_post[:, t] = Q
        history_bonus = getattr(pv, "prsv", 0.0) * history
        decision = values[1] - values[0] + history_bonus - getattr(pv, "bias", 0.0)
        result = _finish_forward(
            session, choices, scored, values, values_post, history, history_post, decision, pv.iTemp
        )
        result["history_bonus"] = history_bonus
        return result
