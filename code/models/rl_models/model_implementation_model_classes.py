"""Reviewed two-action Q-learning and Bayesian model definitions.

Arrays are PRE-choice: state at column t predicts choice t using inputs before
trial t. States reset on each forward/session call. All trials update state;
only explicitly eligible free choices contribute to likelihood. Post-input
arrays also retain the final input's update for mechanistic inspection.

Legacy classes preserve their original fitted equations. New Q_history_model
and Q_forgetting_model use the full signed choice-history scale consistently.
Q_model_bias_prsvEMA_trial retains the local half-scale EMA parameterization.
All histories are positive for recent right choices. Positive bias favors left.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

if __package__:
    from .model_implementation_helper_functions import get_trial_data, log_safe, softmax_probs
else:
    from model_implementation_helper_functions import get_trial_data, log_safe, softmax_probs

__all__ = [
    "Q_model", "Q_model_bias", "Q_model_forget", "Q_model_config",
    "Q_model_bias_prsvEMA_trial", "Q_model_bias_prsvEMA_trial_positive",
    "BayesSym_2AB", "BayesAsym_2AB", "BayesAsym", "Q_history_model", "Q_forgetting_model",
    "Q_drift_model", "Q_mean_model", "make_mean_models", "make_model", "make_models",
    "PRIMARY_MODELS", "DRIFT_MODELS", "MEAN_MODELS",
]


# Shared parameter validation and forward-output assembly.


def _set_parameters(model, names, ranges, param_bounds):
    model.param_names = list(names)
    model.param_ranges = list(ranges)
    model.n_params = len(names)
    model.param_bounds = {} if param_bounds is None else dict(param_bounds)
    unknown = set(model.param_bounds) - set(names)
    if unknown:
        raise ValueError(f"Bounds supplied for unknown parameters: {sorted(unknown)}")
    for name, bounds in model.param_bounds.items():
        if not isinstance(bounds, (tuple, list, np.ndarray)) or np.ndim(bounds) != 1 or len(bounds) != 2:
            raise ValueError(f"Bounds for {name} must be (lower, upper)")
        lower, upper = (None if value is None else float(value) for value in bounds)
        if any(value is not None and not np.isfinite(value) for value in (lower, upper)):
            raise ValueError(f"Bounds for {name} must be finite or None")
        domain = ranges[names.index(name)]
        if domain in ("unit", "pos") and (lower is None or lower < 0):
            raise ValueError(f"Lower bound for {name} leaves its {domain!r} domain")
        if domain == "unit" and (upper is None or upper > 1):
            raise ValueError(f"Upper bound for {name} leaves its unit domain")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound exceeds upper bound for {name}")
        model.param_bounds[name] = (lower, upper)


def _parameters(model, param_values):
    values = np.asarray(param_values, dtype=float)
    if values.shape != (model.n_params,) or not np.isfinite(values).all():
        raise ValueError(f"{model.name} requires {model.n_params} finite parameter values")
    for name, value, domain in zip(model.param_names, values, model.param_ranges):
        if (domain == "unit" and not 0 <= value <= 1) or (domain == "pos" and value < 0):
            raise ValueError(f"Parameter {name} is outside its {domain!r} domain")
        if name in model.param_bounds:
            lower, upper = model.param_bounds[name]
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                raise ValueError(f"Parameter {name} is outside its custom bounds")
    return SimpleNamespace(**dict(zip(model.param_names, values)))


def _choice_history(choices, mode, alp_p=1.0):
    """Full-scale pre/post-input right-choice trace, initially zero."""
    n_trials = len(choices)
    history = np.zeros(n_trials)
    history_post = np.zeros(n_trials)
    trace = 0.0
    for t, choice in enumerate(choices):
        history[t] = trace
        if mode == "single":
            trace = 2.0 * choice - 1.0
        elif mode == "ema":
            trace = (1.0 - alp_p) * trace + alp_p * (2.0 * choice - 1.0)
        history_post[t] = trace
    return history, history_post


def _finish_forward(session, choices, scored, values, values_post, history, history_post, decision, inverse_temp):
    with np.errstate(over="ignore", invalid="ignore"):
        logit_right = inverse_temp * decision
    if not np.isfinite(logit_right).all():
        raise ValueError("Nonfinite policy logit; inspect parameter bounds")
    choice_probs = softmax_probs(np.vstack([np.zeros_like(logit_right), logit_right]), 1.0)
    chosen_probability = choice_probs[choices, np.arange(len(choices))]
    trial_log_likelihood = log_safe(chosen_probability)
    _, _, free_choice = get_trial_data(session, apply_ll_mask=False)
    return dict(
        values=values, values_post=values_post, history=history, history_post=history_post,
        decision=decision, logit_right=logit_right, logit_left=-logit_right, logits=logit_right,
        choice_probs=choice_probs, chosen_probability=chosen_probability,
        trial_log_likelihood=trial_log_likelihood, free_choice=free_choice, scored=scored,
        log_likelihood=float(trial_log_likelihood[scored].sum()),
    )


def _q_forward(model, session, param_values, *, forgetting, target, history_mode, history_scale=1.0):
    pv = _parameters(model, param_values)
    choices, outcomes, scored = get_trial_data(session)
    n_trials = len(choices)
    alpha = pv.alpha
    forgetting_rate = alpha if forgetting == "alpha" else (pv.alp_f if forgetting == "free" else 0.0)
    decay_target = pv.decay_target if target == "free" else target
    history, history_post = _choice_history(choices, history_mode, getattr(pv, "alp_p", 1.0))
    values = np.zeros((2, n_trials))
    values_post = np.zeros_like(values)
    Q = np.zeros(2)
    for t, (choice, reward) in enumerate(zip(choices, outcomes)):
        values[:, t] = Q
        unchosen = 1 - choice
        Q[choice] = (1.0 - alpha) * Q[choice] + alpha * reward
        Q[unchosen] = (1.0 - forgetting_rate) * Q[unchosen] + forgetting_rate * decay_target
        values_post[:, t] = Q
    history_bonus = getattr(pv, "prsv", 0.0) * history_scale * history
    decision = values[1] - values[0] + history_bonus - getattr(pv, "bias", 0.0)
    result = _finish_forward(
        session, choices, scored, values, values_post, history, history_post, decision, pv.iTemp
    )
    result["history_bonus"] = history_bonus
    return result


class _Model:
    """Common inspectable likelihood/export API, without a fitting dependency."""

    def session_log_likelihood(self, session, param_values, return_Q_values=False):
        """Return summed scored log likelihood, or raw pre-choice values for legacy callers."""
        result = self.forward(session, param_values)
        return result["values"].copy() if return_Q_values else result["log_likelihood"]

    def get_trial_variables(self, session, param_values):
        """One row per input trial; columns are ('model_vars', variable, '').

        Choice probabilities use the SAME fitted policy as session likelihood.
        ``free_choice`` describes actual trial type, while ``scored`` additionally
        applies ll_mask. States and full probabilities remain available for all
        trials; legacy choice_prob/surprise are missing outside scored targets.
        """
        result = self.forward(session, param_values)
        choices, outcomes, _ = get_trial_data(session)
        n_trials = len(choices)
        values = result["values"]
        chosen_Q = values[choices, np.arange(n_trials)]
        unchosen_Q = values[1 - choices, np.arange(n_trials)]
        variables = dict(
            value_left=values[0], value_right=values[1], Q_left=values[0], Q_right=values[1],
            value_left_post=result["values_post"][0], value_right_post=result["values_post"][1],
            chosen_Q=chosen_Q, unchosen_Q=unchosen_Q, Q_diff=chosen_Q - unchosen_Q,
            value_difference=values[1] - values[0], total_value=values.sum(axis=0),
            prediction_error=outcomes - chosen_Q, choice_history=result["history"],
            choice_history_post=result["history_post"], decision_variable=result["decision"],
            perseveration_bonus=result["history_bonus"], logit_right=result["logit_right"],
            logit_left=result["logit_left"], choice_prob_left=result["choice_probs"][0],
            choice_prob_right=result["choice_probs"][1], chosen_probability=result["chosen_probability"],
            choice_prob=np.where(result["scored"], result["chosen_probability"], np.nan),
            surprise=np.where(result["scored"], -result["trial_log_likelihood"], np.nan),
            trial_log_likelihood=result["trial_log_likelihood"], free_choice=result["free_choice"],
            forced_choice_contrast=np.where(result["free_choice"], -0.5, 0.5), scored=result["scored"],
        )
        if "belief_b" in result:
            variables.update(belief_b=result["belief_b"], belief_post=result["belief_post"],
                             b_post=result["belief_post"], belief_difference=2.0 * result["belief_b"] - 1.0)
        for name, value in zip(self.param_names, param_values):
            variables[f"param_{name}"] = np.full(n_trials, value)
        index = session.trials_df.index if hasattr(session, "trials_df") else pd.RangeIndex(n_trials)
        variables_df = pd.DataFrame(variables, index=index)
        variables_df.columns = pd.MultiIndex.from_tuples([("model_vars", name, "") for name in variables_df])
        return variables_df


# Original Q classes. EMA history deliberately retains its HALF-scale kernel.


class Q_model_config(_Model):
    """Original configurable Q model; no-history, single or half-scale EMA history.

    alpha,alp_p,alp_f are unit rates; iTemp is nonnegative; bias is signed.
    Legacy single-history prsv has the original positive domain; EMA prsv is
    signed. Unchosen forgetting is towards .5, not zero. Do not compare prsv
    directly to Q_history_model EMA coefficients: kappa_new=kappa_legacy/2.
    """

    def __init__(self, bias=False, prsv=False, prsv_ema=False, forget=False, param_bounds=None):
        if prsv and prsv_ema:
            raise ValueError("Choose single-trial prsv OR prsv_ema, not both")
        self.bias, self.prsv, self.prsv_ema, self.forget = bool(bias), bool(prsv), bool(prsv_ema), bool(forget)
        self.name = "Q_model"
        names, ranges = ["alpha", "iTemp"], ["unit", "pos"]
        if self.bias:
            self.name += "_bias"
            names += ["bias"]
            ranges += ["unc"]
        if self.prsv:
            self.name += "_prsv"
            names += ["prsv"]
            ranges += ["pos"]
        if self.prsv_ema:
            self.name += "_prsvEMA"
            names += ["prsv", "alp_p"]
            ranges += ["unc", "unit"]
        if self.forget:
            self.name += "_forget"
            names += ["alp_f"]
            ranges += ["unit"]
        _set_parameters(self, names, ranges, param_bounds)

    def forward(self, session, param_values):
        """Pre-choice states and the exact original likelihood policy."""
        history_mode = "ema" if self.prsv_ema else ("single" if self.prsv else "none")
        return _q_forward(
            self, session, param_values, forgetting="free" if self.forget else "none", target=0.5,
            history_mode=history_mode, history_scale=0.5 if self.prsv_ema else 1.0,
        )


class Q_model(Q_model_config):
    """Basic chosen-only prediction-error Q learner: alpha and iTemp."""

    def __init__(self, param_bounds=None):
        super().__init__(param_bounds=param_bounds)


class Q_model_bias(Q_model_config):
    """Basic Q with a static left-value bias: alpha,iTemp,bias."""

    def __init__(self, param_bounds=None):
        super().__init__(bias=True, param_bounds=param_bounds)


class Q_model_forget(Q_model_config):
    """Original Q with unchosen forgetting to .5: alpha,iTemp,alp_f."""

    def __init__(self, param_bounds=None):
        super().__init__(forget=True, param_bounds=param_bounds)


# Legacy-scale reward EMA: the one-rule extension and its nonnegative variant.


class Q_model_bias_prsvEMA_trial(_Model):
    """Legacy-scale Q + bias + EMA history, with tied unchosen decay to zero.

    Parameters are alpha,iTemp,bias,prsv,alp_p. The two unit rates are alpha
    and alp_p; iTemp is nonnegative, bias is signed, and prsv is signed unless
    positive_history=True. No independent forgetting rate or target is fitted.

    Starting from Q_L=Q_R=P_R=0, input choice c (0=left,1=right) and reward r:
        Q_c' = (1-alpha)*Q_c + alpha*r
        Q_u' = (1-alpha)*Q_u
        P_R' = (1-alp_p)*P_R + alp_p*(c-.5)
    The pre-choice right logit is iTemp*(Q_R-Q_L+prsv*P_R-bias).
    Relative to local Q_model_config(bias=True,prsv_ema=True,forget=False),
    only unchosen updating changes. Unlike forget=True, the target is zero.

    Exported history/history_post are the actual half-scale P_R, including
    when prsv=0. Reward_EMA_prsvEMA uses twice this trace, so its coefficient
    is prsv/2 for identical policies. ML equivalence requires this conversion;
    unchanged numeric MAP priors are not invariant to the scale change.
    """

    def __init__(self, positive_history=False, param_bounds=None):
        self.positive_history = bool(positive_history)
        self.name = "Q_model_bias_prsvEMA_trial" + ("_positive" if self.positive_history else "")
        names = ["alpha", "iTemp", "bias", "prsv", "alp_p"]
        ranges = ["unit", "pos", "unc", "pos" if self.positive_history else "unc", "unit"]
        _set_parameters(self, names, ranges, param_bounds)

    def forward(self, session, param_values):
        """Use shared likelihood/masks and expose the literal legacy history scale."""
        result = _q_forward(
            self, session, param_values, forgetting="alpha", target=0.0,
            history_mode="ema", history_scale=0.5,
        )
        result["history"] *= 0.5
        result["history_post"] *= 0.5
        return result


class Q_model_bias_prsvEMA_trial_positive(Q_model_bias_prsvEMA_trial):
    """Five-parameter legacy-scale reward EMA with prsv >= 0, as fitted to 11 subjects.

    This applies the upstream single-trial coefficient domain to the local EMA
    extension; it is not an EMA class present in the original upstream notebook.
    """

    def __init__(self, param_bounds=None):
        super().__init__(positive_history=True, param_bounds=param_bounds)


# Matched variants: a full signed history scale and identical single/EMA domains.


class Q_history_model(_Model):
    """Per-visit Q or per-trial reward EMA with an optional signed choice kernel.

    With bias=True, single history has parameters alpha,iTemp,bias,prsv;
    EMA adds alp_p. These are the previously fitted five/four-parameter models.
    learning='trial' makes unchosen forgetting equal alpha with target zero.
    """

    def __init__(self, learning="visit", history="ema", bias=True, param_bounds=None):
        if learning not in ("visit", "trial") or history not in ("none", "single", "ema"):
            raise ValueError("learning must be visit/trial and history none/single/ema")
        self.learning, self.history, self.bias = learning, history, bool(bias)
        self.perseveration = history  # Existing workspace spelling.
        prefix = "Q_visit" if learning == "visit" else "Reward_EMA"
        self.name = prefix + {"none": "", "single": "_prsv", "ema": "_prsvEMA"}[history]
        if learning == "visit" and history == "none":
            self.name = "Q_basic_bias" if bias else "Q_basic"
        elif not bias:
            self.name += "_no_bias"
        names, ranges = ["alpha", "iTemp"], ["unit", "pos"]
        if bias:
            names += ["bias"]
            ranges += ["unc"]
        if history != "none":
            names += ["prsv"]
            ranges += ["unc"]
        if history == "ema":
            names += ["alp_p"]
            ranges += ["unit"]
        _set_parameters(self, names, ranges, param_bounds)

    def forward(self, session, param_values):
        """Return pre/post-input value/history trajectories and fitted probabilities."""
        return _q_forward(
            self, session, param_values, forgetting="alpha" if self.learning == "trial" else "none",
            target=0.0, history_mode=self.history,
        )


class Q_forgetting_model(Q_history_model):
    """Independent unchosen forgetting rate with a fixed0/.5 or free target.

    Extends Q_history_model('visit',history,bias=True) with alp_f, followed by
    decay_target only for target='free'. Full-scale signed history is retained.
    All rates and free targets lie in [0,1]; states start at zero.
    """

    def __init__(self, target="free", history="ema", param_bounds=None):
        if target not in (0.0, 0.5, "free") or history not in ("single", "ema"):
            raise ValueError("target must be0,.5 or 'free'; history must be single/ema")
        super().__init__(learning="visit", history=history, bias=True)
        self.target = target
        label = {0.0: "zero", 0.5: "half", "free": "free"}[target]
        self.name = f"Q_decay_{label}_" + ("prsv" if history == "single" else "prsvEMA")
        names = self.param_names + ["alp_f"]
        ranges = self.param_ranges + ["unit"]
        if target == "free":
            names += ["decay_target"]
            ranges += ["unit"]
        _set_parameters(self, names, ranges, param_bounds)

    def forward(self, session, param_values):
        """Independent learning/forgetting; no rate ordering is imposed."""
        return _q_forward(self, session, param_values, forgetting="free", target=self.target, history_mode=self.history)


# Bayesian models: fitted policy uses belief contrast, NOT expected-reward contrast.


def _bayes_update(b_prev, choice, reward, q, rho_high=0.75, rho_low=0.25, *, asymmetric=False):
    predicted = (1.0 - q) * b_prev + q * (1.0 - b_prev)
    if asymmetric and reward == 0:
        return predicted
    like_right, like_left = (rho_high, rho_low) if choice else (rho_low, rho_high)
    if not reward:
        like_right, like_left = 1.0 - like_right, 1.0 - like_left
    numerator = like_right * predicted
    denominator = numerator + like_left * (1.0 - predicted)
    if asymmetric:
        return numerator / max(denominator, 1e-12)  # Original asymmetric safeguard.
    if denominator <= 0:
        raise ValueError("Observation has zero probability under both latent states")
    return numerator / denominator


class _BayesModel(_Model):
    """Belief P(RightGood), optionally with single/full-scale EMA history."""

    _rho_high_default, _rho_low_default = 0.75, 0.25
    asymmetric = False

    def __init__(self, prsv=True, prsv_ema=False, param_bounds=None):
        if prsv and prsv_ema:
            raise ValueError("Choose single-trial prsv OR prsv_ema, not both")
        self.prsv, self.prsv_ema = bool(prsv), bool(prsv_ema)
        names, ranges = ["beta", "q", "bias_left"], ["pos", "unit", "unc"]
        if self.prsv_ema:
            names += ["prsv", "alp_p"]
            ranges += ["unc", "unit"]
        elif self.prsv:
            names += ["prsv"]
            ranges += ["unc"]
        _set_parameters(self, names, ranges, param_bounds)

    def _env_rhos(self, session):
        return self._rho_high_default, self._rho_low_default

    def forward(self, session, param_values):
        """Single source of truth for the Bayesian likelihood and exported variables."""
        pv = _parameters(self, param_values)
        choices, outcomes, scored = get_trial_data(session)
        rho_high, rho_low = self._env_rhos(session)
        if not (np.isfinite(rho_high) and np.isfinite(rho_low) and 0 <= rho_low <= rho_high <= 1):
            raise ValueError("Reward probabilities require0<=rho_low<=rho_high<=1")
        mode = "ema" if self.prsv_ema else ("single" if self.prsv else "none")
        history, history_post = _choice_history(choices, mode, getattr(pv, "alp_p", 1.0))
        belief_b = np.zeros(len(choices))
        belief_post = np.zeros_like(belief_b)
        belief = 0.5
        for t, (choice, reward) in enumerate(zip(choices, outcomes)):
            belief_b[t] = belief
            belief = _bayes_update(belief, choice, reward, pv.q, rho_high, rho_low, asymmetric=self.asymmetric)
            belief_post[t] = belief
        values = np.vstack([
            rho_low + (rho_high - rho_low) * (1.0 - belief_b),
            rho_low + (rho_high - rho_low) * belief_b,
        ])
        values_post = np.vstack([
            rho_low + (rho_high - rho_low) * (1.0 - belief_post),
            rho_low + (rho_high - rho_low) * belief_post,
        ])
        history_bonus = getattr(pv, "prsv", 0.0) * history
        decision = 2.0 * belief_b - 1.0 + history_bonus - pv.bias_left
        result = _finish_forward(
            session, choices, scored, values, values_post, history, history_post, decision, pv.beta
        )
        result.update(belief_b=belief_b, belief_post=belief_post, history_bonus=history_bonus)
        return result


class BayesSym_2AB(_BayesModel):
    """Symmetric Bayesian inference: both rewards and omissions update belief.

    beta is inverse temperature, q is reversal hazard, bias_left is signed left
    preference. Optional prsv/alp_p use full-scale signed choice history. The
    original fitted policy uses 2*b-1; expected reward difference is not silently
    substituted when exporting probabilities (an error in the old exporter).
    """

    name = "BayesSym_2AB"


class BayesAsym_2AB(_BayesModel):
    """Asymmetric inference: omissions carry hazard propagation but no evidence.

    Optional environment reward probabilities come from session.info when both
    reward_prob_high and reward_prob_low are supplied, as in the original model.
    return_Q_values now returns the actual pre-choice expected values, correcting
    the old export that repeated a final belief. The fitted likelihood is unchanged.
    """

    name = "BayesAsym_2AB"
    asymmetric = True

    def __init__(self, prsv=True, prsv_ema=False, param_bounds=None, read_env_from_session=True):
        super().__init__(prsv=prsv, prsv_ema=prsv_ema, param_bounds=param_bounds)
        self.read_env_from_session = bool(read_env_from_session)

    def _env_rhos(self, session):
        if self.read_env_from_session and hasattr(session, "info"):
            high = getattr(session.info, "reward_prob_high", None)
            low = getattr(session.info, "reward_prob_low", None)
            if isinstance(high, (int, float)) and isinstance(low, (int, float)):
                return float(high), float(low)
        return super()._env_rhos(session)


BayesAsym = BayesAsym_2AB


# Ji-An et al. (2025) drift equations and explicit local controls.
# Source: jil095/tinyRNN a8fc159a2b2d451be6756ae244b29b7d991aac1f,
# agents/RTS_RL_agents_v1/{Model_based_decay,Model_based_mix_decay}.py.
# PRLCogAgent selects MB1 (zero) and MB0md (other), with p_transit=1.

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


# Paper-equation and released-code forgetting-to-mean variants.
# Paper Methods Eq. 10 forgets only the unchosen value. The released MBf
# Model_based_forgetful.py kernel forgets both values after chosen learning.
# The paper's unspecified initial value is tested explicitly as zero or .5;
# released-code initialization is .5. Published equations are written independently.

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


# Named comparison configurations; construction never fits or reads data.

PRIMARY_MODELS = (
    "Q_basic_bias",
    "Q_visit_prsv",
    "Q_visit_prsvEMA",
    "Reward_EMA_prsvEMA",
    "Q_decay_free_prsvEMA",
    "Q_model_bias_prsvEMA_forget",
    "BayesSym_2AB",
    "BayesAsym_2AB",
)

# Additional models are explicit opt-ins; the earlier comparison default is unchanged.
DRIFT_MODELS = (
    "Q_drift_zero",
    "Q_drift_other",
    "Q_drift_zero_matched",
    "Q_drift_other_no_decay",
    "Q_drift_zero_bias_prsvEMA",
    "Q_drift_other_bias_prsvEMA",
    "Q_drift_zero_matched_bias_prsvEMA",
    "Q_drift_other_no_decay_bias_prsvEMA",
    "Q_drift_zero_matched_no_decay",
    "Q_drift_zero_matched_no_decay_bias_prsvEMA",
)


MEAN_MODELS = ("Q_paper_mean_init_half", "Q_paper_mean_init_zero", "Q_release_mean")


def make_model(name, *, param_bounds=None):
    """Construct one explicit model; Bayesian registry entries use EMA history."""
    factories = {
        "Q_basic": lambda: Q_history_model("visit", "none", bias=False, param_bounds=param_bounds),
        "Q_basic_bias": lambda: Q_history_model("visit", "none", bias=True, param_bounds=param_bounds),
        "Q_visit_prsv": lambda: Q_history_model("visit", "single", bias=True, param_bounds=param_bounds),
        "Q_visit_prsvEMA": lambda: Q_history_model("visit", "ema", bias=True, param_bounds=param_bounds),
        "Reward_EMA_prsv": lambda: Q_history_model("trial", "single", bias=True, param_bounds=param_bounds),
        "Reward_EMA_prsvEMA": lambda: Q_history_model("trial", "ema", bias=True, param_bounds=param_bounds),
        "Q_model_bias_prsvEMA_trial": lambda: Q_model_bias_prsvEMA_trial(param_bounds=param_bounds),
        "Q_model_bias_prsvEMA_trial_positive": lambda: Q_model_bias_prsvEMA_trial_positive(
            param_bounds=param_bounds
        ),
        "Q_model_bias_prsvEMA_forget": lambda: Q_model_config(
            bias=True, prsv_ema=True, forget=True, param_bounds=param_bounds
        ),
        "BayesSym_2AB": lambda: BayesSym_2AB(prsv=False, prsv_ema=True, param_bounds=param_bounds),
        "BayesAsym_2AB": lambda: BayesAsym_2AB(prsv=False, prsv_ema=True, param_bounds=param_bounds),
        "Q_paper_mean_init_half": lambda: Q_mean_model("paper", initial_value=.5, param_bounds=param_bounds),
        "Q_paper_mean_init_zero": lambda: Q_mean_model("paper", initial_value=0., param_bounds=param_bounds),
        "Q_release_mean": lambda: Q_mean_model("released", initial_value=.5, param_bounds=param_bounds),
    }
    for target, label in ((0.0, "zero"), (0.5, "half"), ("free", "free")):
        for history, suffix in (("single", "prsv"), ("ema", "prsvEMA")):
            factories[f"Q_decay_{label}_{suffix}"] = (
                lambda target=target, history=history: Q_forgetting_model(
                    target=target, history=history, param_bounds=param_bounds
                )
            )
    for rule in ("zero", "other", "zero_matched"):
        for decaying_bias in ((True,) if rule == "zero" else (True, False)):
            for bias, history in ((False, "none"), (True, "ema")):
                drift_name = f"Q_drift_{rule}"
                if rule != "zero" and not decaying_bias:
                    drift_name += "_no_decay"
                if bias:
                    drift_name += "_bias_prsvEMA"
                factories[drift_name] = (
                    lambda rule=rule, bias=bias, history=history, decaying_bias=decaying_bias: Q_drift_model(
                        rule, bias=bias, history=history, decaying_bias=decaying_bias, param_bounds=param_bounds
                    )
                )
    if name not in factories:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(factories)}")
    model = factories[name]()
    if model.name != name:
        raise RuntimeError(f"Registry name {name!r} disagrees with model.name {model.name!r}")
    return model


def make_models(names=PRIMARY_MODELS):
    """Return distinct models in caller-specified order."""
    names = tuple(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose a nonempty sequence of distinct model names")
    return [make_model(name) for name in names]
