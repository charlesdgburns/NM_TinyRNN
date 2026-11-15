"""In the following script we consider RL models which can be instantiated as 2-unit monoGRUs."""

import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy import stats  # Full scipy submodule

# Import helper functions from the same directory
from model_implementation_helper_functions import get_trial_data, softmax_probs, log_safe


# Define RL model class.

#%%  Q_model - Simple Q-learning using RPEs (Takam)

class Q_model:
    """A simple RL model which updates Q values for the two options using reward prediction errors."""

    def __init__(self, param_bounds=None):
        self.name = "Q_model"
        self.param_names = ["alpha", "iTemp"]
        self.param_ranges = ["unit", "pos"]
        self.n_params = 2
        if param_bounds:
            self.param_bounds = param_bounds

    def session_log_likelihood(self, session, param_values):
        """Return the data likelihood for the session given the provided parameter values."""

        # Get trial events.
        choices, outcomes, free_choice = get_trial_data(session)

        # Unpack parameters.
        alpha, iTemp = param_values

        # Variables.
        Q = np.zeros([2, session.n_trials])  # Q values for left and right choices.

        for i, (c, o) in enumerate(zip(choices[:-1], outcomes)):  # loop over trials.
            # Update action values.
            n = 1 - c  # Not chosen action.

            Q[n, i + 1] = Q[n, i]  # Not chosen action value is unchanged.
            Q[c, i + 1] = (1.0 - alpha) * Q[c, i] + alpha * o  # TD update.

        # Compute data likelihood.
        choice_probs = softmax_probs(Q, iTemp)
        trial_log_likelihood = log_safe(choice_probs[choices, np.arange(len(choices))])
        session_log_likelihood = np.sum(trial_log_likelihood[free_choice])

        return session_log_likelihood



class Q_model_forget:
    """Modified version of Q_model which incorporates forgetting about value of not chosen option implemented
    as decay towards neutral value of 0.5 with forgetting rate parameter alp_f."""
    
    name = "Q_model_forget"
    param_names = ["alpha", "iTemp", "alp_f"]  # Make this a class attribute
    param_ranges = ["unit", "pos", "unit"]     # Make this a class attribute
    n_params = 3                               # Make this a class attribute

    def __init__(self, param_bounds=None):
        if param_bounds:
            self.param_bounds = param_bounds

    def session_log_likelihood(self, session, param_values):
        """Return the data likelihood for the session given the provided parameter values."""

        # Get trial events.
        choices, outcomes, free_choice = get_trial_data(session)

        # Unpack parameters.
        alpha, iTemp, alp_f = param_values

        # Variables.
        Q = np.zeros([2, session.n_trials])  # Q values for left and right choices.

        for i, (c, o) in enumerate(zip(choices[:-1], outcomes)):  # loop over trials.
            # Update action values.
            n = 1 - c  # Not chosen action.

            Q[n, i + 1] = (1.0 - alp_f) * Q[n, i] + alp_f * 0.5  # Not chosen action value decays towards 0.5
            Q[c, i + 1] = (1.0 - alpha) * Q[c, i] + alpha * o  # TD update.

        # Compute data likelihood.
        choice_probs = softmax_probs(Q, iTemp)
        trial_log_likelihood = log_safe(choice_probs[choices, np.arange(len(choices))])
        session_log_likelihood = np.sum(trial_log_likelihood[free_choice])

        return session_log_likelihood


#setup
class BayesAsym_2AB:
    “”"
    Asymmetric Bayesian inference for the 2AB task:

      • Hidden state: z ∈ {LeftGood, RightGood}; belief b_t = P(z=RightGood | history)
      • Reversal step identical to symmetric: b_pred = (1-q)*b + q*(1-b)
      • Observation model is ASYMMETRIC:
          - Rewards are side-specific evidence (reward on Left vs Right)
          - Omissions are treated as the SAME observation under both hidden states
            (no information) -> posterior = predicted prior
      • Policy in VALUE UNITS (to match your Takam variants):
      logit_t = beta * ( (2*b_t - 1) + pers_term - bias_left )
      with pers_term determined by the selected stickiness mode:
        - prsv (single‑lag): prsv * prev
        - prsv_ema:         prsv * P_t, with P_t = (1-alp_p) P_{t-1} + alp_p * prev
    • Likelihood computed exactly like your other models.
    “”"
    name = “BayesAsym_2AB”
    # Class-level defaults (instance overrides in __init__)
    param_names  = [“beta”, “prsv”, “q”, “bias_left”]
    param_ranges = [“pos”,  “unc”,  “unit”, “unc”]
    n_params     = 4

    _rho_high_default = 0.75
    _rho_low_default  = 0.25

    def __init__(self, prsv: bool = True, prsv_ema: bool = False, param_bounds=None, read_env_from_session: bool = True):
        “”"Expose mutually exclusive perseveration toggles like Q_model_config.
        By default prsv=True, prsv_ema=False (single‑lag). Set prsv_ema=True to use EMA
        stickiness; the two cannot be True simultaneously.
        “”"
        if prsv and prsv_ema:
            raise ValueError(“BayesAsym_2AB: prsv and prsv_ema cannot both be True”)
        self.read_env_from_session = read_env_from_session
        self.prsv = bool(prsv)
        self.prsv_ema = bool(prsv_ema)

        # Base params (no perseveration). Add selected stickiness mode.
        self.param_names  = [“beta”, “q”, “bias_left”]
        self.param_ranges = [“pos”,  “unit”, “unc”]
        if self.prsv_ema:
            self.param_names  += [“prsv”, “alp_p”]
            self.param_ranges += [“unc”,  “unit”]
        elif self.prsv:
            self.param_names  += [“prsv”]
            self.param_ranges += [“unc”]
        self.n_params = len(self.param_names)

        if param_bounds:
            self.param_bounds = param_bounds

    def _env_rhos(self, session):
        “”"Optionally read reward probs from session.info; fall back to defaults.“”"
        if self.read_env_from_session and hasattr(session, “info”):
            rh = getattr(session.info, “reward_prob_high”, None)
            rl = getattr(session.info, “reward_prob_low”,  None)
            if isinstance(rh, (int, float)) and isinstance(rl, (int, float)):
                return float(rh), float(rl)
        return self._rho_high_default, self._rho_low_default

    def session_log_likelihood(self, session, param_values, return_Q_values=False):
        “”"Return the data likelihood for the session given the provided parameter values.“”"
        # Unpack according to selected stickiness mode
        if self.prsv_ema:
            beta, q, bias_left, prsv, alp_p = param_values
        elif self.prsv:
            beta, q, bias_left, prsv = param_values
        else:
            beta, q, bias_left = param_values
        rho_high, rho_low = self._env_rhos(session)

        # Get trial events.
        choices, outcomes, free_choice = get_trial_data(session)

        # Variables.
        T = session.n_trials
        b = 0.5  # P(RightGood) prior
        logits = np.zeros(T, dtype=float)

        P_val = 0.0  # EMA state (used only if prsv_ema)

        # Trial loop.
        for t, (c, r) in enumerate(zip(choices, outcomes)):
            # Decision variable from current belief.
            D = 2.0 * b - 1.0

            # Perseveration regressor: symmetric ±1 previous choice (all trials).
            prev = (2 * choices[t-1] - 1) if t > 0 else 0.0
            # If you prefer perseveration from free-only prev trials, use:
            # prev = 0.0 if (t == 0 or not free_choice[t-1]) else (2*choices[t-1] - 1)

            # Choose between single-lag (original) or EMA stickiness
            if self.prsv_ema:
                P_val = (1.0 - alp_p) * P_val + alp_p * prev
                pers_component = prsv * P_val
            elif self.prsv:
                pers_component = prsv * prev
            else:
                pers_component = 0.0

            # Value-unit policy
            logits[t] = beta * ( D + pers_component - bias_left )

            # --- ASYMMETRIC belief update ---
            b = _bayes_update_asym_2ab(
                    b_prev=b, choice=int(c), reward=int(r), q=q,
                    rho_high=rho_high, rho_low=rho_low
                )

        # Likelihood
        Q_like = np.vstack([np.zeros_like(logits), logits])
        choice_probs = softmax_probs(Q_like, iTemp=1.0)
        trial_log_likelihood = log_safe(choice_probs[choices, np.arange(T)])
        session_log_likelihood = np.sum(trial_log_likelihood[free_choice])

        if return_Q_values:
            Q_pseudo = np.vstack([-(2.0*np.array([b]).repeat(T) - 1.0),
                                   (2.0*np.array([b]).repeat(T) - 1.0)])
            return Q_pseudo
        else:
            return float(session_log_likelihood)