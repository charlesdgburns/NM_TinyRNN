"""
Reversal task simulation with Bayesian and Q-learning agents.

Q-learning fitting supports two modes:
  - grid_search_task(task)      : optimise reward rate via simulation
  - grid_search_behaviour(df)   : fit to trial-by-trial choices via MLE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import product
from scipy.optimize import minimize


# ── Task ──────────────────────────────────────────────────────────────────────

class ReversalTask:
    '''Two-armed bandit reversal task with forced-choice and free-choice trials.
    For usage, initialise task and perform task.step() with action indices:
    0 = initiate, 1 = left poke, 2 = right poke.
    The task has three states: 'initiate', 'free_choice', or 'forced_choice' (left or right).
    On 'initiate', the agent must poke the center to start a trial.
    On 'free_choice', the agent can poke either side; reward probabilities depend on task.good_side (hidden) state.
    On 'forced_choice', the agent must poke the indicated side; any other action does not change the task.state 
        the outcome on forced_choice depends on task.good_side as usual.

    To see code for simulating an agent taking actions see simulate() and run_episode() functions below.
    '''
    def __init__(self, good_side_prob=0.75, bad_side_prob=0.25,
                 ema_tau=8, switch_threshold=0.75, forced_prob=0.25):
        self.good_prob        = good_side_prob
        self.bad_prob         = bad_side_prob
        self.ema_tau          = ema_tau
        self.switch_threshold = switch_threshold
        self.forced_prob      = forced_prob

        self.INITIATE    = np.array([1., 0., 0.])
        self.LEFT        = np.array([0., 1., 0.])
        self.RIGHT       = np.array([0., 0., 1.])
        self.FREE_CHOICE = np.array([0., 1., 1.])

        self.idx_to_action = {0: 'initiate', 1: 'left', 2: 'right'}
        self.reset()

    def reset(self):
        self.good_side   = np.random.choice(['left', 'right'])
        self.state       = 'initiate'
        self.ema_reward  = 0.0
        self.trial_count = 0
        return self.INITIATE.copy()

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        reward = 0.0

        if self.state == 'initiate':
            if action == 'initiate':
                if np.random.rand() < self.forced_prob:
                    self.state = np.random.choice(['left', 'right'])
                    obs = self.LEFT if self.state == 'left' else self.RIGHT
                    return obs.copy(), reward, {}
                else:
                    self.state = 'free_choice'
                    return self.FREE_CHOICE.copy(), reward, {}
            return self.INITIATE.copy(), reward, {}

        elif self.state == 'free_choice':
            if action in ['left', 'right']:
                prob   = self.good_prob if action == self.good_side else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.copy(), reward, {'good_side': self.good_side}
            return self.FREE_CHOICE.copy(), reward, {}

        elif self.state == 'left':
            if action == 'left':
                prob   = self.good_prob if self.good_side == 'left' else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.copy(), reward, {'good_side': self.good_side}
            return self.LEFT.copy(), reward, {}

        elif self.state == 'right':
            if action == 'right':
                prob   = self.good_prob if self.good_side == 'right' else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.copy(), reward, {'good_side': self.good_side}
            return self.RIGHT.copy(), reward, {}

    def _update_and_check_reversal(self, reward):
        self.trial_count += 1
        alpha = 1.0 / self.ema_tau
        self.ema_reward += alpha * (reward - self.ema_reward)
        if self.ema_reward >= self.switch_threshold:
            self.good_side  = 'right' if self.good_side == 'left' else 'left'
            self.ema_reward = 0.0


# ── Agents ────────────────────────────────────────────────────────────────────

class BayesAgent:
    """
    Latent-state Bayesian agent.
    Belief: p_A = P(Left is good).
    Update: Bayes likelihood (Ji-An eq. 5) then transition smear (Ji-An eq. 6).
    Policy: deterministic argmax.
    """
    def __init__(self, good_prob=0.75, bad_prob=0.25, p_switch=0.05):
        self.p_good   = good_prob
        self.p_bad    = bad_prob
        self.p_switch = p_switch
        self.p_A      = 0.5

    def reset(self):
        self.p_A = 0.5

    def choose(self, forced_side=None):
        if forced_side is not None:
            return 1 if forced_side == 'left' else 2
        return 1 if self.p_A >= 0.5 else 2

    def update(self, action_idx, reward):
        chose_left = (action_idx == 1)
        if chose_left:
            p_r_h1 = self.p_good if reward == 1 else (1 - self.p_good)
            p_r_h2 = self.p_bad  if reward == 1 else (1 - self.p_bad)
        else:
            p_r_h1 = self.p_bad  if reward == 1 else (1 - self.p_bad)
            p_r_h2 = self.p_good if reward == 1 else (1 - self.p_good)

        numer   = p_r_h1 * self.p_A
        denom   = numer + p_r_h2 * (1 - self.p_A)
        p_bar_A = numer / denom if denom > 0 else self.p_A

        pr       = self.p_switch
        self.p_A = (1 - pr) * p_bar_A + pr * (1 - p_bar_A)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_episode(task, agent, n_trials=500):
    """Run one episode; return list of trial dicts."""
    task.reset()
    agent.reset()

    records = []
    while len(records) < n_trials:
        obs, _, _ = task.step(0)           # initiate

        if   obs[1] == 1 and obs[2] == 0: forced, forced_side = True,  'left'
        elif obs[2] == 1 and obs[1] == 0: forced, forced_side = True,  'right'
        else:                              forced, forced_side = False, None

        action_idx = agent.choose(forced_side)

        good_side_before   = task.good_side
        obs, reward, info  = task.step(action_idx)
        agent.update(action_idx, reward)

        p_A   = getattr(agent, 'p_A', None)
        p_B   = 1.0 - p_A if p_A is not None else None
        eps   = 1e-12
        logit = np.log((p_B + eps) / (p_A + eps)) if p_A is not None else np.nan

        choice     = action_idx - 1
        arm_label  = 'A1' if choice == 0 else 'A2'
        trial_type = f'{arm_label},R={int(reward)}'

        records.append({
            'forced_choice': int(forced),
            'choice'       : choice,
            'outcome'      : int(reward),
            'good_poke'    : good_side_before,
            'prob_A'       : p_A,
            'prob_B'       : p_B,
            'logit_value'  : logit,
            'trial_type'   : trial_type,
        })
    return records


def simulate(agent_class=BayesAgent, agent_kwargs=None, n_episodes=10,
             n_trials=500, seed=42, **task_kwargs):
    np.random.seed(seed)
    agent_kwargs = agent_kwargs or {}
    p_switch     = task_kwargs.pop('p_switch', 0.05)

    task  = ReversalTask(**task_kwargs)
    agent = agent_class(
        **(dict(good_prob=task.good_prob, bad_prob=task.bad_prob, p_switch=p_switch)
           if agent_class is BayesAgent else agent_kwargs)
    )

    all_records = []
    for ep in range(n_episodes):
        records = run_episode(task, agent, n_trials)
        for t, r in enumerate(records):
            r['episode'] = ep
            r['trial']   = t
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df['logit_past']   = df.groupby('episode')['logit_value'].shift(1)
    df['logit_change'] = df['logit_value'] - df['logit_past']

    cols = ['episode', 'trial', 'forced_choice', 'choice', 'outcome',
            'good_poke', 'prob_A', 'prob_B',
            'logit_value', 'logit_past', 'logit_change', 'trial_type']
    return df[cols]


# ── Q-learning fitting ────────────────────────────────────────────────────────

def _softmax_probs(Q, beta):
    logits = beta * Q - beta * Q.max()
    e      = np.exp(logits)
    return e / e.sum()


def _nll_from_trials(alpha, beta, choices, rewards, forced_mask):
    """Negative log-likelihood of choices under Q-learning model."""
    Q, nll = np.zeros(2), 0.0
    for choice, reward, forced in zip(choices, rewards, forced_mask):
        if not forced:
            nll -= np.log(_softmax_probs(Q, beta)[choice] + 1e-12)
        Q[choice] += alpha * (reward - Q[choice])    # update on all trials
    return nll


def _predict_probs_from_trials(alpha, beta, choices, rewards, forced_mask):
    """Return (p_left, p_right) at each trial given fitted params."""
    Q, probs = np.zeros(2), []
    for choice, reward, forced in zip(choices, rewards, forced_mask):
        p = _softmax_probs(Q, beta)
        probs.append(p)
        Q[choice] += alpha * (reward - Q[choice])
    return np.array(probs)



def grid_search_behaviour(df, alphas=None, betas=None, multi_start=True):
    """
    Fit (alpha, beta) to observed choices via MLE, then return df with
    predicted probabilities appended.

    Parameters
    ----------
    df     : DataFrame with columns [episode, choice, outcome, forced_choice]
    alphas : initial alpha values for grid search / multi-start
    betas  : initial beta  values for grid search / multi-start

    Returns
    -------
    best   : dict with alpha, beta, nll
    pred_df: input df with columns [p_left, p_right] appended
    """
    alphas = alphas if alphas is not None else np.linspace(0.05, 0.6,  8)
    betas  = betas  if betas  is not None else np.linspace(1.,  15.,   8)

    choices = df['choice'].values.astype(int)
    rewards = df['outcome'].values.astype(int)
    forced  = df['forced_choice'].values.astype(bool)

    def neg_ll(params):
        a, b = params
        return 1e10 if not (0 < a < 1 and b > 0) else \
               _nll_from_trials(a, b, choices, rewards, forced)

    starts = [(a, b) for a, b in product(alphas, betas)] if multi_start \
             else [(alphas[len(alphas)//2], betas[len(betas)//2])]

    best_res = None
    for a0, b0 in starts:
        res = minimize(neg_ll, [a0, b0], method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-4})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    alpha_hat, beta_hat = best_res.x
    best = {'alpha': alpha_hat, 'beta': beta_hat, 'nll': best_res.fun}

    probs   = _predict_probs_from_trials(alpha_hat, beta_hat, choices, rewards, forced)
    pred_df = df.copy()
    pred_df['p_left']  = probs[:, 0]
    pred_df['p_right'] = probs[:, 1]

    return best, pred_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(df, episode=0):
    ep     = df[df['episode'] == episode].reset_index(drop=True)
    trials = ep['trial']

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Bayesian Reversal Task — Episode {episode}', fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(4, 1, hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(trials, ep['prob_B'], color='steelblue', lw=1.5, label='P(Right good)')
    ax1.plot(trials, ep['prob_A'], color='tomato',    lw=1.5, label='P(Left good)')
    ax1.axhline(0.5, color='grey', lw=0.8, ls='--')
    ax1.set_ylabel('Belief probability'); ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc='upper right'); ax1.set_title('Posterior belief state')
    _shade_reversals(ax1, ep)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(trials, ep['logit_value'], color='darkorchid', lw=1.5)
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.set_ylabel('Log-odds (right good)'); ax2.set_title('Logit belief value')
    _shade_reversals(ax2, ep)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    colors = ep['logit_change'].apply(lambda x: 'seagreen' if x >= 0 else 'tomato')
    ax3.bar(trials, ep['logit_change'], color=colors, width=0.8)
    ax3.axhline(0, color='grey', lw=0.8, ls='--')
    ax3.set_ylabel('Δ logit'); ax3.set_title('Trial-by-trial logit change')
    _shade_reversals(ax3, ep)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    rew  = ep['outcome'] == 1; unrew = ep['outcome'] == 0; forc = ep['forced_choice'] == 1
    ax4.scatter(trials[rew   & ~forc], ep['choice'][rew   & ~forc] + 0.02,
                marker='o', s=18, color='steelblue', label='Free, rewarded',   zorder=3)
    ax4.scatter(trials[unrew & ~forc], ep['choice'][unrew & ~forc] - 0.02,
                marker='x', s=18, color='tomato',    label='Free, unrewarded', zorder=3)
    ax4.scatter(trials[forc],          ep['choice'][forc],
                marker='D', s=14, color='grey', alpha=0.6, label='Forced',     zorder=2)
    ax4.set_yticks([0, 1]); ax4.set_yticklabels(['Left (0)', 'Right (1)'])
    ax4.set_xlabel('Trial'); ax4.set_title('Choices and outcomes')
    ax4.legend(fontsize=7, loc='upper right', ncol=3)
    _shade_reversals(ax4, ep)

    plt.savefig('reversal_task_plot.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_grid_search(grid, title='Q-agent reward rate'):
    pivot = grid.pivot(index='alpha', columns='beta', values='reward_rate')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title(title); plt.tight_layout()
    plt.savefig('grid_search.png', dpi=150, bbox_inches='tight')
    plt.show()


def _shade_reversals(ax, ep):
    good  = ep['good_poke'].values
    start = None
    for i, val in enumerate(good == 'right'):
        if val and start is None:          start = i
        elif not val and start is not None:
            ax.axvspan(start, i, alpha=0.08, color='steelblue'); start = None
    if start is not None:
        ax.axvspan(start, len(good), alpha=0.08, color='steelblue')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── 1. Simulate Bayes agent ───────────────────────────────────────────────
    print('Simulating Bayes agent...')
    df = simulate(n_episodes=20, n_trials=300)
    print(f'Total trials: {len(df)}  |  Reward rate: {df["outcome"].mean():.3f}')

    save_path = Path('./NM_TinyRNN/data/AB_behaviour/bayes_optimal/simulated/')
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / 'trials.hsv', index=False, sep='\t')

    plot_results(df, episode=0)
