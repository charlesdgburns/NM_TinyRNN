"""
Reversal task simulation with Bayesian and Q-learning agents.

Q-learning fitting supports two modes:
  - fit_agent_to_task(task_class, model_cls)   : optimise reward rate via simulation
  - fit_agent_to_behaviour(df, model_cls)      : fit to trial-by-trial choices via MLE

1D Q-learning option available through `QLearningAgent(n=1)`.

Speed optimizations:
  - Use multi_start=False for fast fitting (single optimization run)
  - L-BFGS-B optimizer with bounds for better convergence
  - Optimized softmax computation
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import product
from scipy.optimize import minimize
from NM_TinyRNN.code.simulations.reversal_2ab import ReversalTask, run_episode, simulate, plot_results


# ── Agent ────────────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Q-learner for 2-arm reversal task.

    Supports two dimensionalities:
      - n=2: standard independent Q-learning
      - n=1: anti-correlated update of the unchosen action

    Policy: softmax with inverse temperature beta.
    Forced sides obeyed; outcomes still update Q regardless.
    """
    def __init__(self, alpha=0.2, beta=5.0, n=2):
        self.alpha = alpha   # learning rate
        self.beta  = beta    # softmax inverse temperature
        if n not in (1, 2):
            raise ValueError('n must be 1 or 2')
        self.n = n
        self.Q = np.zeros(2)   # Q[0]=left, Q[1]=right

    def reset(self):
        self.Q = np.zeros(2)

    def choose(self, forced_side=None):
        if forced_side is not None:
            return 1 if forced_side == 'left' else 2
        logits = self.beta * self.Q
        logits -= logits.max()
        probs   = np.exp(logits) / np.exp(logits).sum()
        return int(np.random.choice([1, 2], p=probs))

    def update(self, action_idx, reward):
        a = action_idx - 1                            # map 1/2 → 0/1
        if self.n == 2:
            self.Q[a] += self.alpha * (reward - self.Q[a])
        else:
            other = 1 - a
            self.Q[a] += self.alpha * (reward - self.Q[a])
            self.Q[other] -= self.alpha * (reward + self.Q[other])


# ── Q-learning fitting ───────────────────────────────────────────────────────

def _softmax_probs(Q, beta):
    """Fast softmax computation for 2-element Q array."""
    q0, q1 = Q[0] * beta, Q[1] * beta
    max_q = max(q0, q1)
    e0 = np.exp(q0 - max_q)
    e1 = np.exp(q1 - max_q)
    total = e0 + e1
    return np.array([e0 / total, e1 / total])


def _nll_from_trials(alpha, beta, choices, rewards, forced_mask, n=2):
    """Negative log-likelihood of choices under Q-learning model."""
    Q, nll = np.zeros(2), 0.0
    for choice, reward, forced in zip(choices, rewards, forced_mask):
        if not forced:
            nll -= np.log(_softmax_probs(Q, beta)[choice] + 1e-12)
        if n == 2:
            Q[choice] += alpha * (reward - Q[choice])
        else:
            other = 1 - choice
            Q[choice] += alpha * (reward - Q[choice])
            Q[other] -= alpha * (reward + Q[other])
    return nll


def _predict_probs(alpha, beta, choices, rewards, forced_mask, n=2):
    """Return post-update (p_left, p_right) at each trial for Q-learning."""
    Q, probs = np.zeros(2), []
    for choice, reward, _ in zip(choices, rewards, forced_mask):
        if n == 2:
            Q[choice] += alpha * (reward - Q[choice])
        else:
            other = 1 - choice
            Q[choice] += alpha * (reward - Q[choice])
            Q[other] -= alpha * (reward + Q[other])
        probs.append(_softmax_probs(Q, beta))
    return np.array(probs)

def fit_agent_to_task(task_class, model_cls, task_kwargs=None, model_kwargs=None,
                      alphas=None, betas=None,
                      n_episodes=10, n_trials=300, seed=0):
    """
    Optimise (alpha, beta) for reward rate using a task simulator.

    Parameters
    ----------
    task_class  : class with reset() / step() interface
    model_cls   : agent class to instantiate for each search point
    task_kwargs : dict passed to task_class constructor
    model_kwargs: dict passed to model_cls constructor
    alphas, betas : arrays of values to search (defaults provided)
    n_episodes, n_trials, seed : simulation settings

    Returns
    -------
    best   : dict with alpha, beta, reward_rate
    grid   : DataFrame with full grid results
    """
    task_kwargs  = task_kwargs or {}
    model_kwargs = model_kwargs or {}
    alphas       = alphas if alphas is not None else np.linspace(0.05, 1.0, 10)
    betas        = betas  if betas is not None else np.linspace(1.,  15.,  15)

    rows = []
    for alpha, beta in product(alphas, betas):
        reward_rates = []
        for _ in range(n_episodes):
            task  = task_class(**task_kwargs)
            agent = model_cls(alpha=alpha, beta=beta, **model_kwargs)
            outcomes = [r['outcome'] for r in run_episode(task, agent, n_trials)]
            reward_rates.append(np.mean(outcomes))
        rows.append({'alpha': alpha, 'beta': beta, 'reward_rate': np.mean(reward_rates)})

    grid = pd.DataFrame(rows)
    best = grid.loc[grid['reward_rate'].idxmax()].to_dict()
    return best, grid


def fit_agent_to_behaviour(df, model_cls, model_kwargs=None,
                           alphas=None, betas=None, multi_start=False):
    """
    Fit (alpha, beta) to observed choices via MLE and return predicted choice probabilities.

    Parameters
    ----------
    df      : DataFrame with columns [episode, choice, outcome, forced_choice]
    model_cls : agent class used to determine dimensionality
    model_kwargs : dict passed to model_cls constructor
    alphas, betas : arrays of values to search (defaults provided)
    multi_start : if True, try multiple starting points; if False, use single best guess

    Returns
    -------
    best   : dict with alpha, beta, nll
    pred_df: input df with columns [p_left, p_right] appended
    """
    model_kwargs = model_kwargs or {}
    n = model_kwargs.get('n', 2)

    alphas = alphas if alphas is not None else np.linspace(0.05, 1.0,  10)
    betas  = betas  if betas is not None else np.linspace(1.,  15.,   15)

    choices = df['choice'].values.astype(int)
    rewards = df['outcome'].values.astype(int)
    forced  = df['forced_choice'].values.astype(bool)

    def neg_ll(params):
        a, b = params
        if not (0 < a < 1 and b > 0):
            return 1e10
        return _nll_from_trials(a, b, choices, rewards, forced, n=n)

    if multi_start:
        starts = [(a, b) for a, b in product(alphas, betas)]
    else:
        starts = [(alphas[len(alphas)//2], betas[len(betas)//2])]

    best_res = None
    for a0, b0 in starts:
        res = minimize(neg_ll, [a0, b0], method='L-BFGS-B',
                       bounds=[(0.01, 0.99), (0.1, 50.0)],
                       options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 100})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    alpha_hat, beta_hat = best_res.x
    best = {'alpha': alpha_hat, 'beta': beta_hat, 'nll': best_res.fun}

    probs   = _predict_probs(alpha_hat, beta_hat, choices, rewards, forced, n=n)
    pred_df = df.copy()
    pred_df['p_left']      = probs[:, 0]
    pred_df['p_right']     = probs[:, 1]

    eps = 1e-12
    pred_df['prob_A']      = pred_df['p_left']
    pred_df['prob_B']      = pred_df['p_right']
    pred_df['logit_value'] = np.log((pred_df['p_right'] + eps) / (pred_df['p_left'] + eps))
    pred_df['logit_change'] = pred_df.groupby('episode')['logit_value'].diff()
    pred_df['logit_past']   = pred_df.groupby('episode')['logit_value'].shift(1)

    return best, pred_df


def grid_search_task(task_class=ReversalTask, task_kwargs=None,
                     alphas=None, betas=None,
                     n_episodes=10, n_trials=300, seed=0):
    return fit_agent_to_task(task_class, QLearningAgent,
                             task_kwargs=task_kwargs,
                             model_kwargs={'n': 2},
                             alphas=alphas, betas=betas,
                             n_episodes=n_episodes, n_trials=n_trials, seed=seed)


def grid_search_behaviour(df, alphas=None, betas=None, multi_start=False):
    return fit_agent_to_behaviour(df, QLearningAgent,
                                  model_kwargs={'n': 2},
                                  alphas=alphas, betas=betas,
                                  multi_start=multi_start)


def grid_search_task_1d(task_class=ReversalTask, task_kwargs=None,
                        alphas=None, betas=None,
                        n_episodes=10, n_trials=300, seed=0, n=1):
    return fit_agent_to_task(task_class, QLearningAgent,
                             task_kwargs=task_kwargs,
                             model_kwargs={'n': n},
                             alphas=alphas, betas=betas,
                             n_episodes=n_episodes, n_trials=n_trials, seed=seed)


def grid_search_behaviour_1d(df, alphas=None, betas=None, multi_start=False, n=1):
    return fit_agent_to_behaviour(df, QLearningAgent,
                                  model_kwargs={'n': n},
                                  alphas=alphas, betas=betas,
                                  multi_start=multi_start)


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
    try:
        print('loading existing Bayes agent data...')
        from pathlib import Path
        DATA_PATH = Path('/ceph/behrens/wsilver/reversal/code/NM_TinyRNN/data')
        df = pd.read_csv(DATA_PATH / 'AB_behaviour/bayes_optimal/episode_9/trials.htsv', sep='\t')
        all_episode_dfs = []
        for each_episode in range(10):
            savepath = DATA_PATH/'AB_behaviour/bayes_optimal'/f'episode_{each_episode}'
            episode_df = pd.read_csv(savepath/'trials.htsv', sep='\t')
            all_episode_dfs.append(episode_df)
        df = pd.concat(all_episode_dfs, ignore_index=True)
    except:
        print('Simulating Bayes agent...')
        df = simulate(n_episodes=10, n_trials=500)
        for each_episode in df.groupby('episode'):
            savepath = DATA_PATH/'AB_behaviour/bayes_optimal'/f'episode_{each_episode[0]}'
            savepath.mkdir(exist_ok=True,parents=True)
            each_episode[1].to_csv(savepath/'trials.htsv', index=False, sep='\t')

    print(f'Total trials: {len(df)}  |  Reward rate: {df["outcome"].mean():.3f}')

    # ── 2. Fit Q-agent to Bayes agent's choices (MLE) ─────────────────────────
    print('\nFitting Q-agents to Bayes choices...')
    for d in [1, 2]:
        best_fit, pred_df = fit_agent_to_behaviour(df, QLearningAgent,
                                model_kwargs={'n': d},
                                alphas=None, betas=None,
                                multi_start=True)
        for each_episode in pred_df.groupby('episode'):
            savepath = DATA_PATH/f'AB_behaviour/q_learning_{d}D/episode_{each_episode[0]}'
            savepath.mkdir(exist_ok=True,parents=True)
            each_episode[1].to_csv(savepath/'trials.htsv', index=False, sep='\t')#
        print(f'Best fit params: alpha={best_fit["alpha"]:.3f}, '
            f'beta={best_fit["beta"]:.3f}, NLL={best_fit["nll"]:.1f}')
        sns.scatterplot(pred_df.query('forced_choice==0'), 
                    x= 'logit_value',
                    y='logit_change',
                    hue='trial_type')
        plt.title(f'{d}D Q-learning')
        plt.show()

        free = pred_df[pred_df['forced_choice'] == 0]
        p_chosen = np.where(free['choice'] == 0, free['p_left'], free['p_right'])
        nll      = -np.log(p_chosen + 1e-12).mean()   # cross-entropy per trial
        print(f'Free-choice NLL (cross-entropy): {nll:.4f}  (chance = {np.log(2):.4f})')
        print(pred_df[['episode', 'trial', 'choice', 'outcome',
                    'p_left', 'p_right']].head(10).to_string(index=False))
        plot_results(pred_df, episode=0)
