'''Tiny RNNs can also be used as generative models for behaviour.
We may simulate the reversal task and ask a TinyRNN to generate actions.'''

import numpy as np
import pandas as pd
import torch
import seaborn as sns

# ── Task ──────────────────────────────────────────────────────────────────────

class ReversalTask:
    def __init__(self, good_side_prob=0.75, bad_side_prob=0.25,
                 ema_tau=8, switch_threshold=0.75, forced_prob=0.1):
        self.good_prob       = good_side_prob
        self.bad_prob        = bad_side_prob
        self.ema_tau         = ema_tau
        self.switch_threshold = switch_threshold
        self.forced_prob     = forced_prob

        self.INITIATE    = torch.tensor([1., 0., 0.])
        self.LEFT        = torch.tensor([0., 1., 0.])
        self.RIGHT       = torch.tensor([0., 0., 1.])
        self.FREE_CHOICE = torch.tensor([0., 1., 1.])

        self.idx_to_action = {0: 'initiate', 1: 'left', 2: 'right'}
        self.reset()

    def reset(self):
        self.good_side  = np.random.choice(['left', 'right'])
        self.state      = 'initiate'
        self.ema_reward = 0.0
        self.trial_count = 0
        return self.INITIATE.clone()

    def step(self, action_idx):
        action = self.idx_to_action[action_idx]
        reward = 0.0

        if self.state == 'initiate':
            if action == 'initiate':
                if np.random.rand() < self.forced_prob:
                    self.state = np.random.choice(['left', 'right'])
                    obs = self.LEFT if self.state == 'left' else self.RIGHT
                    return obs.clone(), reward, {}
                else:
                    self.state = 'free_choice'
                    return self.FREE_CHOICE.clone(), reward, {}
            return self.INITIATE.clone(), reward, {}

        elif self.state == 'free_choice':
            if action in ['left', 'right']:
                prob   = self.good_prob if action == self.good_side else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.clone(), reward, {'good_side': self.good_side}
            return self.FREE_CHOICE.clone(), reward, {}

        elif self.state == 'left':
            if action == 'left':
                prob   = self.good_prob if self.good_side == 'left' else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.clone(), reward, {'good_side': self.good_side}
            return self.LEFT.clone(), reward, {}

        elif self.state == 'right':
            if action == 'right':
                prob   = self.good_prob if self.good_side == 'right' else self.bad_prob
                reward = 1.0 if np.random.rand() < prob else 0.0
                self._update_and_check_reversal(reward)
                self.state = 'initiate'
                return self.INITIATE.clone(), reward, {'good_side': self.good_side}
            return self.RIGHT.clone(), reward, {}

    def _update_and_check_reversal(self, reward):
        self.trial_count += 1
        alpha = 1.0 / self.ema_tau
        self.ema_reward += alpha * (reward - self.ema_reward)
        if self.ema_reward >= self.switch_threshold:
            self.good_side  = 'right' if self.good_side == 'left' else 'left'
            self.ema_reward = 0.0


# ── Bayesian agent ────────────────────────────────────────────────────────────

class BayesAgent:
    """
    Latent-state Bayesian agent (d=1 model from the paper).

    Belief state: p_A = Pr(h=1), i.e. P(Left/action-A is currently good).

    Each trial does two steps in order:
      1. Likelihood update  — standard Bayes' rule given (action, reward).
      2. Transition smear   — accounts for the possibility that the latent
                              state flipped (reversal probability p_r), per
                              equation (6) in the Ji-An et al. 2025:
                                Pr_t(h=1) = (1-p_r)*Pr̄_t(h=1) + p_r*(1-Pr̄_t(h=1))

    Deterministic policy: choose the arm with higher posterior belief.
    """
    def __init__(self, good_prob=0.75, bad_prob=0.25, p_switch=0.05):
        self.p_good   = good_prob   # P(reward | chosen arm is good)
        self.p_bad    = bad_prob    # P(reward | chosen arm is bad)
        self.p_switch = p_switch    # p_r: per-trial reversal probability
        self.p_A      = 0.5         # prior: P(Left is good)

    def reset(self):
        self.p_A = 0.5

    def choose(self, forced_side=None):
        """Return action index (1=left, 2=right). Forced sides are obeyed."""
        if forced_side is not None:
            return 1 if forced_side == 'left' else 2
        return 1 if self.p_A >= 0.5 else 2

    def update(self, action_idx, reward):
        """
        Step 1 — Bayesian likelihood update (eq. 5):
          p̄_A = P(r | h=1, s) * p_A  /  P(r)
        Step 2 — Transition smear (eq. 6):
          p_A = (1 - p_r) * p̄_A  +  p_r * (1 - p̄_A)
        """
        chose_left = (action_idx == 1)

        # Likelihoods P(reward | state)
        if chose_left:
            p_r_given_h1 = self.p_good if reward == 1 else (1 - self.p_good)
            p_r_given_h2 = self.p_bad  if reward == 1 else (1 - self.p_bad)
        else:
            p_r_given_h1 = self.p_bad  if reward == 1 else (1 - self.p_bad)
            p_r_given_h2 = self.p_good if reward == 1 else (1 - self.p_good)

        # Step 1: posterior p̄_A  (eq. 5)
        numer   = p_r_given_h1 * self.p_A
        denom   = numer + p_r_given_h2 * (1 - self.p_A)
        p_bar_A = numer / denom if denom > 0 else self.p_A

        # Step 2: transition smear  (eq. 6)
        pr      = self.p_switch
        self.p_A = (1 - pr) * p_bar_A + pr * (1 - p_bar_A)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_episode(task, agent, n_trials=200):
    """Run one episode; return list of trial dicts."""
    task.reset()
    agent.reset()
    obs = task.INITIATE.clone()

    records = []

    while len(records) < n_trials:
        # ── Step 1: initiate ──────────────────────────────────────────────────
        obs, _, _ = task.step(0)          # action 0 = initiate

        # ── Step 2: determine forced vs free, then choose ─────────────────────
        state_vec = obs.numpy()

        if state_vec[1] == 1 and state_vec[2] == 0:   # LEFT forced
            forced      = True
            forced_side = 'left'
        elif state_vec[2] == 1 and state_vec[1] == 0: # RIGHT forced
            forced      = True
            forced_side = 'right'
        else:                                           # FREE_CHOICE
            forced      = False
            forced_side = None

        action_idx = agent.choose(forced_side)

        # ── Step 3: execute choice ────────────────────────────────────────────
        good_side_before = task.good_side
        obs, reward, info = task.step(action_idx)

        # ── Step 4: Bayesian update ───────────────────────────────────────────
        agent.update(action_idx, reward)

        p_A = agent.p_A                        # P(Left is good)  — posterior
        p_B = 1.0 - p_A                        # P(Right is good)

        eps = 1e-12
        logit = np.log((p_B + eps) / (p_A + eps))   # log-odds that RIGHT is good

        # ── Trial type ────────────────────────────────────────────────────────
        # A1 = chose left (0), A2 = chose right (1) — fixed to choice, not good_side
        choice = action_idx - 1  # 0=left, 1=right
        arm_label  = 'A1' if choice == 0 else 'A2'
        trial_type = f'{arm_label},R={int(reward)}'

        records.append({
            'forced_choice': int(forced),
            'choice'       : action_idx - 1,          # 0=left, 1=right
            'outcome'      : int(reward),
            'good_poke'    : good_side_before,
            'prob_A'       : p_A,
            'prob_B'       : p_B,
            'logit_value'  : logit,
            'trial_type'   : trial_type,
        })

    return records


def simulate(n_episodes=10, n_trials=200, seed=42, **task_kwargs):
    np.random.seed(seed)

    p_switch = task_kwargs.pop('p_switch', 0.05)
    task     = ReversalTask(**task_kwargs)
    agent    = BayesAgent(
        good_prob=task.good_prob,
        bad_prob =task.bad_prob,
        p_switch =p_switch,
    )

    all_records = []
    for ep in range(n_episodes):
        records = run_episode(task, agent, n_trials)
        for t, r in enumerate(records):
            r['episode'] = ep
            r['trial']   = t
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    # Derived columns
    df['logit_past']   = df.groupby('episode')['logit_value'].shift(1)
    df['logit_change'] = df['logit_value'] - df['logit_past']

    # Reorder columns
    cols = ['episode', 'trial', 'forced_choice', 'choice', 'outcome',
            'good_poke', 'prob_A', 'prob_B',
            'logit_value', 'logit_past', 'logit_change', 'trial_type']
    return df[cols]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(df, episode=0):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ep = df[df['episode'] == episode].reset_index(drop=True)
    trials = ep['trial']

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Bayesian Reversal Task — Episode {episode}', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(4, 1, hspace=0.55)

    # ── Panel 1: belief state ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(trials, ep['prob_B'], color='steelblue', lw=1.5, label='P(Right good)')
    ax1.plot(trials, ep['prob_A'], color='tomato',    lw=1.5, label='P(Left good)')
    ax1.axhline(0.5, color='grey', lw=0.8, ls='--')
    ax1.set_ylabel('Belief probability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_title('Posterior belief state')

    # shade reversals (good_poke changes)
    _shade_reversals(ax1, ep)

    # ── Panel 2: logit value ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(trials, ep['logit_value'], color='darkorchid', lw=1.5)
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.set_ylabel('Log-odds (right good)')
    ax2.set_title('Logit belief value')
    _shade_reversals(ax2, ep)

    # ── Panel 3: logit change ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    colors = ep['logit_change'].apply(lambda x: 'seagreen' if x >= 0 else 'tomato')
    ax3.bar(trials, ep['logit_change'], color=colors, width=0.8)
    ax3.axhline(0, color='grey', lw=0.8, ls='--')
    ax3.set_ylabel('Δ logit')
    ax3.set_title('Trial-by-trial logit change')
    _shade_reversals(ax3, ep)

    # ── Panel 4: choices & outcomes ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    rewarded   = ep['outcome'] == 1
    unrewarded = ep['outcome'] == 0
    forced     = ep['forced_choice'] == 1

    ax4.scatter(trials[rewarded   & ~forced], ep['choice'][rewarded   & ~forced] + 0.02,
                marker='o', s=18, color='steelblue',  label='Free, rewarded',   zorder=3)
    ax4.scatter(trials[unrewarded & ~forced], ep['choice'][unrewarded & ~forced] - 0.02,
                marker='x', s=18, color='tomato',     label='Free, unrewarded', zorder=3)
    ax4.scatter(trials[forced], ep['choice'][forced],
                marker='D', s=14, color='grey', alpha=0.6, label='Forced',      zorder=2)

    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Left (0)', 'Right (1)'])
    ax4.set_xlabel('Trial')
    ax4.set_title('Choices and outcomes')
    ax4.legend(fontsize=7, loc='upper right', ncol=3)
    _shade_reversals(ax4, ep)

    plt.savefig('reversal_task_plot.png', dpi=150, bbox_inches='tight')
    print('Plot saved to reversal_task_plot.png')
    plt.show()


def _shade_reversals(ax, ep):
    """Shade background whenever good_poke changes."""
    good = ep['good_poke'].values
    in_block = good == 'right'
    start = None
    for i, val in enumerate(in_block):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ax.axvspan(start, i, alpha=0.08, color='steelblue')
            start = None
    if start is not None:
        ax.axvspan(start, len(good), alpha=0.08, color='steelblue')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = simulate(n_episodes=10, n_trials=300)
    print(f'Total trials: {len(df)}')
    print(f'Trial-type counts:\n{df.trial_type.value_counts()}')
    df.to_csv('reversal_task_output.csv', index=False)
    print('Saved to reversal_task_output.csv')
    # ── Reversal diagnostics ──────────────────────────────────────────────────
    reversals = df.groupby('episode').apply(
        lambda e: (e['good_poke'] != e['good_poke'].shift()).sum() - 1  # subtract initial
    )
    print(f'\nReversals per episode (mean ± std): {reversals.mean():.1f} ± {reversals.std():.1f}')
    print(f'Min: {reversals.min()}  Max: {reversals.max()}')

    # Reward rate by trial type
    print(f'\nOverall reward rate: {df["outcome"].mean():.3f}')
    print(f'Free-choice reward rate: {df[df["forced_choice"]==0]["outcome"].mean():.3f}')
    print(f'Forced-choice reward rate: {df[df["forced_choice"]==1]["outcome"].mean():.3f}')

    plot_results(df, episode=0)
    
    sns.scatterplot(df, x='logit_value',y='logit_change',hue='trial_type', 
                alpha =0.3)