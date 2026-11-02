"""Generate synthetic trials for a reproducible, private-data-free fitting example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

if __package__:
    from .run_comparison import run_comparison
else:
    from run_comparison import run_comparison


def get_example_trials_df(*, n_subjects=3, n_sessions=3, n_trials=100, seed=14):
    """Return one row per synthetic trial from reward EMA plus EMA choice history.

    This is an illustrative data generator, not a fitted animal dataset or a
    task-optimal agent. Values/history reset each session. Forced inputs update
    both traces and are excluded from the subsequent fitting likelihood.
    """
    if min(n_subjects, n_sessions, n_trials) < 1:
        raise ValueError("Subject/session/trial counts must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for subject_index in range(n_subjects):
        alpha, eta = .2 + .05 * subject_index, .35
        if alpha > 1:
            raise ValueError("This simple example supports at most17 subjects")
        for session_index in range(n_sessions):
            Q, history = np.zeros(2), 0.0
            for trial in range(n_trials):
                forced = trial % 11 == 0
                logit_right = 3.0 * (Q[1] - Q[0] + .25 * history - .03 * subject_index)
                choice = int(rng.integers(2) if forced else rng.random() < expit(logit_right))
                high_side = ((trial // 25) + session_index) % 2
                reward = int(rng.random() < (.75 if choice == high_side else .25))
                rows.append(dict(subject=f"example_{subject_index + 1:02d}", session_id=f"session_{session_index + 1}",
                                 trial=trial + 1, choice=choice, outcome=reward, forced_choice=forced))
                Q *= 1 - alpha
                Q[choice] += alpha * reward
                history = (1 - eta) * history + eta * (2 * choice - 1)
    return pd.DataFrame(rows)


def main(argv=None):
    """Write synthetic data and optionally run ML/MAP/LOSO in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "synthetic_trials.csv"
    get_example_trials_df(n_trials=args.trials).to_csv(csv_path, index=False)
    if args.fit:
        run_comparison(csv_path, args.output / "fits", repeats=args.repeats, compute_cv=args.cv)
    print(f"Synthetic example saved to {args.output}")


if __name__ == "__main__":
    main()
