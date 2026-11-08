# RL and Bayesian models

Core files for defining, fitting and comparing two-action reversal-learning
models, following the original local RL folder and Thomas Akam's code style.
All previously published model equations, parameter domains, fitting procedures
and eight-model defaults are retained. Additional named models are explicit opt-ins.

| File | Purpose |
|---|---|
| `model_implementation_model_classes.py` | All Q/Bayesian/drift/mean-forgetting classes and named configurations |
| `model_implementation_helper_functions.py` | Trial validation, likelihood helpers, reset sequences and CSV loading |
| `model_fitting_helper_functions.py` | Multistart ML/MAP fitting, priors, bounds and frozen-parameter scoring |
| `model_comparison_helper_functions.py` | Recording-session LOSO and subject-level comparison metrics |
| `script.py` | CSV fitting/comparison command and explicit train/validation/evaluation splits |
| `__init__.py`, `__main__.py` | Package imports and command-line entry point |
| `requirements.txt`, `LICENSE` | Runtime dependencies and GPLv3 license |

## Models

The core comparison models are listed below. The original eight remain the default;
the two legacy-scale reward-EMA variants are opt-in additions.

| Name | Parameters | Value/history rule |
|---|---:|---|
| `Q_basic_bias` | 3 | Chosen-only prediction-error learning, side bias |
| `Q_visit_prsv` | 4 | Chosen-only learning, single-trial history |
| `Q_visit_prsvEMA` | 5 | Chosen-only learning, EMA history |
| `Reward_EMA_prsvEMA` | 5 | Per-trial reward EMA, EMA history |
| `Q_model_bias_prsvEMA_trial` | 5 | Same reward EMA with signed, legacy half-scale EMA history (opt-in) |
| `Q_model_bias_prsvEMA_trial_positive` | 5 | Legacy-scale reward EMA with nonnegative history strength; the variant fitted to all 11 subjects (opt-in) |
| `Q_decay_free_prsvEMA` | 7 | Independent unchosen forgetting towards a fitted target, EMA history |
| `Q_model_bias_prsvEMA_forget` | 6 | Legacy half-scale EMA history, unchosen forgetting towards .5 |
| `BayesSym_2AB` | 5 | Reward/omission evidence, reversal hazard and EMA history |
| `BayesAsym_2AB` | 5 | Reward-only evidence, reversal hazard and EMA history |

Additional named choices are:

- `Q_basic` (2 parameters) and `Reward_EMA_prsv` (4).
- `Q_decay_{zero,half,free}_{prsv,prsvEMA}`: fixed zero/.5 or fitted forgetting
  target, with single/EMA history (5/6, 5/6 and 6/7 parameters respectively).
- `Q_drift_zero` (3), `Q_drift_other` (4), `Q_drift_zero_matched` (4), and
  their `_bias_prsvEMA` extensions (6/7/7).
- `Q_drift_other_no_decay` and `Q_drift_zero_matched_no_decay` (3), and their
  `_bias_prsvEMA` extensions (6). Here `no_decay` removes the rewarded unchosen
  decrement; it does not remove all value dynamics.
- `Q_paper_mean_init_half`, `Q_paper_mean_init_zero`, `Q_release_mean` (3 each).
  Paper variants forget only the unchosen value towards .5. Released code
  forgets both values after learning and starts at .5.

There are 30 named choices, including equivalent parameterizations and controls.
`make_model(name)` constructs one; `make_models()` constructs the eight defaults.
`DRIFT_MODELS` and `MEAN_MODELS` provide the explicit opt-in sets.

Direct classes include `Q_model`, `Q_model_bias`, `Q_model_forget`,
`Q_model_config`, `Q_history_model`, `Q_forgetting_model`, `Q_drift_model`,
`Q_mean_model`, `Q_model_bias_prsvEMA_trial`, `Q_model_bias_prsvEMA_trial_positive`,
`BayesSym_2AB` and `BayesAsym_2AB` (`BayesAsym` alias).
Bayesian class constructors default to single history; named comparisons
explicitly select `prsv=False, prsv_ema=True`.

Model docstrings and forward methods specify equations and parameter order.
Rates/free targets use [0,1], inverse temperatures are nonnegative, and signed
coefficients are unrestricted. Independent priors are Beta(2,2), Gamma(2,2)
and Normal(0,5), respectively. Legacy Q single-history strength is nonnegative;
its EMA trace retains the original half-scale convention. Drift-other values
are unclipped and may be negative or exceed one.

### Legacy-scale reward EMA

`Q_model_bias_prsvEMA_trial_positive` is the literal legacy-scale, nonnegative-history
variant used in the 11-subject refits. Its signed counterpart is
`Q_model_bias_prsvEMA_trial`; both retain the same five-parameter order:

| Parameter | Meaning | Bounds |
|---|---|---|
| `alpha` | Chosen learning and tied unchosen decay rate | [0,1] |
| `iTemp` | Inverse temperature, beta | [0,infinity) |
| `bias` | Left-side bias before multiplication by beta | Unrestricted |
| `prsv` | Legacy half-scale history strength | Unrestricted; [0,infinity) in `_positive` |
| `alp_p` | History update rate, eta | [0,1] |

With choice `c=0` (left) or `c=1` (right), reward `r`, and unchosen action `u`:

```text
Q_c' = (1-alpha)*Q_c + alpha*r
Q_u' = (1-alpha)*Q_u
P_R' = (1-alp_p)*P_R + alp_p*(c-0.5)
logit_right = iTemp*(Q_R-Q_L + prsv*P_R - bias)
```

States start at zero on every supplied reset sequence. The logit uses pre-choice
states; primes denote the state after receiving the current choice/outcome.
Forced and unscored inputs update both values and history; masks affect only scoring.
`history` and `history_post` export the actual half-scale `P_R` for these classes.

Relative to local `Q_model_config(bias=True, prsv_ema=True, forget=False)`, the
only changed value rule is decay of the unchosen value towards zero on every trial,
at the existing `alpha` rate. There is no additional forgetting rate or fitted target.
The `_positive` bound borrows the original upstream single-trial coefficient domain;
upstream does not itself define this EMA extension.

The signed model is an exact reparameterization of `Reward_EMA_prsvEMA`:
`prsv_legacy = 2*prsv_reward_ema`, since its trace is half as large. The positive
model instead restricts the allowed policies, excluding negative-history solutions.
All remaining parameters were allowed to adapt in the earlier 11-subject ML refits.
This package update supplies the fitting definitions; it does not rerun or bundle
those animal-data fits.

ML equivalence uses the coefficient conversion above. MAP uses this framework's
standard prior in the supplied parameter units: Normal(0,5) for signed `prsv`,
Gamma(shape=2,scale=2) for positive `prsv`. An equivalent MAP comparison in full-scale
units requires transforming that coefficient prior (Normal SD 2.5 or Gamma scale 1).
When `alp_p=0`, history is identically zero and `prsv` is unidentified; when `prsv=0`,
`alp_p` is unidentified. Nonnegative boundary solutions need this interpretation.

## Run

From the NM_TinyRNN repository root:

```bash
python -m pip install -r code/models/rl_models/requirements.txt
python code/models/rl_models/script.py --help
python code/models/rl_models/script.py --input trials.csv --output results/run_001 \
  --models Q_basic_bias Reward_EMA_prsvEMA Q_decay_free_prsvEMA \
  --methods ML MAP --repeats 10 --seed 0 --cv --test both
```

From `code/models`, `python -m rl_models ...` provides the same command.
Choose a new output directory; existing results are never overwritten.

CSV columns: `subject,session_id,trial,choice,outcome,forced_choice`.
One row is one physical trial; choice is 0=left/1=right (or poke_4/poke_6),
outcome is binary, and forced_choice is 0/1 or true/false. Trials must be
strictly chronological within each subject/recording. All inputs update state;
only eligible free choices are scored. Optional `--score-column eligible`
adds a target mask. Optional `sequence_id` specifies smaller reset blocks;
LOSO holds out all blocks from a physical recording together.

Optional `--split-plan splits.json` fits training recordings and scores frozen
parameters on disjoint validation/evaluation recordings. The JSON format is:

```json
{"splits": [{"name": "protocol_1", "subject": "mouse_1",
             "train": ["session_1", "session_2"], "eval": ["session_3"]}]}
```

Include every input subject. With `--cv`, only training recordings enter LOSO.
Saved RNN trial matching and fit selection are supplied separately by the caller.

## Python use and scoring

```python
from rl_models import make_model, load_sessions
from rl_models.model_fitting_helper_functions import fit_subjects
from rl_models.model_comparison_helper_functions import model_comparison

sessions = load_sessions("trials.csv")
model = make_model("Reward_EMA_prsvEMA")
fits_df = fit_subjects(sessions, model, repeats=10, seed=0, compute_CVLL=True)
comparison = model_comparison(fits_df, metric="CVLL", normalise=True)
```

To fit the legacy-scale nonnegative variant, substitute
`make_model("Q_model_bias_prsvEMA_trial_positive")`; the same name is accepted by
`script.py --models`. Direct construction is also available through
`Q_model_bias_prsvEMA_trial(positive_history=True)` or the dedicated `_positive` class.

Pre-choice state t scores choice t; post-input state t predicts the next choice.
Forward outputs and `get_trial_variables()` expose actual states and fitted
probabilities. Positive bias favors left; signed history is right-positive.
Bayesian policies use belief contrast, not expected-reward contrast.

Available metrics: log likelihood, NLL, NLL/trial, accuracy, AIC/AICc/BIC,
CVLL, CV NLL/trial and CV accuracy. Information criteria use unpenalized
likelihood, actual scored targets and the number of parameters not fixed by
equal bounds. `n_params` remains the vector length; `n_ic_params` gives the
penalty dimension. MAP-based criteria retain the original framework's diagnostic
interpretation. Held-out NLL has no added parameter penalty.

For eta=1 or Bayesian q=.5 controls, pass `param_bounds={"alp_p": (1., 1.)}`
or `param_bounds={"q": (.5, .5)}` to `make_model` and use the Python fitter.
Model comparisons require identical subject/input/reset/target identities.
Losses/counts pool within subject before ratios; means/sample SEM use subjects.
Paired tests are descriptive, selected-reference and multiplicity-uncorrected.
Inspect saved convergence diagnostics before interpreting fits.

## Sources and layout

Adapted from [Thomas Akam's RL_model_fitting](https://github.com/cognitive-circuits/RL_model_fitting/tree/192bb197d6b4e20f5401bde95b9bbe1265b13365)
and the supplied local RL code. Drift and mean-forgetting variants refer to
[Ji-An, Benna and Mattar (2025)](https://doi.org/10.1038/s41586-025-09142-4)
and [tinyRNN a8fc159](https://github.com/jil095/tinyRNN/tree/a8fc159a2b2d451be6756ae244b29b7d991aac1f).
Paper-equation, released-code and local-control variants are labelled separately.
Source details also appear beside their definitions. GPLv3 is preserved.

In this compact layout, the previous `registry.py`, `drift_models.py` and
`mean_baselines.py` contents live in the model-class file; `sessions.py` lives
in the implementation helpers; `run_comparison.py` is now `script.py`.
Package-level model/loader imports and the module CLI are retained.
The complete pre-cleanup package, tests and validation records remain in commit
`367ad83f289da4ca05486269c290f18da637d434` and the local review archive.
