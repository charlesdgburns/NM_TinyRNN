# Model catalogue (release 1.1.0)

28 explicit registry choices. The eight default models are marked below.

Parameters are in constructor/fit-vector order. Counts assume no equal-bound controls.

| Registry name | Parameters | Default | Parameter order |
|---|---:|:---:|---|
| `Q_basic_bias` | 3 | yes | alpha, iTemp, bias |
| `Q_visit_prsv` | 4 | yes | alpha, iTemp, bias, prsv |
| `Q_visit_prsvEMA` | 5 | yes | alpha, iTemp, bias, prsv, alp_p |
| `Reward_EMA_prsvEMA` | 5 | yes | alpha, iTemp, bias, prsv, alp_p |
| `Q_decay_free_prsvEMA` | 7 | yes | alpha, iTemp, bias, prsv, alp_p, alp_f, decay_target |
| `Q_model_bias_prsvEMA_forget` | 6 | yes | alpha, iTemp, bias, prsv, alp_p, alp_f |
| `BayesSym_2AB` | 5 | yes | beta, q, bias_left, prsv, alp_p |
| `BayesAsym_2AB` | 5 | yes | beta, q, bias_left, prsv, alp_p |
| `Q_basic` | 2 |  | alpha, iTemp |
| `Reward_EMA_prsv` | 4 |  | alpha, iTemp, bias, prsv |
| `Q_decay_zero_prsv` | 5 |  | alpha, iTemp, bias, prsv, alp_f |
| `Q_decay_zero_prsvEMA` | 6 |  | alpha, iTemp, bias, prsv, alp_p, alp_f |
| `Q_decay_half_prsv` | 5 |  | alpha, iTemp, bias, prsv, alp_f |
| `Q_decay_half_prsvEMA` | 6 |  | alpha, iTemp, bias, prsv, alp_p, alp_f |
| `Q_decay_free_prsv` | 6 |  | alpha, iTemp, bias, prsv, alp_f, decay_target |
| `Q_drift_zero` | 3 |  | alpha, retention, iTemp |
| `Q_drift_other` | 4 |  | alpha1, alpha2, decay_bias, iTemp |
| `Q_drift_zero_matched` | 4 |  | alpha1, alpha2, decay_bias, iTemp |
| `Q_drift_other_no_decay` | 3 |  | alpha1, alpha2, iTemp |
| `Q_drift_zero_bias_prsvEMA` | 6 |  | alpha, retention, iTemp, bias, prsv, alp_p |
| `Q_drift_other_bias_prsvEMA` | 7 |  | alpha1, alpha2, decay_bias, iTemp, bias, prsv, alp_p |
| `Q_drift_zero_matched_bias_prsvEMA` | 7 |  | alpha1, alpha2, decay_bias, iTemp, bias, prsv, alp_p |
| `Q_drift_other_no_decay_bias_prsvEMA` | 6 |  | alpha1, alpha2, iTemp, bias, prsv, alp_p |
| `Q_drift_zero_matched_no_decay` | 3 |  | alpha1, alpha2, iTemp |
| `Q_drift_zero_matched_no_decay_bias_prsvEMA` | 6 |  | alpha1, alpha2, iTemp, bias, prsv, alp_p |
| `Q_paper_mean_init_half` | 3 |  | alpha, retention, iTemp |
| `Q_paper_mean_init_zero` | 3 |  | alpha, retention, iTemp |
| `Q_release_mean` | 3 |  | alpha, forget_rate, iTemp |

## Direct legacy and configurable classes

- `Q_model`: basic chosen-only Q (2 parameters).
- `Q_model_bias`: chosen-only Q with side bias (3).
- `Q_model_forget`: unchosen forgetting towards .5 (3).
- `Q_model_config`: original bias, single/EMA history and .5-forgetting switches.
- `BayesSym_2AB`, `BayesAsym_2AB` (`BayesAsym` alias): constructor defaults use single history; the registry explicitly uses EMA history (5). History-free constructors have 3 parameters.
- `Q_history_model`, `Q_forgetting_model`, `Q_drift_model`, `Q_mean_model`: explicit configurable value/history families.

Constructor configurations are additional API choices, not extra automatically fitted models. Some classes/registry entries are mathematically equivalent under parameter conversion; the count above is a count of named choices, not distinct mechanisms.

## Definitions and restricted controls

See `EQUATIONS.md`, `DRIFT_MODELS.md` and `MEAN_BASELINES.txt` for equations, bounds, initialization, source distinctions and history scaling. All parameters are fixed within a reset sequence; the model states evolve after every input.

For eta=1 use `param_bounds={"alp_p": (1., 1.)}`; for Bayesian q=.5 use `param_bounds={"q": (.5, .5)}` with `make_model`. Fit the configured object through the Python fitting API. Equal-bound entries stay in the vector but do not count in AIC/AICc/BIC. These are explicit restrictions, not additional registry names.

The portable folder fits explicit CSV/Session inputs. It does not discover saved RNN trials, select GRU fits, or automatically reproduce the project-specific matching pipeline.
