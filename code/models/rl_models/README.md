# Cognitive models for NM_TinyRNN

Compact two-action Q-learning and Bayesian models, written in the explicit style
of Thomas Akam's `RL_model_fitting` notebook. This directory is self-contained:
NumPy, pandas and SciPy are the only runtime dependencies. It does not import
PyTorch, private project code, saved animal data, or the RNN training pipeline.

Copy the complete **`rl_models` directory** into `NM_TinyRNN/code/models/`.
It is additive: do not replace the existing `qlearning.py`, `rnns.py` or trainers.
No files have been published or pushed. See `PROVENANCE.json`, `CHANGELOG.md`
and `VALIDATION.md` for the source revisions, intentional fixes and tested scope.
The GPLv3 license is preserved in `LICENSE`.

## Contents

| File | Purpose |
|---|---|
| `model_implementation_model_classes.py` | Requested legacy classes and consistent history/forgetting variants |
| `model_implementation_helper_functions.py` | Input checks, original binary softmax and likelihood floor |
| `model_fitting_helper_functions.py` | Reproducible multistart ML/MAP, priors, bounds and optimizer diagnostics |
| `model_comparison_helper_functions.py` | Recording-session LOSO and paired subject comparisons |
| `sessions.py` | Array/CSV input, scoring masks and reset identities |
| `registry.py` | Explicit named comparison models |
| `run_comparison.py` | Portable command-line fitting, split scoring and numerical comparisons |
| `example.py` | Synthetic data generator and small end-to-end example |
| `EQUATIONS.md` | Equations, parameter order, bounds, timing and history conventions |
| `tests/` | Independent equation/parity, numerical, fitting and integration tests |

There are no OpAL, meta-RL or experimental BRL classes. `BayesAsym` is an alias
for the source's `BayesAsym_2AB`; no separate source class named `BayesAsym` existed.

## Models

The default registry reproduces the five screenshot comparisons and three
previously successful reference models:

| Registry name | Free parameters | Description |
|---|---:|---|
| `Q_basic_bias` | 3 | Per-visit chosen-only Q with side bias |
| `Q_visit_prsv` | 4 | Per-visit Q with single-trial history |
| `Q_visit_prsvEMA` | 5 | Per-visit Q with EMA history |
| `Reward_EMA_prsvEMA` | 5 | Per-trial reward EMA with EMA history |
| `Q_decay_free_prsvEMA` | 7 | Independent unchosen forgetting towards a fitted target, with EMA history |
| `Q_model_bias_prsvEMA_forget` | 6 | Unchanged legacy half-scale EMA history and forgetting towards .5 |
| `BayesSym_2AB` | 5 | Symmetric Bayesian evidence with EMA history |
| `BayesAsym_2AB` | 5 | Reward-only evidence with EMA history |

Additional registry choices are the unbiased `Q_basic`, `Reward_EMA_prsv`, and
`Q_decay_{zero,half,free}_{prsv,prsvEMA}`. Their inclusion does not silently fit
them: pass the desired names. Direct class constructors also expose basic
unbiased Q, history-free Bayesian models and other legacy configurations.
The Bayesian registry entries deliberately use `prsv=False, prsv_ema=True`;
the legacy classes retain their original single-history constructor defaults.

The opt-in `DRIFT_MODELS` set adds the published conventional zero-drift and
other-drift cores, b=0 ablations, and explicit side-bias/EMA-history extensions.
A new matched omission-zero control keeps the other-model rewarded rule to
isolate omission coupling. See [DRIFT_MODELS.md](DRIFT_MODELS.md) for equations,
source mappings, parameter orders, bounds and distinctions from the published
models. The previous `PRIMARY_MODELS` default is unchanged.

`MEAN_MODELS` adds three explicit, three-parameter forgetting-to-mean models:
`Q_paper_mean_init_half`, `Q_paper_mean_init_zero` and `Q_release_mean`.
The paper variants forget only the unchosen value; the released-code variant
forgets both values after learning. They are available through the same
registry/CLI, with distinct initialization and parameter definitions preserved.
See `MEAN_BASELINES.txt` and `MODEL_CATALOG.md` for the complete model inventory.

## Installation and commands

From a terminal in the directory **containing** `rl_models` (e.g. `code/models`):

```bash
python -m pip install -r rl_models/requirements.txt
python -m unittest discover -s rl_models/tests -v
python -m rl_models --help
python -m rl_models.example --output /tmp/rl_example --fit --cv --repeats 2
```

Alternatively install the directory itself, `python -m pip install ./rl_models`.
This optional standalone distribution installs the import package `rl_models`;
normal repository use does not require installation. From another directory,
install it or put its parent on `PYTHONPATH`. Importing an uninstalled sibling
directory from an unrelated working directory is not supported.

The example generates three **synthetic** subjects with three sessions each.
It is a workflow check, not a recovery study or a new fit to experimental data.
Its two-start setting is for speed; scientific fitting defaults to ten starts.

Fit real CSV data:

```bash
python -m rl_models --input trials.csv --output results/run_001 \
  --methods ML MAP --repeats 10 --seed 0 --cv --test both
```

Use `--models Q_basic_bias Reward_EMA_prsvEMA Q_decay_free_prsvEMA` to select
models. The output directory must be new; existing results are never overwritten.
The same CLI also works as `python rl_models/run_comparison.py ...`.

## Trial input and reset contract

Required CSV columns: `subject,session_id,trial,choice,outcome,forced_choice`.
One row is one physical trial. `choice` is **0=left, 1=right**, or explicitly
`poke_4/poke_6`; one/two coding is rejected. Outcome is binary. `forced_choice`
uses 0/1 or true/false. Subjects and session IDs are read as strings.

Within each subject/recording, trial numbers must be unique and strictly
increasing in file order. No sorting, imputation or subject dropping is hidden
in the loader. Both forced and free trials update values, beliefs and history.

By default every recording resets Q/history to 0 or belief to .5. Optional
`sequence_id` creates smaller, contiguous, independently reset sequences within
a recording. **LOSO holds out all sequences from the recording together.**
An optional Boolean column chosen with `--score-column eligible` intersects
free-choice eligibility without removing history inputs. Supply explicit masks
for block-start/tail exclusions required by a particular experiment; this
generic folder does not infer RNN block membership or align saved RNN trials.

Optional `reward_prob_high,reward_prob_low` must both be supplied and constant
within recordings. Only legacy asymmetric Bayes reads these metadata by default;
symmetric Bayes retains its original fixed .75/.25 observation model.

## Explicit train/validation/evaluation splits

`--split-plan splits.json` fits only the listed training recordings and scores
validation/evaluation with those frozen parameters. No evaluation data select
parameters. IDs refer to physical CSV `session_id`, not `sequence_id`:

```json
{"splits": [
  {"name": "fixed_split", "subject": "mouse_1",
   "train": ["session_1", "session_2"], "eval": ["session_3"]}
]}
```

Specify every input subject in each named protocol. Train/val/eval recordings
must be disjoint. Unused recordings are explicitly recorded in `RUN.json`.
When `--cv` is combined with a plan, LOSO uses only its training recordings and
is labeled `train_LOSO`; it does not touch the external val/eval recordings.

For the previous eight-session block-fit versus nine-complete-session protocol,
prepare the appropriate reset sequences and masks separately for each run.
This folder preserves user-supplied histories; it does not claim those two
different training-history protocols become identical through a toggle.

## Python interface

```python
from rl_models import Q_model_config, make_model, load_sessions
from rl_models.model_fitting_helper_functions import fit_subjects
from rl_models.model_comparison_helper_functions import model_comparison

legacy = Q_model_config(bias=True, prsv_ema=True, forget=True)
models = [make_model("Q_visit_prsvEMA"), make_model("Reward_EMA_prsvEMA")]
sessions = load_sessions("trials.csv")
fits = [fit_subjects(sessions, model, repeats=10, seed=0,
                     MAP_fit=False, compute_CVLL=True) for model in models]
comparison = model_comparison(fits, metric="CVLL", normalise=True)
summary_df = comparison["summary_df"]
subject_values_df = comparison["subject_values_df"]
```

Each fit table row is one subject/model/method. Individual attempts and CV folds
remain in the table and saved JSON. Use `model.forward(session, params)` for
inspectable arrays and `model.get_trial_variables(...)` for a three-level
`('model_vars', variable, '')` DataFrame. In the module docstrings, pre-choice
column t predicts choice t from earlier inputs; post-input column t predicts
the next choice. Do not apply an additional global shift to both products.

## Comparisons and reproducibility

Bounds, independent Beta(2,2)/Gamma(shape=2, scale=2)/Normal(0,5) priors, per-parameter
log-prior floor, L-BFGS-B, ten random starts and lowest-objective selection
follow the original framework. A local NumPy Generator makes new runs repeatable
without changing caller global RNG. Identical seeds are not asserted to recreate
the original global-RNG starting vectors. Optional explicit feasible starts add
to, rather than replace, random starts. Custom-bound starts are clipped into
those bounds. AIC/AICc/BIC count parameters not fixed by equal lower/upper bounds;
`n_params` remains the parameter-vector length, while `n_ic_params` and
`n_optimized_params` record the free dimension. This corrects the earlier penalty
for fixed-bound controls; unrestricted model scores are unchanged.

For example, `make_model("BayesAsym_2AB", param_bounds={"q": (.5, .5)})`
has five vector entries and four optimized/penalized parameters. An eta=1
history control uses `param_bounds={"alp_p": (1., 1.)}`. Fit these configured
objects through `fit_sessions`/`fit_subjects`; the CSV CLI currently accepts
named registry models rather than custom-bound specifications.

The original likelihood uses a probability floor, retained for parity, not a
replacement log-sigmoid objective. `unit`=[0,1], `pos`=[0,infinity), `unc`=signed.
All parameters remain fixed throughout a sequence; model states evolve.

Available metrics: log likelihood, NLL, NLL/trial, accuracy, AIC, AICc, BIC, CVLL,
CV NLL/trial and CV accuracy. AIC/BIC use the unpenalized likelihood even for
MAP estimates; MAP posterior scores are stored separately. Information criteria
at MAP estimates are source-compatible diagnostics, not a guarantee of the
standard ML-based asymptotic interpretation. AICc requires n>k+1; undefined
values remain explicit and the CLI records any omitted AICc comparison.

Raw and per-trial extensive metrics are both exported. Ratio metrics are never
divided twice. CV sums held-out log likelihoods and target counts before the
within-subject ratio. Comparisons require identical subjects, input/reset
identities and score masks; counts alone are insufficient. Mean/sample SEM use
subjects. Paired t/Wilcoxon P values compare alternatives with a reference
selected in the same table: they are descriptive and multiplicity-uncorrected.

`RUN.json` records source/code hashes, model settings, reset/split definitions,
versions, warnings and convergence. `fits.json` preserves every parameter vector,
attempt and fold. `subject_fits.csv`, `comparison_subject_values.csv` and
`comparison_summary.csv` expose numerical results without plotting side effects.
Explicit splits additionally write `held_out_scores.csv`. JSON null represents
an unavailable/undefined numeric diagnostic; inspect its field and context.

`complete=true` means the run finished; it is **not** a convergence guarantee.
Selected unsuccessful fits are retained, warned and counted, including CV fits.
An interrupted/failed run retains `complete=false`. Inspect diagnostics before
scientific interpretation. Neither finite multistart fitting nor the tests prove
global optimality or exact equivalence of these models with a monoGRU.

## Source parity audit

Portable tests use independent reference equations. To additionally compare
against an unedited copy of the original local framework:

```bash
ORIGINAL_RL_SOURCE=/path/to/original/rl \
  python -m unittest discover -s rl_models/tests -v
```

This optional audit is skipped when the source directory is not supplied. It
requires no private behavioral data. Bayesian exporter corrections and legacy
versus consistent history scaling are deliberately tested and documented.

## Attribution

The model interface and fitting conventions are adapted from
[Thomas Akam's RL_model_fitting notebook](https://github.com/cognitive-circuits/RL_model_fitting/blob/192bb197d6b4e20f5401bde95b9bbe1265b13365/model_fitting.ipynb)
and the user's supplied local adaptation. The additional signed-history and
free-target model definitions reproduce the separately validated comparison
implementations described in `PROVENANCE.json`. The intended destination is an
additive subfolder of [NM_TinyRNN's models directory](https://github.com/charlesdgburns/NM_TinyRNN/tree/main/code/models).
