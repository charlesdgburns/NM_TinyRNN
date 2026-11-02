# Drift models: published rules and matched controls

These additions adapt the two-action models in Ji-An, Benna and Mattar (2025),
[Methods, equations 8–13](https://www.nature.com/articles/s41586-025-09142-4).
The frozen code reference is
[tinyRNN commit a8fc159](https://github.com/jil095/tinyRNN/tree/a8fc159a2b2d451be6756ae244b29b7d991aac1f).
The original default comparison set is unchanged. Pass explicit registry names
or use the separate `DRIFT_MODELS` tuple to request the new models.

Here c is the chosen action, u is the unchosen action and r is binary reward.
Primes denote the state after the current choice/outcome. Both Q values and
history start at zero at each supplied reset sequence. All forced inputs update
the state, while only free targets intersected with `ll_mask` are scored.

## Published conventional drift to zero

```text
Q_c' = (1-alpha)*Q_c + alpha*r
Q_u' = retention*Q_u
```

`Q_drift_model("zero")`, named `Q_drift_zero`, has three parameters in order:
`alpha, retention, iTemp`. The first two are in [0,1]; inverse temperature is nonnegative.
The source calls retention `beta`; it is not the policy inverse temperature.
Source code: `agents/RTS_RL_agents_v1/Model_based_decay.py`, selected as `MB1`
for the probabilistic reversal task with `p_transit=1` and second_step=choice.

The existing local zero-target forgetting model has exactly the same value
dynamics when `alp_f=1-retention`. The six-parameter version with side bias and
EMA history is therefore an exact reparameterization of `Q_decay_zero_prsvEMA`,
with parameter order converted appropriately. This equivalence includes ML
predictions and the existing symmetric Beta(2,2) rate priors under complementation.
Finite random optimizer starts need not be the same under a seed without mapping
the starts too. Setting retention=1 gives chosen-only Q; retention=1-alpha gives
the reward-EMA value dynamics.

## Published drift to the other

For rewarded inputs:

```text
Q_c' = alpha1*Q_c + 1
Q_u' = Q_u - decay_bias
```

For omitted rewards:

```text
Q_c' = Q_c + alpha2*(Q_u-Q_c)
Q_u' = Q_u
```

`Q_drift_model("other")`, named `Q_drift_other`, has four parameters in source
order: `alpha1, alpha2, decay_bias, iTemp`. The first three lie in [0,1] and
inverse temperature is nonnegative. `alpha1` is rewarded chosen-value **retention**
(paper D1); `alpha2` is omission interpolation towards the other value (paper
alpha0). `decay_bias` is the source's b, a reward-contingent decrement of the
unchosen value. It is not a static left/right choice bias.

Source code: `agents/RTS_RL_agents_v1/Model_based_mix_decay.py`, selected as
`MB0md` by `agents/PRLCogAgent.py` with p_transit=1. The `MB0mdnb` ablation fixes
b=0; use `decaying_bias=False`, named `Q_drift_other_no_decay` (three parameters).

Values are deliberately **unclipped**: repeated unchosen decrements can make them
negative, reward increments can make them exceed one, and alpha1=1 permits linear
growth under repeated rewards. These states are not bounded expected rewards.
Adding clipping or ReLU would change the published model. The generic exported
`prediction_error=reward-Q_chosen` is a descriptive contrast; it is not the
learning signal used by this model's rewarded rule.

## New matched omission-zero control

`Q_drift_model("zero_matched")`, named `Q_drift_zero_matched`, keeps exactly the
published other-model rewarded update, parameter names, bounds and parameter
count. It changes only the omission rule to:

```text
Q_c' = (1-alpha2)*Q_c
Q_u' = Q_u
```

This is a new control, not a published tinyRNN model. Comparing it against
`Q_drift_other` isolates omission coupling without also changing rewarded learning.
The published conventional zero model differs in its rewarded rule, omission
learning, unchosen updates and parameter count. A direct performance difference
between the two published cores cannot identify a unique causal contribution
from drift to the other.

Writing D=Q_R-Q_L, S=Q_R+Q_L and s=2*choice-1 makes the distinction explicit.
On omissions the published other rule gives D'=(1-alpha2)*D and
S'=S-s*alpha2*D. Its value contrast contracts towards indifference regardless
of chosen side. The matched zero rule instead gives
D'=(1-alpha2/2)*D-s*(alpha2/2)*S: preference updating couples to total value.
These are exact identities, not fitted regression approximations; tests verify
them against the trial updates. With history, full choice logit additionally
includes the separate choice-history term, so it need not follow D alone.

## Local bias and history extensions

All cores can add `bias=True` and `history="none"`, `"single"` or `"ema"`.
These are our explicitly identified extensions, rather than paper parameters:

```text
H' = (1-alp_p)*H + alp_p*(2*choice-1)
L_right = iTemp*(Q_R-Q_L + prsv*H - bias)
```

Choice is 0=left/1=right. H is right-positive, and positive bias favors left.
Single history fixes alp_p=1. History-free models remove both history parameters.
Side bias and prsv are signed; alp_p is in [0,1]. Policy parameters are appended
to each core in order: optional `bias`, optional `prsv`, and optional `alp_p`.
The registered `*_bias_prsvEMA` variants therefore have 6 parameters for standard
zero drift and 7 for other/matched-zero drift (6 with b fixed to zero).

The local original binary softmax/floored likelihood is retained for fitting
comparability. tinyRNN's scalar max-subtracted softmax is mathematically equivalent
but can differ at floating-point saturation; unchanged-source state-kernel parity
does not assert bit-identical extreme-probability likelihoods across frameworks.
The clean framework's bounds, independent priors, fitting and scoring rules apply.

## Source and mathematical checks

`tests/test_drift_models.py` checks the equations, parameter domains, unchanged
default registry, unclipped states, matched reward rules, history/mask timing,
exports and exact model nesting. To additionally run unchanged upstream step
function bodies (without requiring their Numba decorators), provide an unedited
repository copy:

```bash
TINY_RNN_SOURCE=/path/to/tinyRNN \
  python -m unittest discover -s rl_models/tests -p test_drift_models.py -v
```

The optional source test extracts only the requested top-level step function and
removes its JIT decorator; its statements are unchanged. It tests the scientific
update kernels, not the original package imports, Numba compilation or trainers.
The original-source copy and its hashes belong in the analysis provenance rather
than as a private absolute runtime dependency of this portable folder.
