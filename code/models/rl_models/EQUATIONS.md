# Equations and parameter definitions

Choice a=0 is left; a=1 is right. Reward r is 0/1. Original Q/history and drift
models start at Q_L=Q_R=H=0; Bayesian belief starts at b=.5. The explicit mean
baselines instead use their named zero/.5 initialization. Signed history is positive for
right choices. State at t is **before choice t**, after inputs through t-1.
Below, primes mean after receiving the current choice/outcome.

## Q values and history

```text
Q_chosen'   = (1-alpha) Q_chosen + alpha*r
Q_unchosen' = (1-f) Q_unchosen + f*tau
H'          = (1-eta) H + eta*(2*a-1)
L_right     = iTemp * (Q_R-Q_L + prsv*H - bias)
L_left      = -L_right
P(right)    = sigmoid(L_right)
```

`alpha` is chosen learning; `f=alp_f` is the independent unchosen forgetting
rate; `tau=decay_target` is the forgetting target; `eta=alp_p` is history update;
`prsv` weights history; `iTemp` scales the entire policy; `bias` favors left.
Single history sets eta=1. No history removes its term and parameters.

| Class/configuration | Value restriction | Parameter order |
|---|---|---|
| `Q_model()` | f=0, no bias/history | alpha,iTemp |
| `Q_model_bias()` | f=0, no history | alpha,iTemp,bias |
| `Q_model_forget()` | tau=.5, no bias/history | alpha,iTemp,alp_f |
| `Q_history_model("visit","none")` | f=0 | alpha,iTemp,bias |
| `Q_history_model("visit","single")` | f=0, eta=1 | alpha,iTemp,bias,prsv |
| `Q_history_model("visit","ema")` | f=0 | alpha,iTemp,bias,prsv,alp_p |
| `Q_history_model("trial","single")` | f=alpha,tau=0,eta=1 | alpha,iTemp,bias,prsv |
| `Q_history_model("trial","ema")` | f=alpha,tau=0 | alpha,iTemp,bias,prsv,alp_p |
| `Q_forgetting_model(0,"ema")` | tau=0 | alpha,iTemp,bias,prsv,alp_p,alp_f |
| `Q_forgetting_model(.5,"ema")` | tau=.5 | alpha,iTemp,bias,prsv,alp_p,alp_f |
| `Q_forgetting_model("free","ema")` | independent f,tau | alpha,iTemp,bias,prsv,alp_p,alp_f,decay_target |

The corresponding single-history forgetting variants omit alp_p. Q_history_model
also permits bias=False, which omits bias from the parameter list. No rate
ordering is imposed on behavioral fits: alpha may exceed f. The monoGRU core
construction discussed elsewhere covers only a restricted subset; it is not a
constraint on these cognitive models.

### Original configurable Q convention

`Q_model_config` appends parameters in this order: alpha,iTemp; optional bias;
optional prsv (single) OR prsv,alp_p (EMA); optional alp_f. Forgetting targets .5.
Both history flags together raise an error instead of duplicating a parameter.
The original single-history prsv is nonnegative; the original EMA prsv is signed.

Legacy EMA uses P'= (1-eta)P+eta*(a-.5), hence P=H/2. Its history contribution
is `prsv_legacy*H/2`. Therefore equal policy strength requires
`prsv_new=prsv_legacy/2` when mapping to the full-scale variants. The legacy
single-history action bonus is softmax-equivalent to prsv*H in the contrast.

The two parameters prsv and alp_p belong to **one** choice-history process;
they are not separate processes called perseveration and choice history.
Forced choices enter this history even when their own likelihood is excluded.

## Bayesian belief and policy

Let b=P(RightGood). Hazard q first gives b_pred=q+(1-2q)b. The symmetric
model updates using both reward and omission likelihoods:

```text
ell_R = P(observation | RightGood, chosen action)
ell_L = P(observation | LeftGood, chosen action)
b' = ell_R*b_pred / (ell_R*b_pred + ell_L*(1-b_pred))
```

For rewarded right, (ell_R,ell_L)=(rho_high,rho_low); rewarded left swaps them.
Omission uses (1-ell_R,1-ell_L). BayesAsym_2AB instead makes omission
uninformative: b'=b_pred. Its source denominator safeguard max(denominator,1e-12)
is retained for fidelity. Degenerate reward-probability settings deserve care:
this safeguard is not an alternative generative observation model.

```text
H' = (1-alp_p) H + alp_p*(2*a-1)       [EMA; single sets alp_p=1]
L_right = beta * (2*b-1 + prsv*H - bias_left)
Q_R = rho_low + (rho_high-rho_low)*b
Q_L = rho_low + (rho_high-rho_low)*(1-b)
```

Expected reward values Q_R,Q_L are exported for interpretation; the fitted
policy uses **belief contrast** 2*b-1, not Q_R-Q_L. They differ by rho_high-rho_low,
which is .5 under defaults. Exported probabilities now use the fitted policy.

Both Bayesian classes have parameter order beta,q,bias_left; append prsv for
single history or prsv,alp_p for EMA. Default constructors use single history;
the named comparison registry uses EMA explicitly. beta is nonnegative; q is
in [0,1] as in the source (not silently restricted to [0,.5]); bias_left/prsv are
signed. alp_p is in [0,1]. Sym uses fixed rho_high=.75,rho_low=.25; Asym defaults
to these but can read session.info.reward_prob_high/low. Its
read_env_from_session=False option disables that metadata override.

## Bounds, priors and scoring

Rates and free target: unit=[0,1], Beta(2,2) prior.
Inverse temperatures: pos=[0,infinity), Gamma(shape=2, scale=2) prior.
Signed coefficients: unc=(-infinity,infinity), Normal(mean=0, SD=5) prior.
The original per-parameter log-prior floor is -1000. Custom bounds must stay
inside mathematical domains. ML minimizes negative summed log likelihood;
MAP additionally subtracts summed log priors once per shared parameter vector.

Original binary softmax and log_safe are retained, including right probability
computed as 1-P(left) and log_safe(p)=log(0.5e-200+(1-1e-200)*p). This bounded
log likelihood differs numerically from an unfloored log-sigmoid at saturation.

Only free AND ll_mask targets count in log likelihood and n. All supplied
inputs update state. Let k count parameters not fixed by equal lower/upper
bounds; model.n_params still describes the complete parameter vector:

```text
NLL = -sum(log likelihood over scored targets)
AIC = 2*NLL + 2*k
BIC = 2*NLL + log(n)*k
AICc = AIC + 2*k*(k+1)/(n-k-1)          [defined only for n>k+1]
CVLL = sum(held-out recording log likelihoods)
CV_NLL_per_trial = -CVLL/sum(held-out scored-target counts)
```

Comparisons use one row per subject/model/method. Subject SEM is sample SD
divided by sqrt(number of subjects); one-subject SEM is undefined. Accuracy
chooses left at an exact probability tie. No penalty is added to held-out NLL.

For drift equations see `DRIFT_MODELS.md`; for the distinct paper-equation and
released-code forgetting-to-mean equations see `MEAN_BASELINES.txt`.
