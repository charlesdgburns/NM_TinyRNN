"""Named comparison sets; constructing a model never fits or reads data."""

if __package__:
    from .drift_models import Q_drift_model
    from .mean_baselines import Q_mean_model
    from .model_implementation_model_classes import (
        BayesAsym_2AB,
        BayesSym_2AB,
        Q_forgetting_model,
        Q_history_model,
        Q_model_config,
    )
else:
    from drift_models import Q_drift_model
    from mean_baselines import Q_mean_model
    from model_implementation_model_classes import (
        BayesAsym_2AB,
        BayesSym_2AB,
        Q_forgetting_model,
        Q_history_model,
        Q_model_config,
    )


PRIMARY_MODELS = (
    "Q_basic_bias",
    "Q_visit_prsv",
    "Q_visit_prsvEMA",
    "Reward_EMA_prsvEMA",
    "Q_decay_free_prsvEMA",
    "Q_model_bias_prsvEMA_forget",
    "BayesSym_2AB",
    "BayesAsym_2AB",
)

# Additional models are explicit opt-ins; the earlier comparison default is unchanged.
DRIFT_MODELS = (
    "Q_drift_zero",
    "Q_drift_other",
    "Q_drift_zero_matched",
    "Q_drift_other_no_decay",
    "Q_drift_zero_bias_prsvEMA",
    "Q_drift_other_bias_prsvEMA",
    "Q_drift_zero_matched_bias_prsvEMA",
    "Q_drift_other_no_decay_bias_prsvEMA",
    "Q_drift_zero_matched_no_decay",
    "Q_drift_zero_matched_no_decay_bias_prsvEMA",
)


MEAN_MODELS = ("Q_paper_mean_init_half", "Q_paper_mean_init_zero", "Q_release_mean")


def make_model(name, *, param_bounds=None):
    """Construct one explicit model; Bayesian registry entries use EMA history."""
    factories = {
        "Q_basic": lambda: Q_history_model("visit", "none", bias=False, param_bounds=param_bounds),
        "Q_basic_bias": lambda: Q_history_model("visit", "none", bias=True, param_bounds=param_bounds),
        "Q_visit_prsv": lambda: Q_history_model("visit", "single", bias=True, param_bounds=param_bounds),
        "Q_visit_prsvEMA": lambda: Q_history_model("visit", "ema", bias=True, param_bounds=param_bounds),
        "Reward_EMA_prsv": lambda: Q_history_model("trial", "single", bias=True, param_bounds=param_bounds),
        "Reward_EMA_prsvEMA": lambda: Q_history_model("trial", "ema", bias=True, param_bounds=param_bounds),
        "Q_model_bias_prsvEMA_forget": lambda: Q_model_config(
            bias=True, prsv_ema=True, forget=True, param_bounds=param_bounds
        ),
        "BayesSym_2AB": lambda: BayesSym_2AB(prsv=False, prsv_ema=True, param_bounds=param_bounds),
        "BayesAsym_2AB": lambda: BayesAsym_2AB(prsv=False, prsv_ema=True, param_bounds=param_bounds),
        "Q_paper_mean_init_half": lambda: Q_mean_model("paper", initial_value=.5, param_bounds=param_bounds),
        "Q_paper_mean_init_zero": lambda: Q_mean_model("paper", initial_value=0., param_bounds=param_bounds),
        "Q_release_mean": lambda: Q_mean_model("released", initial_value=.5, param_bounds=param_bounds),
    }
    for target, label in ((0.0, "zero"), (0.5, "half"), ("free", "free")):
        for history, suffix in (("single", "prsv"), ("ema", "prsvEMA")):
            factories[f"Q_decay_{label}_{suffix}"] = (
                lambda target=target, history=history: Q_forgetting_model(
                    target=target, history=history, param_bounds=param_bounds
                )
            )
    for rule in ("zero", "other", "zero_matched"):
        for decaying_bias in ((True,) if rule == "zero" else (True, False)):
            for bias, history in ((False, "none"), (True, "ema")):
                drift_name = f"Q_drift_{rule}"
                if rule != "zero" and not decaying_bias:
                    drift_name += "_no_decay"
                if bias:
                    drift_name += "_bias_prsvEMA"
                factories[drift_name] = (
                    lambda rule=rule, bias=bias, history=history, decaying_bias=decaying_bias: Q_drift_model(
                        rule, bias=bias, history=history, decaying_bias=decaying_bias, param_bounds=param_bounds
                    )
                )
    if name not in factories:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(factories)}")
    model = factories[name]()
    if model.name != name:
        raise RuntimeError(f"Registry name {name!r} disagrees with model.name {model.name!r}")
    return model


def make_models(names=PRIMARY_MODELS):
    """Return distinct models in caller-specified order."""
    names = tuple(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose a nonempty sequence of distinct model names")
    return [make_model(name) for name in names]
