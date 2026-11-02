"""Command-line fitting/comparison; inputs and output directory are always explicit."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy

if __package__:
    from .model_comparison_helper_functions import model_comparison
    from .model_fitting_helper_functions import fit_subjects, score_sessions
    from .model_implementation_model_classes import PRIMARY_MODELS, make_models
    from .model_implementation_helper_functions import load_sessions
else:
    from model_comparison_helper_functions import model_comparison
    from model_fitting_helper_functions import fit_subjects, score_sessions
    from model_implementation_model_classes import PRIMARY_MODELS, make_models
    from model_implementation_helper_functions import load_sessions


def _json_values(value):
    """Write strict JSON; undefined numeric diagnostics become explicitly documented nulls."""
    if isinstance(value, dict):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_values(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    """Save reviewable strict JSON, without pickle or code evaluation."""
    Path(path).write_text(json.dumps(_json_values(value), indent=2, allow_nan=False) + "\n")


def _split_groups(sessions, split_plan):
    """Resolve disjoint physical-session splits; never infer a held-out fold."""
    if split_plan is None:
        return [("full_data", sessions, {})]
    if set(split_plan) != {"splits"} or not isinstance(split_plan["splits"], list) or not split_plan["splits"]:
        raise ValueError("Split-plan JSON must contain a nonempty 'splits' list")
    groups = {}
    assigned = set()
    available_subjects = {session.info.subject for session in sessions}
    for specification in split_plan["splits"]:
        if not isinstance(specification, dict) or not {"name", "subject", "train"}.issubset(specification):
            raise ValueError("Each split specification needs name,subject,train")
        if set(specification) - {"name", "subject", "train", "val", "eval"}:
            raise ValueError("Unknown split-plan key; use name,subject,train,val,eval")
        name, subject = str(specification["name"]), str(specification["subject"])
        if not name or subject not in available_subjects or (name, subject) in assigned:
            raise ValueError("Split names/subjects must be valid and unique per protocol/subject")
        assigned.add((name, subject))
        subset = [session for session in sessions if session.info.subject == subject]
        available = {session.info.source_session_id for session in subset}
        identifiers = {}
        for split in ("train", "val", "eval"):
            values = specification.get(split, [])
            if not isinstance(values, list) or len(values) != len(set(map(str, values))):
                raise ValueError("Each split must be a list of distinct physical session IDs")
            identifiers[split] = set(map(str, values))
            if not identifiers[split].issubset(available):
                raise ValueError(f"Unavailable {subject}/{split} sessions: {identifiers[split] - available}")
        if not identifiers["train"] or not (identifiers["val"] or identifiers["eval"]):
            raise ValueError("An explicit split needs training and at least one held-out session")
        if any(identifiers[a] & identifiers[b] for a, b in (("train", "val"), ("train", "eval"), ("val", "eval"))):
            raise ValueError("Train, validation and evaluation physical sessions must be disjoint")
        entry = groups.setdefault(name, {"train": [], "val": [], "eval": []})
        for split in entry:
            entry[split].extend(session for session in subset if session.info.source_session_id in identifiers[split])
    for name, entry in groups.items():
        if {session.info.subject for session in entry["train"]} != available_subjects:
            raise ValueError(f"Protocol {name!r} must explicitly include every input subject")
        for split in ("val", "eval"):
            if entry[split] and {session.info.subject for session in entry[split]} != available_subjects:
                raise ValueError(f"Protocol {name!r}/{split} must cover every input subject")
    return [(name, entry["train"], {key: value for key, value in entry.items() if key != "train" and value})
            for name, entry in groups.items()]


def run_comparison(
    csv_path,
    output_dir,
    *,
    models=PRIMARY_MODELS,
    methods=("ML", "MAP"),
    repeats=10,
    seed=0,
    compute_cv=False,
    score_column=None,
    split_plan=None,
    optimizer_options=None,
    test="paired_t",
):
    """Fit models and save subject-unit comparisons on exactly matched targets.

    The output is a NEW directory. Numerical summaries include both raw and
    per-trial AIC/BIC/log-likelihood/NLL, accuracy and optionally recording-LOSO
    metrics. Explicit split plans add frozen-parameter validation/evaluation
    scores; no evaluation target selects model parameters or optimizer starts.
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    sessions = load_sessions(csv_path, score_column=score_column)
    models = tuple(models)
    model_objects = make_models(models)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"ML", "MAP"}:
        raise ValueError("methods must be distinct ML/MAP entries")
    if test not in ("paired_t", "wilcoxon", "both"):
        raise ValueError("test must be paired_t,wilcoxon or both")
    tests = (True, False) if test == "both" else (test == "paired_t",)
    groups = _split_groups(sessions, split_plan)
    if output_dir.exists():
        raise FileExistsError(f"Preserve existing output; choose a new directory: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = dict(
        complete=False, input_csv=str(csv_path.resolve()),
        input_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        models=list(models), methods=list(methods), random_restarts=repeats, seed=seed,
        compute_cv=compute_cv, score_column=score_column, split_plan=split_plan,
        optimizer_options=optimizer_options, paired_test=test,
        implementation_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        model_specifications=[dict(
            name=model.name, model_class=type(model).__name__, param_names=model.param_names,
            param_ranges=model.param_ranges, param_bounds=model.param_bounds,
            options={key: getattr(model, key) for key in (
                "bias", "prsv", "prsv_ema", "forget", "learning", "history", "target", "read_env_from_session",
                "rule", "decaying_bias", "initial_value",
            ) if hasattr(model, key)},
        ) for model in model_objects],
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__
        ),
        data_use=dict(
            source="Explicit user-provided CSV; no RNN loading or fit selection",
            subjects=sorted({session.info.subject for session in sessions}),
            n_reset_sequences=len(sessions), n_input_trials=sum(session.n_trials for session in sessions),
            timing="Pre-choice states score current choice; post-input states predict next choice",
            resets="Every Session/sequence_id; physical recordings kept together in LOSO",
            scoring="Free choices AND optional score column; all forced/masked inputs still update state",
            row_unit="One subject/model/method/protocol per fit; trial rows in input",
            aggregation="Sum losses/counts within subject before ratios; mean/sample SEM across subjects",
            exclusions="Forced/masked targets and explicitly unused split-plan recordings; no silent subject/row drops",
            protocols=[dict(
                name=name,
                split_source_sessions={key: sorted({(session.info.subject, session.info.source_session_id)
                                                  for session in subset})
                                       for key, subset in {"train": training, **test_sets}.items()},
                unused_source_sessions=sorted(
                    {(session.info.subject, session.info.source_session_id) for session in sessions}
                    - {(session.info.subject, session.info.source_session_id)
                       for subset in [training, *test_sets.values()] for session in subset}
                ),
                unused_input_trials=sum(
                    session.n_trials for session in sessions
                    if (session.info.subject, session.info.source_session_id) not in {
                        (selected.info.subject, selected.info.source_session_id)
                        for subset in [training, *test_sets.values()] for selected in subset
                    }
                ),
            ) for name, training, test_sets in groups],
        ),
        json_null=(
            "Undefined/nonfinite numeric diagnostics, e.g. single-subject SEM or unavailable AICc; "
            "inspect field/context"
        ),
    )
    write_json(output_dir / "RUN.json", manifest)
    fit_tables, summary_tables, value_tables, target_tables = [], [], [], []
    omitted_metrics = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for protocol, training, held_out in groups:
            for model_index, model in enumerate(model_objects):
                for method in methods:
                    fits_df = fit_subjects(
                        training, model, MAP_fit=method == "MAP", compute_CVLL=compute_cv,
                        repeats=repeats, seed=seed + 1000 * model_index, optimizer_options=optimizer_options,
                    )
                    fits_df["protocol"] = protocol
                    fit_tables.append(fits_df)
                    for _, fitted in fits_df.iterrows():
                        for split, test in held_out.items():
                            subset = [session for session in test if session.info.subject == fitted["subject"]]
                            scores = score_sessions(subset, model, fitted["param_values"])
                            row = dict(subject=fitted["subject"], model=model.name, method=method, **scores)
                            row["protocol"], row["split"] = protocol, split
                            value_tables.append(pd.DataFrame([row]))
        all_fits_df = pd.concat(fit_tables, ignore_index=True)
        for protocol, protocol_df in all_fits_df.groupby("protocol", sort=False):
            metrics = ["AIC", "BIC", "log_likelihood", "nll", "nll_per_trial", "accuracy"]
            if np.isfinite(protocol_df["AICc"]).all():
                metrics.append("AICc")
            else:
                omitted_metrics.append(dict(protocol=protocol, metric="AICc", reason="n_scored <= k + 1"))
            if compute_cv:
                metrics += ["CVLL", "CV_nll_per_trial", "CV_accuracy"]
            for metric in metrics:
                for normalise in ([False] if "per_trial" in metric or "accuracy" in metric else [False, True]):
                    for parametric in tests:
                        compared = model_comparison(
                            protocol_df, metric=metric, normalise=normalise, parametric=parametric
                        )
                        score_split = "train_LOSO" if metric.startswith("CV") else "train"
                        table = compared["summary_df"].assign(protocol=protocol, split=score_split, normalise=normalise)
                        summary_tables.append(table)
                        target_tables.append(compared["subject_values_df"].assign(
                            protocol=protocol, split=score_split, normalise=normalise, parametric=parametric
                        ))
        if value_tables:
            held_out_df = pd.concat(value_tables, ignore_index=True)
            for (protocol, split), selected_df in held_out_df.groupby(["protocol", "split"], sort=False):
                for metric in ("log_likelihood", "nll", "nll_per_trial", "accuracy"):
                    for normalise in ([False] if metric in ("nll_per_trial", "accuracy") else [False, True]):
                        for parametric in tests:
                            compared = model_comparison(
                                selected_df, metric=metric, normalise=normalise, parametric=parametric
                            )
                            summary_tables.append(compared["summary_df"].assign(
                                protocol=protocol, split=split, normalise=normalise
                            ))
                            target_tables.append(compared["subject_values_df"].assign(
                                protocol=protocol, split=split, normalise=normalise, parametric=parametric
                            ))
            held_out_df.to_csv(output_dir / "held_out_scores.csv", index=False)
        summaries_df = pd.concat(summary_tables, ignore_index=True)
        summaries_df.to_csv(output_dir / "comparison_summary.csv", index=False)
        pd.concat(target_tables, ignore_index=True).to_csv(output_dir / "comparison_subject_values.csv", index=False)
        # Full JSON preserves arrays, folds, masks/signatures and every optimizer attempt.
        write_json(output_dir / "fits.json", all_fits_df.to_dict(orient="records"))
        scalar_columns = [
            name for name in all_fits_df
            if not all_fits_df[name].map(lambda value: isinstance(value, (dict, list, tuple))).any()
        ]
        all_fits_df[scalar_columns].to_csv(output_dir / "subject_fits.csv", index=False)
    manifest.update(
        complete=True, n_fit_rows=len(all_fits_df),
        n_nonconverged_selected_fits=int((~all_fits_df["best_success"]).sum()),
        n_nonconverged_cv_fits=sum(
            not fold["fit"]["best_success"]
            for folds in all_fits_df["cv_folds"] if isinstance(folds, list) for fold in folds
        ),
        omitted_comparison_metrics=omitted_metrics,
        warnings=sorted({f"{item.category.__name__}: {item.message}" for item in caught}),
    )
    write_json(output_dir / "RUN.json", manifest)
    return dict(fits_df=all_fits_df, summary_df=summaries_df, manifest=manifest)


def main(argv=None):
    """Parse a portable CSV fitting command; never publish or change source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, dest="csv_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir")
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS))
    parser.add_argument("--methods", nargs="+", choices=("ML", "MAP"), default=["ML", "MAP"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv", action="store_true", dest="compute_cv")
    parser.add_argument("--score-column")
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--test", choices=("paired_t", "wilcoxon", "both"), default="paired_t")
    args = vars(parser.parse_args(argv))
    if args["split_plan"] is not None:
        args["split_plan"] = json.loads(args["split_plan"].read_text())
    result = run_comparison(**args)
    print(f"Saved {len(result['fits_df'])} subject/model/method fits to {args['output_dir']}")
    if result["manifest"]["n_nonconverged_selected_fits"]:
        print("Some selected fits did not converge; inspect RUN.json and fits.json before interpretation.")


if __name__ == "__main__":
    main()
