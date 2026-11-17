"""Export all 110 action-aligned GRUs using monoGRU's native trial regeneration.

Source/cohort: nested_DA_S32_B8, WS01 WS02 WS05 WS08 WS09 WS10 WS13 WS14 WS16
WS20 WS22, two-unit tanh GRU with decoder bias, exact published ten outer
winners per mouse and stored inner indices. Matching uses release SHA-256,
trial identities and original split labels. Rows are continuous after-input
readouts, including all saved trials; no resets, lags or exclusions. Fits stay
separate. Parameter summaries average folds within subject, then subjects
equally; subjects are the SD/SEM unit. No inferential analysis or retraining.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from NM_TinyRNN.code.analyses import GRU_alignment as alignment
from NM_TinyRNN.code.analyses import export_aligned_monoGRU as mono_export
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.models import datasets as ds, rnns, save_data


def check_trials(regenerated_df, saved_df, permutation, signs):
    """Check predictions, signed states and both unsigned, permuted GRU gates."""
    hidden_columns = ["hidden_1", "hidden_2"]
    expected = saved_df[hidden_columns].to_numpy()[:, permutation] * signs
    np.testing.assert_allclose(regenerated_df[hidden_columns], expected, atol=5e-5, rtol=2e-4)
    record = dict(max_hidden_error=float(np.abs(regenerated_df[hidden_columns].to_numpy() - expected).max()))
    for gate in ("update", "reset"):
        columns = [f"gate_{gate}_1", f"gate_{gate}_2"]
        expected_gate = saved_df[columns].to_numpy()[:, permutation]
        np.testing.assert_allclose(regenerated_df[columns], expected_gate, atol=5e-5, rtol=2e-4)
        record[f"max_{gate}_gate_error"] = float(np.abs(regenerated_df[columns].to_numpy() - expected_gate).max())
    for columns, name in [(["logit_value"], "logit"), (["prob_A", "prob_B"], "probability")]:
        np.testing.assert_allclose(regenerated_df[columns], saved_df[columns], atol=5e-5, rtol=2e-4)
        record[f"max_{name}_error"] = float(np.abs(regenerated_df[columns].to_numpy() - saved_df[columns].to_numpy()).max())
    activations = set(hidden_columns + ["logit_value", "prob_A", "prob_B", "logit_past", "logit_change"]
                      + [f"gate_{gate}_{i}" for gate in ("update", "reset") for i in (1, 2)])
    unchanged = [column for column in saved_df if column not in activations]
    pd.testing.assert_frame_equal(regenerated_df[unchanged], saved_df[unchanged], check_dtype=False)
    return record


def export_subject(subject, rows_df, output_dir, source_dir, alignment_manifest):
    """One native single-threaded process per mouse; all ten folds share raw data."""
    torch.set_num_threads(1)
    subject_df = mono_export.load_subject_data(subject, rows_df.iloc[0])
    session_hashes = {f"data/AB_behaviour/{session}/trials.htsv":
                      mv.sha256(ROOT / "data/AB_behaviour" / session / "trials.htsv")
                      for session in subject_df.session_folder_name.unique()}
    published = json.loads((ROOT / "data/rnns/subject_examples_aligned_monoGRU_provenance.json").read_text())
    for path, digest in session_hashes.items():
        if published["subject_data_sha256"].get(path) != digest:
            raise ValueError(f"Raw session differs from frozen published regeneration input: {path}")
    exports, audits, artifacts = [], [], []
    for row in rows_df.itertuples():
        source_paths = alignment.source_paths(row, source_dir)
        model = alignment.load_model(row.aligned_model_path)
        saved_df = pd.read_csv(source_paths["trials"], sep="\t")
        regenerated_df = mono_export.regenerate_trials(model, subject_df, saved_df)
        permutation = np.asarray(json.loads(row.perm), dtype=int)
        signs = np.asarray(json.loads(row.signs), dtype=float)
        audit = check_trials(regenerated_df, saved_df, permutation, signs)
        info = json.loads(source_paths["info"].read_text())
        info["alignment"] = dict(reference_model="GRU_population_medoid_model.pickle",
                                 source_run=alignment.SOURCE_RUN,
                                 source_example_model=alignment.relative(source_paths["model"]),
                                 source_example_sha256=mv.sha256(source_paths["model"]),
                                 perm=permutation.tolist(), signs=signs.tolist(),
                                 label_rule=alignment.LABEL_RULE,
                                 original_training_metrics="Retained from source; no retraining. Signed permutations preserve parameter L1/L2 norms and squared hidden-state energy.")
        folder = output_dir / subject / "GRU" / f"outer_fold_{row.outer_loop_n}"
        folder.mkdir(parents=True, exist_ok=False)
        info_path = folder / f"{row.model_id}_info.json"
        model_path = folder / f"{row.model_id}_model.pickle"
        trial_path = folder / f"{row.model_id}_trials_data.htsv"
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        regenerated_df.to_csv(trial_path, sep="\t", index=False)
        reread_df = pd.read_csv(trial_path, sep="\t")
        numerical = ["hidden_1", "hidden_2", "gate_update_1", "gate_update_2", "gate_reset_1", "gate_reset_2",
                     "logit_value", "prob_A", "prob_B", "logit_past", "logit_change"]
        np.testing.assert_array_equal(reread_df[numerical].to_numpy(dtype=np.float32),
                                      regenerated_df[numerical].to_numpy(dtype=np.float32))
        check_trials(reread_df, saved_df, permutation, signs)
        restored = alignment.load_model(model_path)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        entry = rows_df.loc[row.Index].to_dict()
        entry.update(example_info_path=alignment.relative(info_path), example_model_pickle_path=alignment.relative(model_path),
                     example_trials_data_path=alignment.relative(trial_path))
        exports.append(entry)
        audit.update(subject_id=subject, outer_loop_n=int(row.outer_loop_n), inner_loop_idx=int(row.inner_loop_idx),
                     n_trials=len(saved_df), n_sessions=saved_df.session_folder_name.nunique())
        audits.append(audit)
        for path in (info_path, model_path, trial_path):
            artifacts.append(dict(path=str(path.relative_to(output_dir)), bytes=path.stat().st_size, sha256=mv.sha256(path)))
    print(f"Exported {subject}: ten action-aligned GRUs and regenerated continuous trial tables", flush=True)
    return dict(exports=exports, audits=audits, artifacts=artifacts, session_hashes=session_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=ROOT / f"data/analyses/GRU_alignment/{alignment.SOURCE_RUN}/GRU_tanh")
    parser.add_argument("--output", type=Path, default=ROOT / "data/rnns/subject_examples_aligned")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output_dir = args.output.resolve()
    if (output_dir / "GRU_export_manifest.json").exists() or any((output_dir / subject / "GRU").exists() for subject in alignment.SUBJECTS):
        raise FileExistsError("GRU exports are immutable; choose a new output or inspect the existing build")
    frozen_path = args.alignment_dir / "manifest.json"
    frozen = json.loads(frozen_path.read_text())
    source_dir = alignment.resolve(frozen["source_dir"])
    source_df, hashes = alignment.load_sources(source_dir)
    if hashes != frozen["source_artifact_sha256"]:
        raise ValueError("Frozen alignment source artifacts changed")
    for item in frozen["artifacts"]:
        path = args.alignment_dir / item["path"]
        if path.stat().st_size != item["bytes"] or mv.sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen alignment artifact changed: {path}")
    for path, digest in frozen["code_sha256"].items():
        if mv.sha256(alignment.resolve(path)) != digest:
            raise ValueError(f"Alignment implementation changed since reference construction: {path}")
    rows_df = pd.read_csv(args.alignment_dir / "aligned_models.htsv", sep="\t")
    identity = ["subject_id", "outer_loop_n", "inner_loop_idx", "model_id", "selection_score"]
    pd.testing.assert_frame_equal(rows_df[identity], source_df[identity])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs)(delayed(export_subject)(subject, group, output_dir, source_dir, frozen)
                                            for subject, group in rows_df.groupby("subject_id", sort=True))
    exports_df = pd.DataFrame([entry for result in results for entry in result["exports"]])
    audits_df = pd.DataFrame([entry for result in results for entry in result["audits"]])
    exports_df.to_csv(output_dir / "GRU_outer_fold_winners.htsv", sep="\t", index=False)
    audits_df.to_csv(output_dir / "GRU_regeneration_checks.htsv", sep="\t", index=False)
    models = [alignment.load_model(path) for path in exports_df.example_model_pickle_path]
    reference = alignment.load_model(args.alignment_dir / "reference_model.pickle")
    with (output_dir / "GRU_population_medoid_model.pickle").open("wb") as handle:
        pickle.dump(reference, handle)
    mean_model, parameters_df = mono_export.average_parameters(models, exports_df)
    with (output_dir / "GRU_average_aligned_model.pickle").open("wb") as handle:
        pickle.dump(mean_model, handle)
    parameters_df.to_csv(output_dir / "GRU_average_parameters.htsv", sep="\t", index=False)
    lines = ["# Average action-aligned GRU parameters", "",
             "Equal mean of 11 subject means, each averaging ten aligned outer winners.",
             "Subject SD/SEM use subjects as the descriptive unit; folds share observations.",
             "The parameter-mean model is descriptive, not a fitted network or an ensemble.", "",
             "| Parameter | Stored index | Mean | Subject SD | Subject SEM |", "|---|---|---:|---:|---:|"]
    for row in parameters_df.itertuples():
        lines.append(f"| {row.parameter} | {row.index} | {row.mean:.7f} | {row.subject_sd:.7f} | {row.subject_sem:.7f} |")
    (output_dir / "GRU_average_parameters.md").write_text("\n".join(lines) + "\n")
    metadata = dict(source_run=alignment.SOURCE_RUN, model_type="GRU", nonlinearity="tanh", decoder_bias=True,
                    n_models=len(exports_df), n_subjects=exports_df.subject_id.nunique(), subjects=alignment.SUBJECTS,
                    reference=frozen["reference_source"], label_rule=alignment.LABEL_RULE,
                    alignment=frozen["transformation"], reference_rule=frozen["reference_rule"],
                    reference_orientation=frozen["reference_orientation"],
                    trial_data="Regenerated from exact raw sessions using monoGRU's load_subject_data/regenerate_trials and native mechanistic_variability._run_model_on_trials; continuous whole-subject history, original splits, no shifts or resets",
                    selection="Original published block-reset train+validation winners; all ten outer folds retained, exact inner indices, no re-selection",
                    parameter_average="Equal subject mean after within-subject mean of ten fits; all parameters and decoder bias included; descriptive network",
                    statistical_unit="Subjects for descriptive parameter SD/SEM; no inference. Trials remain separate within each exact fitted model.",
                    exclusions="None; all 110 requested fits and all continuous source trial rows retained",
                    execution=dict(requested_jobs=args.jobs, backend="loky", worker_threads=1,
                                   rollout_seconds=time.perf_counter() - started),
                    alignment_manifest_path=alignment.relative(frozen_path), alignment_manifest_sha256=mv.sha256(frozen_path),
                    source_artifact_sha256=hashes,
                    subject_data_sha256={path: digest for result in results for path, digest in result["session_hashes"].items()},
                    code_sha256={alignment.relative(path): mv.sha256(path) for path in
                                 [__file__, alignment.__file__, mono_export.__file__, mv.__file__, ds.__file__, rnns.__file__, save_data.__file__]},
                    maxima=audits_df.filter(like="max_").max().to_dict(),
                    artifacts=[entry for result in results for entry in result["artifacts"]])
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name.startswith("GRU_"):
            metadata["artifacts"].append(dict(path=path.name, bytes=path.stat().st_size, sha256=mv.sha256(path)))
    (output_dir / "GRU_export_manifest.json").write_text(json.dumps(metadata, indent=2, default=alignment.mono.json_value) + "\n")
    print(json.dumps(dict(output=str(output_dir), n_models=len(exports_df), parameter_entries=len(parameters_df),
                          maxima=metadata["maxima"])), flush=True)


if __name__ == "__main__":
    main()
