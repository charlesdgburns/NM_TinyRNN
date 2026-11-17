"""Audit the completed GRU addition independently from serialized artifacts."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from NM_TinyRNN.code.analyses import GRU_alignment as alignment
from NM_TinyRNN.code.measures import mechanistic_variability as mv


def main():
    output_dir = ROOT / "data/rnns/subject_examples_aligned"
    manifest_path = output_dir / "GRU_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        path = output_dir / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or mv.sha256(path) != artifact["sha256"]:
            raise ValueError(f"Changed serialized artifact: {path}")
    for path, digest in manifest["code_sha256"].items():
        if mv.sha256(ROOT / path) != digest:
            raise ValueError(f"Changed export implementation: {path}")
    source_dir = ROOT / "data/rnns/subject_examples_latest"
    sources_df, hashes = alignment.load_sources(source_dir)
    if hashes != manifest["source_artifact_sha256"]:
        raise ValueError("Source artifacts differ from the completed release")
    exported_df = pd.read_csv(output_dir / "GRU_outer_fold_winners.htsv", sep="\t")
    ledger_df = pd.read_csv(output_dir / "GRU_regeneration_checks.htsv", sep="\t")
    keys = ["subject_id", "outer_loop_n", "inner_loop_idx", "model_id", "selection_score"]
    pd.testing.assert_frame_equal(exported_df[keys], sources_df[keys])
    counts, min_h1, max_h2 = [], np.inf, -np.inf
    for row in exported_df.itertuples():
        model = alignment.load_model(row.example_model_pickle_path)
        contrast = alignment.decoder_contrast(model)
        if not (contrast[0] > 1e-7 and contrast[1] < -1e-7):
            raise ValueError(f"Wrong action labels: {row.subject_id}, outer {row.outer_loop_n}")
        min_h1, max_h2 = min(min_h1, float(contrast[0])), max(max_h2, float(contrast[1]))
        paths = alignment.source_paths(row, source_dir)
        original_df = pd.read_csv(paths["trials"], sep="\t")
        aligned_df = pd.read_csv(ROOT / row.example_trials_data_path, sep="\t")
        perm, signs = json.loads(row.perm), np.asarray(json.loads(row.signs))
        for columns in (["hidden_1", "hidden_2"],):
            np.testing.assert_allclose(aligned_df[columns], original_df[columns].to_numpy()[:, perm] * signs,
                                       atol=5e-5, rtol=2e-4)
        for gate in ("reset", "update"):
            columns = [f"gate_{gate}_1", f"gate_{gate}_2"]
            np.testing.assert_allclose(aligned_df[columns], original_df[columns].to_numpy()[:, perm],
                                       atol=5e-5, rtol=2e-4)
        np.testing.assert_allclose(aligned_df[["logit_value", "prob_A", "prob_B"]],
                                   original_df[["logit_value", "prob_A", "prob_B"]], atol=5e-5, rtol=2e-4)
        pd.testing.assert_frame_equal(aligned_df[["session_folder_name", "session_trial_idx", "split"]],
                                      original_df[["session_folder_name", "session_trial_idx", "split"]])
        record = ledger_df.loc[ledger_df.subject_id.eq(row.subject_id) & ledger_df.outer_loop_n.eq(row.outer_loop_n)].iloc[0]
        if len(aligned_df) != record.n_trials:
            raise ValueError("Serialized row count differs from regeneration ledger")
        info = json.loads((ROOT / row.example_info_path).read_text())
        original_info = json.loads(paths["info"].read_text())
        if {key: value for key, value in info.items() if key != "alignment"} != original_info:
            raise ValueError("Original fit metadata changed")
        counts.append(dict(subject_id=row.subject_id, outer_loop_n=int(row.outer_loop_n),
                           inner_loop_idx=int(row.inner_loop_idx), n_trials=len(aligned_df),
                           n_sessions=aligned_df.session_folder_name.nunique()))
    parameters_df = pd.read_csv(output_dir / "GRU_average_parameters.htsv", sep="\t")
    if len(parameters_df) != 44 or not parameters_df.n_subjects.eq(11).all() or not parameters_df.n_fits.eq(110).all():
        raise ValueError("Incorrect parameter summary coverage")
    report = dict(n_models=len(exported_df), n_subjects=exported_df.subject_id.nunique(),
                  artifacts_verified=len(manifest["artifacts"]), all_source_hashes_unchanged=True,
                  all_action_labels_valid=True, min_A1_h1_contrast=min_h1, max_A1_h2_contrast=max_h2,
                  trial_rows_across_separate_fits=sum(row["n_trials"] for row in counts),
                  unique_trial_rows=sum(row["n_trials"] for row in counts if row["outer_loop_n"] == 1),
                  parameter_entries=len(parameters_df), maxima=manifest["maxima"],
                  export_manifest_sha256=mv.sha256(manifest_path), fit_counts=counts)
    path = ROOT.parent / "tests/release_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "fit_counts"}), flush=True)


if __name__ == "__main__":
    main()
