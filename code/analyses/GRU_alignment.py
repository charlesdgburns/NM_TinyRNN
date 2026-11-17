"""Action-labeled tanh GRU outer winners and an equal-subject population medoid.

Source: nested_DA_S32_B8, all 11 published mice and their ten exact two-unit
GRU block-reset train+validation winners. Rows are fitted models; no new fit
selection, trial aggregation, exclusions or inferential analysis. Exported
readouts are continuous after-input predictions over full subject histories.
"""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from NM_TinyRNN.code.analyses import monoGRU_alignment as mono
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures import gru_alignment_constraints as constraints
from NM_TinyRNN.code.measures.monogru_balancing import core, parameters, transform

SOURCE_RUN = "nested_DA_S32_B8"
SUBJECTS = ("WS01", "WS02", "WS05", "WS08", "WS09", "WS10", "WS13", "WS14", "WS16", "WS20", "WS22")
LABEL_RULE = "decoder[0,h1]-decoder[1,h1] > 1e-7 (A1/left/0); decoder[0,h2]-decoder[1,h2] < -1e-7 (A2/right/1)"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_model(path):
    with resolve(path).open("rb") as handle:
        model = mv.CPUUnpickler(handle).load().cpu().float().eval()
    validate_model(model)
    return model


def validate_model(model):
    if (model.rnn_type, model.H, model.O, model.nonlinearity) != ("GRU", 2, 2, "tanh"):
        raise ValueError("Expected a two-unit, two-output tanh GRU")
    if not model.decoder_bias or model.decoder.bias is None or model.batch_norm:
        raise ValueError("Expected the published bias-on GRU without batch normalization")
    if model.input_encoding != "unipolar" or not model.input_forced_choice or model.rnn.W_ih.shape != (3, 2):
        raise ValueError("Expected the published three-input unipolar GRU")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite source parameters")
    contrast = decoder_contrast(model)
    if np.any(np.abs(contrast) <= 1e-7):
        raise ValueError("Near-zero decoder contrast cannot support the requested two action labels")


def decoder_contrast(model):
    return (model.decoder.weight[0] - model.decoder.weight[1]).detach().cpu().numpy()


def semantic_transformations(model):
    """The two signed permutations satisfying the required A1/A2 decoder signs.

    Prefer fewer sign flips only to fix the reference's global orientation;
    exact ties use native permutation enumeration. Alignment to that reference
    minimizes parameter distance over BOTH candidates, without this preference.
    """
    contrast = decoder_contrast(model)
    if np.any(np.abs(contrast) <= 1e-7):
        raise ValueError("Near-zero decoder contrast: cannot assign both action labels")
    candidates = []
    for permutation in ((0, 1), (1, 0)):
        signs = np.sign(contrast[list(permutation)]) * np.array([1., -1.])
        candidates.append((permutation, tuple(signs.tolist())))
    return sorted(candidates, key=lambda item: (sum(s < 0 for s in item[1]), item[0]))


def align_model(model, reference):
    """Native exhaustive alignment with action-label constraints kept hard."""
    validate_model(model)
    validate_model(reference)
    aligned, audit = constraints.align_rnns(reference, model, nonlinearity="tanh",
                                 allowed_transformations=semantic_transformations(model))
    contrast = decoder_contrast(aligned)
    if not (contrast[0] > 1e-7 and contrast[1] < -1e-7):
        raise RuntimeError("Native alignment violated the required action labels")
    return aligned, audit


def check_equivalence(original, aligned, permutation, signs):
    """Reuse monoGRU's fixed synthetic histories, checking both GRU gate vectors."""
    inputs = mono.probes(original)
    with torch.no_grad():
        logits_a, hidden_a = original(inputs)
        logits_b, hidden_b = aligned(inputs)
        _, gates_a = original.rnn(inputs, return_gate_activations=True)
        _, gates_b = aligned.rnn(inputs, return_gate_activations=True)
    expected = hidden_a[..., list(permutation)] * torch.tensor(signs, dtype=hidden_a.dtype)
    torch.testing.assert_close(logits_b, logits_a, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(hidden_b, expected, atol=2e-4, rtol=2e-4)
    record = dict(max_logit_error=float((logits_b - logits_a).abs().max()),
                  max_probability_error=float((logits_b.softmax(-1) - logits_a.softmax(-1)).abs().max()),
                  max_hidden_error=float((hidden_b - expected).abs().max()),
                  min_unit_rms=float(hidden_b.square().mean((0, 1)).sqrt().min()))
    for gate in ("update", "reset"):
        expected_gate = gates_a[gate][..., list(permutation)]
        torch.testing.assert_close(gates_b[gate], expected_gate, atol=2e-4, rtol=2e-4)
        record[f"max_{gate}_gate_error"] = float((gates_b[gate] - expected_gate).abs().max())
    torch.testing.assert_close(aligned.decoder.bias, original.decoder.bias, atol=0, rtol=0)
    return record


def semantic_core_vectors(model):
    source = parameters(model)
    return np.stack([core(transform(source, permutation, np.asarray(signs)), 2)
                     for permutation, signs in semantic_transformations(model)])


def choose_medoid(vectors, rows_df, eligible):
    """Equal-subject mean Euclidean core distance, minimizing allowed symmetry.

    Each model has two valid label-preserving coordinate versions, related by
    simultaneous negation and swapping. That orthogonal symmetry leaves the
    medoid objective invariant to its global orientation. Use version zero to
    freeze that remaining global convention, then align every fit to it.
    """
    distances = np.minimum(cdist(vectors[:, 0], vectors[:, 0]), cdist(vectors[:, 0], vectors[:, 1]))
    np.testing.assert_allclose(distances, distances.T, atol=1e-10, rtol=1e-10)
    weights = mono.cohort_weights(rows_df)
    scores = distances @ weights
    candidates = np.where(eligible, scores, np.inf)
    if not np.isfinite(candidates).any():
        raise ValueError("No two-active-unit reference candidates")
    return int(np.argmin(candidates)), scores, distances


def source_paths(row, source_dir):
    prefix = Path(source_dir) / row.subject_id / "GRU" / f"outer_fold_{row.outer_loop_n}" / row.model_id
    return {kind: Path(str(prefix) + suffix) for kind, suffix in
            (("info", "_info.json"), ("model", "_model.pickle"), ("trials", "_trials_data.htsv"))}


def load_sources(source_dir):
    """Verify release artifacts and preserve the exact published 11-by-10 grid."""
    source_dir = Path(source_dir).resolve()
    manifest = json.loads((source_dir / "export_manifest.json").read_text())
    winners_path = source_dir / "outer_fold_winners.htsv"
    if manifest["source_run"] != SOURCE_RUN or mv.sha256(winners_path) != manifest["outer_fold_winners_sha256"]:
        raise ValueError("Source run or published winner-table hash mismatch")
    rows_df = pd.read_csv(winners_path, sep="\t")
    rows_df = rows_df.loc[rows_df.model_type.eq("GRU")].sort_values(["subject_id", "outer_loop_n"]).reset_index(drop=True)
    if (set(rows_df.subject_id) != set(SUBJECTS) or len(rows_df) != 110
            or rows_df.duplicated(["subject_id", "outer_loop_n"]).any()
            or not rows_df.groupby("subject_id").outer_loop_n.apply(lambda x: sorted(x) == list(range(1, 11))).all()):
        raise ValueError("Expected all 11 published subjects and ten exact GRU winners each")
    if (not rows_df.selection_method.eq("block_reset").all() or not rows_df.hidden_size.eq(2).all()
            or not rows_df.nonlinearity.eq("tanh").all() or not rows_df.decoder_bias.all()
            or not rows_df.sequence_length.eq(32).all() or not rows_df.batch_size.eq(8).all()
            or not rows_df.model_pickle_path.map(lambda p: SOURCE_RUN in Path(p).parts).all()):
        raise ValueError("Wrong GRU configuration, source run or winner selection")
    np.testing.assert_allclose(rows_df.selection_score, rows_df.block_train_val_CE, atol=1e-12, rtol=1e-12)
    artifacts = {a["destination"].split("/subject_examples_latest/", 1)[1]: a for a in manifest["artifacts"]}
    hashes = {}
    for row in rows_df.itertuples():
        for path in source_paths(row, source_dir).values():
            artifact = artifacts[str(path.relative_to(source_dir))]
            digest = mv.sha256(path)
            if path.stat().st_size != artifact["bytes"] or digest != artifact["sha256"]:
                raise ValueError(f"Changed source artifact: {path}")
            hashes[relative(path)] = digest
    return rows_df, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/rnns/subject_examples_latest")
    parser.add_argument("--output", type=Path, default=ROOT / f"data/analyses/GRU_alignment/{SOURCE_RUN}/GRU_tanh")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Alignment output is immutable; choose a new empty --output")
    rows_df, source_hashes = load_sources(args.source)
    models = [load_model(source_paths(row, args.source)["model"]) for row in rows_df.itertuples()]
    vectors = np.stack([semantic_core_vectors(model) for model in models])
    active = []
    for model in models:
        with torch.no_grad():
            _, hidden = model(mono.probes(model))
        active.append(float(hidden.square().mean((0, 1)).sqrt().min()) > 1e-6)
    winner, scores, distances = choose_medoid(vectors, rows_df, np.asarray(active))
    permutation, signs = semantic_transformations(models[winner])[0]
    reference, _ = constraints.align_rnns(models[winner], models[winner], nonlinearity="tanh",
                               allowed_transformations=[(permutation, signs)])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "reference_model.pickle").open("wb") as handle:
        pickle.dump(reference, handle)
    audits, aligned_vectors = [], []
    for index, (row, model) in enumerate(zip(rows_df.itertuples(), models)):
        aligned, info = align_model(model, reference)
        audit = check_equivalence(model, aligned, info["perm"], info["transform_vec"])
        path = args.output / "aligned" / row.subject_id / row.model_id / f"outer_fold_{row.outer_loop_n}.pickle"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(aligned, handle)
        restored = load_model(path)
        for key, value in aligned.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
        contrast = decoder_contrast(aligned)
        audit.update(perm=json.dumps(info["perm"].tolist()), signs=json.dumps(info["transform_vec"].tolist()),
                     contrast_A1_h1=float(contrast[0]), contrast_A1_h2=float(contrast[1]),
                     reference_eligible=active[index], population_medoid_objective=float(scores[index]),
                     distance_to_reference=info["aligned_distance"], aligned_model_path=relative(path))
        audits.append(audit)
        aligned_vectors.append(core(parameters(aligned), 2))
    pd.concat([rows_df, pd.DataFrame(audits)], axis=1).to_csv(args.output / "aligned_models.htsv", sep="\t", index=False)
    sensitivity = []
    for subject in SUBJECTS:
        keep = rows_df.subject_id.ne(subject).to_numpy()
        local, _, _ = choose_medoid(vectors[keep], rows_df.loc[keep], np.asarray(active)[keep])
        selected = int(np.flatnonzero(keep)[local])
        sensitivity.append(dict(omitted_subject=subject, subject_id=rows_df.iloc[selected].subject_id,
                                outer_loop_n=int(rows_df.iloc[selected].outer_loop_n),
                                inner_loop_idx=int(rows_df.iloc[selected].inner_loop_idx),
                                distance_to_full_reference=float(distances[winner, selected])))
    pd.DataFrame(sensitivity).to_csv(args.output / "leave_subject_out.htsv", sep="\t", index=False)
    np.savez_compressed(args.output / "semantic_core_parameters.npz", semantic_candidates=vectors,
                        aligned_parameters=np.stack(aligned_vectors), pair_distances=distances)
    metadata = dict(source_run=SOURCE_RUN, source_dir=relative(args.source), n_models=110, subjects=SUBJECTS,
                    model_type="GRU", nonlinearity="tanh", decoder_bias=True, label_rule=LABEL_RULE,
                    transformation="Signed permutation only; all gain magnitudes are one; GRU-only constrained extension of native align_rnns",
                    reference_rule="Actual two-active-unit model minimizing equal-subject mean Euclidean core distance after constrained signed-permutation alignment to every cohort model",
                    reference_orientation="Fewest source-unit sign flips among the two valid action-labeled variants; stable permutation order breaks ties. Global orientation is a coordinate convention, not biological evidence.",
                    tie_rule="Stable subject/outer-fold row order breaks exact medoid-score ties",
                    objective_blocks=["rnn.W_ih", "rnn.bias_h", "rnn.W_hh", "decoder.weight"],
                    objective_exclusions=["gate parameters", "initial state", "decoder bias"],
                    reference_index=winner, reference_source=rows_df.iloc[winner].to_dict(),
                    reference_candidates=int(np.sum(active)), reference_sha256=mv.sha256(args.output / "reference_model.pickle"),
                    source_artifact_sha256=source_hashes,
                    source_winners_sha256=mv.sha256(args.source / "outer_fold_winners.htsv"),
                    validation="32 fixed synthetic binary histories of length 64, logits, probabilities, signed/permuted hidden states and permuted update/reset gates; saved model state_dict roundtrip",
                    scope="Descriptive reference estimated from this entire cohort; no held-out mechanistic inference or retraining",
                    code_sha256={relative(p): mv.sha256(p) for p in
                                 [__file__, mv.__file__, constraints.__file__, mono.__file__, sys.modules[transform.__module__].__file__]})
    metadata["artifacts"] = [dict(path=str(path.relative_to(args.output)), bytes=path.stat().st_size,
                                  sha256=mv.sha256(path)) for path in sorted(args.output.rglob("*")) if path.is_file()]
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, default=mono.json_value) + "\n")
    print(json.dumps(dict(output=str(args.output), reference=rows_df.iloc[winner][["subject_id", "outer_loop_n", "inner_loop_idx"]].to_dict(),
                          n_models=110, reference_candidates=int(np.sum(active))), default=mono.json_value), flush=True)


if __name__ == "__main__":
    main()
