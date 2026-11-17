"""Constrained GRU alignment and equal-subject medoid regression tests."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from NM_TinyRNN.code.analyses import GRU_alignment as alignment
from NM_TinyRNN.code.analyses import export_aligned_GRU as export
from NM_TinyRNN.code.analyses import export_aligned_monoGRU as mono_export
from NM_TinyRNN.code.measures import gru_alignment_constraints as mv
from NM_TinyRNN.code.measures.monogru_balancing import core, parameters, transform
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.models.save_data import get_model_trial_by_trial_df


def model(seed, contrast):
    fitted = TinyRNN(rnn_type="GRU", input_size=3, hidden_size=2, out_size=2,
                     nonlinearity="tanh", decoder_bias=True, input_encoding="unipolar",
                     input_forced_choice=True, weight_seed=seed).cpu().float().eval()
    with torch.no_grad():
        fitted.rnn.hidden_0.copy_(torch.tensor([0.23, -0.41]))
        fitted.decoder.weight[0].copy_(torch.tensor(contrast))
        fitted.decoder.weight[1].zero_()
        fitted.decoder.bias.copy_(torch.tensor([0.17, -0.29]))
    return fitted


class TestConstrainedGRU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_decoder_sign_patterns_and_minimum(self):
        reference = model(11, [1.2, -0.8])
        for contrast in ((1.0, 2.0), (-1.0, -2.0), (1.0, -2.0), (-1.0, 2.0)):
            with self.subTest(contrast=contrast):
                source = model(23, contrast)
                aligned, audit = alignment.align_model(source, reference)
                contrasts = alignment.decoder_contrast(aligned)
                self.assertGreater(contrasts[0], 1e-7)
                self.assertLess(contrasts[1], -1e-7)
                source_parameters = parameters(source)
                target = core(parameters(reference), 2)
                candidates = [np.linalg.norm(core(transform(source_parameters, perm, np.asarray(signs)), 2) - target)
                              for perm, signs in alignment.semantic_transformations(source)]
                self.assertAlmostEqual(audit["aligned_distance"], min(candidates), places=5)
                alignment.check_equivalence(source, aligned, audit["perm"], audit["transform_vec"])

    def test_identity_fallback_cannot_undo_hard_signs(self):
        source = model(42, [1.0, 2.0])
        aligned, audit = mv.align_rnns(source, source, nonlinearity="tanh",
                                     allowed_transformations=[((0, 1), (1., -1.))])
        np.testing.assert_array_equal(audit["transform_vec"], [1., -1.])
        self.assertGreater(audit["aligned_distance"], 0)
        self.assertLess(alignment.decoder_contrast(aligned)[1], 0)

    def test_invalid_constraints_and_near_zero_contrast_raise(self):
        source = model(17, [1.0, -2.0])
        for candidates in ([], [((0, 0), (1., -1.))], [((0, 1), (0., 1.))]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                mv.align_rnns(source, source, nonlinearity="tanh", allowed_transformations=candidates)
        with self.assertRaises(ValueError):
            mv.align_rnns(source, source, nonlinearity="tanh", allowed_permutations=[(0, 1)],
                          allowed_transformations=[((1, 0), (1., -1.))])
        with self.assertRaises(ValueError):
            alignment.validate_model(model(7, [1e-9, -2.]))

    def test_medoid_weights_and_global_orientation_invariance(self):
        models = [model(31, [1.0, 2.0]), model(32, [-0.7, 1.4]), model(33, [0.8, -0.6])]
        rows = pd.DataFrame(dict(subject_id=["one", "one", "two"], decoder_bias=True))
        vectors = np.stack([alignment.semantic_core_vectors(item) for item in models])
        winner, scores, distances = alignment.choose_medoid(vectors, rows, np.ones(3, dtype=bool))
        np.testing.assert_allclose(scores, distances @ [.25, .25, .5])
        switched = vectors.copy()
        switched[[0, 2]] = switched[[0, 2], ::-1]
        other, alternate_scores, alternate_distances = alignment.choose_medoid(switched, rows, np.ones(3, dtype=bool))
        self.assertEqual(winner, other)
        np.testing.assert_allclose(scores, alternate_scores, atol=1e-12)
        np.testing.assert_allclose(distances, alternate_distances, atol=1e-12)
        # Replicating one subject's identical fit must not increase its total weight.
        repeated_rows = pd.DataFrame(dict(subject_id=["one", "one", "one", "one", "two"], decoder_bias=True))
        repeated = vectors[[0, 1, 0, 1, 2]]
        _, repeated_scores, _ = alignment.choose_medoid(repeated, repeated_rows, np.ones(5, dtype=bool))
        np.testing.assert_allclose(scores, repeated_scores[[0, 1, 4]], atol=1e-12)

    def test_regenerated_trials_preserve_both_permuted_gate_vectors(self):
        source = model(101, [-1.0, 2.0])
        reference = model(102, [1.2, -0.8])
        raw_df = pd.DataFrame(dict(forced_choice=[0, 0, 1, 0] * 3, choice=[0, 1, 0, 1] * 3,
                                   outcome=[1, 0, 1, 1] * 3, good_poke=[0] * 12,
                                   session_folder_name=["mouse/session"] * 12, session_trial_idx=range(12),
                                   sequence_block_idx=np.repeat([0., 1., 2.], 4), block_trial_idx=range(12)))
        saved_df = get_model_trial_by_trial_df(source, SimpleNamespace(subject_df=raw_df),
                                               dict(train=[0], val=[1], eval=[2]))
        aligned, audit = alignment.align_model(source, reference)
        regenerated_df = mono_export.regenerate_trials(aligned, raw_df, saved_df)
        errors = export.check_trials(regenerated_df, saved_df, audit["perm"], audit["transform_vec"])
        self.assertLess(max(errors.values()), 2e-6)
        broken_df = regenerated_df.copy()
        broken_df["gate_reset_1"] = -broken_df["gate_reset_1"]
        with self.assertRaises(AssertionError):
            export.check_trials(broken_df, saved_df, audit["perm"], audit["transform_vec"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
