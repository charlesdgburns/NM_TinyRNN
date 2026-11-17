"""Synthetic tests of the pinned native tanh GRU alignment engine.

These tests use generated parameters and binary inputs only. They do not load
fitted models, choose a cohort reference or change any scientific data.
"""

from copy import deepcopy
import itertools
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import torch

SOURCE_PARENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_PARENT))
from NM_TinyRNN.code.measures import gru_alignment_constraints as mv
from NM_TinyRNN.code.models.rnns import TinyRNN


def explicit_signed_permutation(model, permutation, signs):
    """Apply the independently written row-vector GRU symmetry formulas."""
    transformed = deepcopy(model)
    indices = list(permutation)
    sign = torch.tensor(signs, dtype=torch.float32)
    with torch.no_grad():
        transformed.rnn.W_ih.copy_(model.rnn.W_ih[:, indices] * sign)
        transformed.rnn.W_hh.copy_(model.rnn.W_hh[indices][:, indices] * sign[:, None] * sign)
        transformed.rnn.bias_h.copy_(model.rnn.bias_h[indices] * sign)
        transformed.rnn.hidden_0.copy_(model.rnn.hidden_0[indices] * sign)
        for gate in ("z", "r"):
            getattr(transformed.rnn, f"W_i{gate}").copy_(getattr(model.rnn, f"W_i{gate}")[:, indices])
            getattr(transformed.rnn, f"W_h{gate}").copy_(
                getattr(model.rnn, f"W_h{gate}")[indices][:, indices] * sign[:, None]
            )
            getattr(transformed.rnn, f"bias_{gate}").copy_(getattr(model.rnn, f"bias_{gate}")[indices])
        transformed.decoder.weight.copy_(model.decoder.weight[:, indices] * sign)
    return transformed


class TestGRUSignedPermutation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = TinyRNN(
            rnn_type="GRU", input_size=3, hidden_size=2, out_size=2,
            nonlinearity="tanh", decoder_bias=True, input_encoding="unipolar",
            input_forced_choice=True, weight_seed=104,
        ).cpu().float().eval()
        generator = torch.Generator().manual_seed(20261117)
        with torch.no_grad():
            for parameter in cls.model.parameters():
                parameter.copy_(0.4 * torch.randn(parameter.shape, generator=generator))
            cls.model.rnn.hidden_0.copy_(torch.tensor([0.35, -0.62]))
            cls.model.decoder.bias.copy_(torch.tensor([0.27, -0.41]))
        cls.inputs = torch.randint(0, 2, (6, 64, 3), generator=generator).float()

    def test_all_eight_transformations_and_pickle(self):
        maxima = dict(logit=0.0, probability=0.0, hidden=0.0, gate=0.0, parameter=0.0)
        with torch.no_grad():
            base_logits, base_hidden = self.model(self.inputs)
            _, base_gates = self.model.rnn(self.inputs, return_gate_activations=True)
        for permutation in itertools.permutations(range(2)):
            for signs in itertools.product((1.0, -1.0), repeat=2):
                with self.subTest(permutation=permutation, signs=signs):
                    reference = explicit_signed_permutation(self.model, permutation, signs)
                    aligned, audit = mv.align_rnns(reference, self.model, nonlinearity="tanh")
                    np.testing.assert_array_equal(audit["perm"], permutation)
                    np.testing.assert_array_equal(audit["transform_vec"], signs)
                    self.assertLess(audit["aligned_distance"], 1e-6)
                    expected_hidden = base_hidden[..., list(permutation)] * torch.tensor(signs)
                    for name, parameter in aligned.state_dict().items():
                        torch.testing.assert_close(parameter, reference.state_dict()[name], atol=1e-7, rtol=1e-6)
                        maxima["parameter"] = max(
                            maxima["parameter"], float((parameter - reference.state_dict()[name]).abs().max())
                        )
                    with torch.no_grad():
                        logits, hidden = aligned(self.inputs)
                        _, gates = aligned.rnn(self.inputs, return_gate_activations=True)
                    torch.testing.assert_close(logits, base_logits, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(hidden, expected_hidden, atol=1e-6, rtol=1e-5)
                    maxima["logit"] = max(maxima["logit"], float((logits - base_logits).abs().max()))
                    maxima["probability"] = max(
                        maxima["probability"], float((logits.softmax(-1) - base_logits.softmax(-1)).abs().max())
                    )
                    maxima["hidden"] = max(maxima["hidden"], float((hidden - expected_hidden).abs().max()))
                    for gate in ("reset", "update"):
                        expected_gate = base_gates[gate][..., list(permutation)]
                        torch.testing.assert_close(gates[gate], expected_gate, atol=1e-6, rtol=1e-5)
                        maxima["gate"] = max(maxima["gate"], float((gates[gate] - expected_gate).abs().max()))
                    restored = pickle.loads(pickle.dumps(aligned))
                    with torch.no_grad():
                        restored_logits, restored_hidden = restored(self.inputs)
                    torch.testing.assert_close(restored_logits, logits, atol=0, rtol=0)
                    torch.testing.assert_close(restored_hidden, hidden, atol=0, rtol=0)
                    torch.testing.assert_close(aligned.decoder.bias, self.model.decoder.bias, atol=0, rtol=0)
        print("Synthetic maxima across eight signed permutations:", maxima, flush=True)

    def test_source_model_is_unchanged(self):
        original = {key: value.clone() for key, value in self.model.state_dict().items()}
        reference = explicit_signed_permutation(self.model, (1, 0), (-1.0, 1.0))
        mv.align_rnns(reference, self.model, nonlinearity="tanh")
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
