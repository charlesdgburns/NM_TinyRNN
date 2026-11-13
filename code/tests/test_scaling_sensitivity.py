"""Check accelerated alignment and recovery of known continuous transformations."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from copy import deepcopy
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.scaling_sensitivity import parameters, transform, core, exponents, fit_pair, cosine


class ScalingSensitivity(unittest.TestCase):
    def test_baseline_matches_existing_alignment_including_gates(self):
        for kind in ('GRU', 'monoGRU', 'lightGRU'):
            for activation in ('relu', 'tanh'):
                a = TinyRNN(rnn_type=kind, input_size=3, hidden_size=2, nonlinearity=activation, decoder_bias=True, weight_seed=17)
                b = TinyRNN(rnn_type=kind, input_size=3, hidden_size=2, nonlinearity=activation, decoder_bias=True, weight_seed=31)
                aligned, info = mv.align_rnns(a, b)
                exact = transform(parameters(b), info['perm'], info['transform_vec'])
                for key, value in parameters(aligned).items():
                    np.testing.assert_allclose(exact[key], value, rtol=1e-5, atol=1e-7)
                result = fit_pair(parameters(a), parameters(b), activation)
                self.assertAlmostEqual(result['baseline_distance'], info['aligned_distance'], places=6)
                self.assertAlmostEqual(result['baseline_similarity'], mv.compute_weight_similarity(a, aligned), places=6)
                self.assertLessEqual(result['continuous_distance'], result['baseline_distance']+1e-10)

    def test_known_scale_outside_grid_is_recovered(self):
        for activation, scales in [('relu', [3.1, .31]), ('tanh', [-3.1, .31])]:
            model = TinyRNN(rnn_type='GRU', input_size=3, hidden_size=2, nonlinearity=activation, decoder_bias=True, weight_seed=17)
            b = parameters(model)
            a = transform(b, (1, 0), np.array(scales))
            result = fit_pair(a, b, activation)
            self.assertGreater(result['baseline_distance'], 1e-5)
            self.assertLess(result['continuous_distance'], 1e-7)
            self.assertAlmostEqual(result['continuous_similarity'], cosine(a, a), places=6)

    def test_relu_continuous_transform_preserves_predictions(self):
        torch.set_num_threads(1)
        for kind in ('GRU', 'monoGRU', 'lightGRU'):
            model = TinyRNN(rnn_type=kind, input_size=3, hidden_size=2, nonlinearity='relu', weight_seed=17)
            changed = deepcopy(model)
            values = transform(parameters(model), (1, 0), np.array([3.1, .31]))
            with torch.no_grad():
                for name, param in changed.named_parameters():
                    if name in values:
                        param.copy_(torch.as_tensor(values[name], dtype=param.dtype))
                x = torch.tensor([[[0., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 0., 0.]]])
                original = model.decoder(model.rnn(x)[0])
                aligned = changed.decoder(changed.rnn(x)[0])
            np.testing.assert_allclose(original.numpy(), aligned.numpy(), rtol=2e-5, atol=1e-7)

    def test_exponential_parameterization_matches_direct_transform(self):
        model = TinyRNN(rnn_type='GRU', input_size=3, hidden_size=2, nonlinearity='relu', weight_seed=17)
        p = parameters(model)
        signs = np.array([-1., 1.]); x = np.array([.7, -.9])
        coeff = core(transform(p, (1, 0), signs), 2)
        expected = core(transform(p, (1, 0), signs*np.exp(x)), 2)
        np.testing.assert_allclose(coeff*np.exp(exponents(p, 2)@x), expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
