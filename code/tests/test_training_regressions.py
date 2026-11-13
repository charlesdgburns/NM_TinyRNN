"""Run from the repository root: .venv/bin/python code/tests/test_training_regressions.py."""
import sys
import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.models import training
from NM_TinyRNN.code.models.datasets import get_dataloader
from NM_TinyRNN.code.models.rnns import TinyRNN


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def init_weights(self):
        with torch.no_grad():
            self.weight.fill_(1)

    def forward(self, x):
        return x * self.weight, x

    def compute_losses(self, predictions, **kwargs):
        return {'prediction': predictions.square().mean()}


class ToyDataset(TensorDataset):
    def __init__(self):
        x = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1, 1)
        super().__init__(x, torch.zeros_like(x), torch.zeros_like(x))

    def _session_split(self, **kwargs):
        return {'train': list(range(6)), 'val': [6, 7]}


class ValidationGRU(TinyRNN):
    def init_weights(self):
        super().init_weights()
        # Make losses distinguishable; tiny production weights are near log(2).
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if ('W' in name) or name == 'decoder.weight':
                    parameter.mul_(1000)


class UnevenValidationDataset(TensorDataset):
    def __init__(self, validation_mask):
        x = torch.tensor([[[1.], [2.]], [[3.], [4.]],
                          [[5.], [6.]], [[7.], [8.]]])
        y = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        mask = torch.cat([torch.zeros(1, 2), torch.tensor(validation_mask)])
        super().__init__(x, y, mask)

    def _session_split(self, **kwargs):
        return {'train': [0], 'val': [1, 2, 3]}


class TrainingRegressions(unittest.TestCase):
    def test_validation_is_mean_over_free_choices_independent_of_batching(self):
        # Unequal free-choice counts, a shorter final batch, and an all-forced batch.
        for masks in [[[0., 1.], [0., 0.], [1., 0.]],
                      [[1., 1.], [1., 1.], [0., 1.]]]:
            dataset = UnevenValidationDataset(masks)
            model = ValidationGRU(rnn_type='GRU', input_size=1, hidden_size=2, weight_seed=0)
            with torch.no_grad():
                logits, _ = model(dataset.tensors[0][1:])
                free = dataset.tensors[2][1:] == 0
                expected = torch.nn.functional.cross_entropy(
                    logits[free], dataset.tensors[1][1:][free]).item()
            for batch_size in [1, 2, 3]:
                trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0],
                                           energy_lambdas=[0], max_epochs=1,
                                           learning_rate=0, batch_size=batch_size)
                trainer.fit(model, dataset)
                self.assertAlmostEqual(float(trainer._last_best_val_loss), expected, places=6)

    def test_validation_without_free_choices_raises(self):
        dataset = UnevenValidationDataset([[1., 1.]] * 3)
        model = TinyRNN(rnn_type='GRU', input_size=1, hidden_size=2)
        trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0],
                                   energy_lambdas=[0], max_epochs=1, batch_size=2)
        with self.assertRaisesRegex(ValueError, 'no free-choice trials'):
            trainer.fit(model, dataset)

    def test_stopped_candidate_is_frozen_and_cannot_improve_again(self):
        snapshots = []
        validation = [[1., 4.], [2., 3.], [.5, 2.], [.1, 1.5]]
        real_vmap = training.vmap

        def instrument_vmap(*args, **kwargs):
            loss_fn = real_vmap(*args, **kwargs)

            def loss(params, *rest):
                if not torch.is_grad_enabled():
                    snapshots.append(params['weight'].detach().clone())
                    return {'prediction': torch.tensor(validation[len(snapshots)-1],
                                                       device=params['weight'].device)}
                return loss_fn(params, *rest)
            return loss

        trainer = training.Trainer(weight_seeds=[0, 1], sparsity_lambdas=[0],
                                   energy_lambdas=[0], batch_size=6,
                                   max_epochs=4, early_stop=1, learning_rate=.1)
        with patch.object(training, 'vmap', instrument_vmap):
            state, config, _ = trainer.fit(ToyModel(), ToyDataset())
        self.assertEqual(len(snapshots), 4)
        self.assertTrue(torch.equal(snapshots[1][0], snapshots[2][0]))
        self.assertTrue(torch.equal(snapshots[1][0], snapshots[3][0]))
        self.assertFalse(torch.equal(snapshots[1][1], snapshots[3][1]))
        self.assertEqual(float(trainer._last_best_val_loss), 1.)
        self.assertEqual(config[-1], 0)
        self.assertTrue(torch.equal(state['weight'], snapshots[0][0].cpu()))

    def test_shuffle_seed_reaches_loader_and_controls_order(self):
        orders = []
        for seed in [17, 17, 18]:
            seen = []

            def loader(*args, **kwargs):
                self.assertEqual(kwargs.get('seed', 0), seed)
                result = get_dataloader(*args, **kwargs)
                if args[1] == 'train':
                    seen.extend(torch.cat([x.flatten() for x, _, _ in result]).tolist())
                return result

            trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0],
                                       energy_lambdas=[0], max_epochs=1,
                                       batch_size=2, train_seed=99, shuffle_seed=seed)
            with patch.object(training, 'get_dataloader', loader):
                trainer.fit(ToyModel(), ToyDataset())
            orders.append(seen)
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0], orders[2])

    def test_default_shuffle_and_global_rng_use_train_seed(self):
        orders = []
        for fold_seed in [42, 420008]:
            seen = []
            def loader(*args, **kwargs):
                self.assertEqual(kwargs['seed'], fold_seed)
                result = get_dataloader(*args, **kwargs)
                if args[1] == 'train':
                    seen.extend(torch.cat([x.flatten() for x, _, _ in result]).tolist())
                return result
            trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0], energy_lambdas=[0],
                max_epochs=1, batch_size=2, train_seed=fold_seed)
            self.assertEqual(trainer.early_stop, 100)
            with patch.object(training, 'get_dataloader', loader):
                trainer.fit(ToyModel(), ToyDataset())
            self.assertEqual(torch.initial_seed(), fold_seed)
            orders.append(seen)
        self.assertNotEqual(orders[0], orders[1])

    def test_loader_default_seed_is_reproducible_despite_global_rng_changes(self):
        # This explains why the original code could pass reproducibility checks:
        # the loader's private seed 0 overrides the global RNG for shuffling.
        orders = []
        for global_seed in [42, 999]:
            torch.manual_seed(global_seed)
            loader = get_dataloader(ToyDataset(), 'train',
                                    ToyDataset()._session_split(), batch_size=2)
            orders.append([torch.cat([x.flatten() for x, _, _ in loader]).tolist()
                           for _ in range(2)])
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0][0], orders[0][1])

    def test_identical_training_calls_return_identical_results(self):
        results = []
        for global_seed in [123, 987]:
            torch.manual_seed(global_seed)
            trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0],
                                       energy_lambdas=[0], max_epochs=3,
                                       batch_size=2, train_seed=42)
            state, config, _ = trainer.fit(ToyModel(), ToyDataset())
            results.append((state, config, float(trainer._last_best_val_loss)))
        self.assertEqual(results[0][1:], results[1][1:])
        for key in results[0][0]:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))

    def test_parallel_training_matches_independent_adam_updates(self):
        # Check actual GRU/monoGRU updates independently of the vmap trainer.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor([[[1., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.]],
                          [[0., 1., 1.], [1., 0., 0.], [0., 0., 1.], [1., 1., 1.]],
                          [[1., 1., 0.], [0., 0., 1.], [1., 0., 1.], [0., 1., 0.]]])
        y = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
        mask = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.]])
        dataset = TensorDataset(x, y, mask)
        splits = {'train': [0, 1], 'val': [2]}
        dataset._session_split = lambda **kwargs: splits
        for architecture, activation in [('GRU', 'tanh'), ('monoGRU', 'relu'), ('lightGRU', 'relu')]:
            configs = list(itertools.product([0., 1e-4], [0., 1e-3], [0, 1]))
            manual = [TinyRNN(rnn_type=architecture, input_size=3, hidden_size=2,
                              nonlinearity=activation, weight_seed=seed).to(device) for _, _, seed in configs]
            optimizers = [torch.optim.AdamW(m.parameters(), lr=.01, weight_decay=0.) for m in manual]
            independent_steps = []
            for bx, by, bm in get_dataloader(dataset, 'train', splits, batch_size=1, seed=17):
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                candidates = []
                for model, optimizer, (sparsity, energy, _) in zip(manual, optimizers, configs):
                    optimizer.zero_grad()
                    prediction, hidden = model(bx)
                    losses = model.compute_losses(prediction, by, bm, dict(model.named_parameters()),
                        hidden, sparsity, energy, 0., 0.)
                    sum(losses.values()).backward()
                    candidates.append(dict(losses={k: v.detach().clone() for k, v in losses.items()},
                        gradients={k: p.grad.detach().clone() for k, p in model.named_parameters()}))
                    optimizer.step()
                independent_steps.append(candidates)
            snapshots = []
            parallel_steps = []
            real_vmap = training.vmap
            def capture(*args, **kwargs):
                fn = real_vmap(*args, **kwargs)
                def evaluate(params, *rest):
                    result = fn(params, *rest)
                    if torch.is_grad_enabled():
                        gradients = torch.autograd.grad(sum(v.sum() for v in result.values()),
                            tuple(params.values()), retain_graph=True)
                        parallel_steps.append(dict(losses={k: v.detach().clone() for k, v in result.items()},
                            gradients={k: g.detach().clone() for k, g in zip(params, gradients)}))
                    else:
                        snapshots.append({k: v.detach().clone() for k, v in params.items()})
                    return result
                return evaluate
            trainer = training.Trainer(weight_seeds=[0, 1], sparsity_lambdas=[0., 1e-4],
                energy_lambdas=[0., 1e-3], batch_size=1, max_epochs=1, learning_rate=.01, train_seed=99, shuffle_seed=17)
            base = TinyRNN(rnn_type=architecture, input_size=3, hidden_size=2, nonlinearity=activation)
            with patch.object(training, 'vmap', capture):
                trainer.fit(base, dataset)
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(len(parallel_steps), len(independent_steps))
            for actual, expected in zip(parallel_steps, independent_steps):
                for i, candidate in enumerate(expected):
                    for name, loss in candidate['losses'].items():
                        tolerance = 1e-7 if name == 'prediction' else 1e-11
                        self.assertTrue(torch.allclose(loss, actual['losses'][name][i], atol=tolerance, rtol=1e-5),
                                        (architecture, configs[i], name, 'loss'))
                    for name, gradient in candidate['gradients'].items():
                        self.assertTrue(torch.allclose(gradient, actual['gradients'][name][i], atol=2e-7, rtol=1e-4),
                                        (architecture, configs[i], name, 'gradient'))
            for i, model in enumerate(manual):
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.allclose(parameter, snapshots[0][name][i], atol=2e-6, rtol=1e-4),
                                    (architecture, i, name))

    def test_one_unit_covariance_is_zero_and_preserves_existing_gradients(self):
        model = TinyRNN(input_size=2, hidden_size=1)
        predictions, hidden = model(torch.ones(2, 4, 2))
        losses = model.compute_losses(predictions, torch.zeros(2, 4, dtype=torch.long),
            torch.zeros(2, 4), dict(model.named_parameters()), hidden, 0., 0., 0., 0.)
        self.assertEqual(float(losses['covariance']), 0.)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        # The old empty mean had a NaN value but zero parameter derivatives.
        centered = hidden.reshape(-1, 1) - hidden.reshape(-1, 1).mean(0)
        cov = centered.T @ centered / (hidden.numel()-1)
        old_covariance = cov[~torch.eye(1, dtype=torch.bool)].square().mean() * 0.
        before = torch.autograd.grad(total + old_covariance, tuple(model.parameters()), retain_graph=True)
        after = torch.autograd.grad(total, tuple(model.parameters()))
        for previous, current in zip(before, after):
            self.assertTrue(torch.equal(previous, current))

    def test_lstm_recurrent_weight_name_and_backward(self):
        model = TinyRNN(rnn_type='LSTM', input_size=3, hidden_size=2)
        self.assertIn('rnn.Wu', model.state_dict())
        self.assertNotIn('rnn.U', model.state_dict())
        predictions, _ = model(torch.ones(2, 4, 3))
        predictions.square().sum().backward()
        self.assertIsNotNone(model.rnn.Wu.grad)
        self.assertTrue(torch.isfinite(model.rnn.Wu.grad).all())

    def test_gru_and_lightgru_initial_state_is_learned_shared_and_can_be_overridden(self):
        for architecture in ['GRU', 'lightGRU']:
            model = TinyRNN(rnn_type=architecture, input_size=3, hidden_size=2)
            self.assertIn('rnn.hidden_0', dict(model.named_parameters()))
            self.assertTrue(torch.equal(model.rnn.hidden_0, torch.zeros(2)))
            x = torch.ones(3, 4, 3)
            initial, _ = model.rnn(x)
            predictions, _ = model(x)
            loss = torch.nn.functional.cross_entropy(predictions.reshape(-1, 2), torch.zeros(12, dtype=torch.long))
            loss.backward()
            self.assertIsNotNone(model.rnn.hidden_0.grad)
            self.assertTrue(torch.isfinite(model.rnn.hidden_0.grad).all())
            self.assertGreater(float(model.rnn.hidden_0.grad.abs().sum()), 0.)
            with torch.no_grad():
                model.rnn.hidden_0.fill_(.5)
            changed, _ = model.rnn(x)
            self.assertFalse(torch.allclose(changed, initial))
            self.assertTrue(torch.equal(changed[0], changed[1]))
            overridden, _ = model.rnn(x, init_states=torch.zeros(3, 2))
            self.assertTrue(torch.equal(initial, overridden))
            model.init_weights()
            self.assertTrue(torch.equal(model.rnn.hidden_0, torch.zeros(2)))
            del model.rnn.hidden_0
            historical, _ = model.rnn(x)
            self.assertTrue(torch.equal(initial, historical))


    def test_encoder_small_initialization_and_existing_sparsity_gradient(self):
        model = TinyRNN(rnn_type='monoGRU', hidden_size=2, input_size=3,
                        input_encoding='encoder', nonlinearity='relu')
        expected = model.encoder[0].weight.detach().clone()
        self.assertLessEqual(float(expected.abs().max()), .001001)
        self.assertTrue(torch.equal(model.encoder[0].bias, torch.zeros(3)))
        with torch.no_grad():
            model.encoder[0].weight.fill_(7)
            model.encoder[0].bias.fill_(7)
        model.init_weights()
        self.assertTrue(torch.equal(model.encoder[0].weight, expected))
        self.assertTrue(torch.equal(model.encoder[0].bias, torch.zeros(3)))
        predictions, hidden = model(torch.ones(2, 4, 3))
        losses = model.compute_losses(predictions, torch.zeros(2, 4, dtype=torch.long),
            torch.zeros(2, 4), dict(model.named_parameters()), hidden,
            sparsity_lambda=.1, energy_lambda=0., hebbian_lambda=0., covariance_lambda=0.)
        weight_grad, bias_grad = torch.autograd.grad(losses['sparsity'],
            (model.encoder[0].weight, model.encoder[0].bias), allow_unused=True)
        self.assertTrue(torch.allclose(weight_grad, .1 * model.encoder[0].weight.sign()))
        self.assertIsNone(bias_grad)

    def test_initial_states_and_weights_are_explicitly_reinitialised(self):
        model = TinyRNN(rnn_type='monoGRU', hidden_size=2)
        for name in ['hidden_0', 'z_0']:
            self.assertTrue(torch.equal(getattr(model.rnn, name),
                                        torch.zeros_like(getattr(model.rnn, name))))
        with torch.no_grad():
            model.rnn.hidden_0.fill_(7)
        model.init_weights()
        self.assertTrue(torch.equal(model.rnn.hidden_0, torch.zeros(2)))
        for architecture in ['vanilla', 'GRU', 'LSTM', 'monoGRU', 'stereoGRU',
                             'stereoGRU2', 'stereoGRUx', 'lightGRU', 'NMRNN']:
            model = TinyRNN(rnn_type=architecture, hidden_size=2)
            expected = {k: p.detach().clone() for k, p in model.named_parameters()}
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(7)
            model.init_weights()
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, expected[name]), (architecture, name))
                if name.rsplit('.', 1)[-1].startswith('W'):
                    self.assertLessEqual(float(p.detach().abs().max()), .001001)


if __name__ == '__main__':
    unittest.main()
